import torch
import torchaudio
from dataclasses import dataclass
from transformers import WhisperProcessor, PreTrainedTokenizer

@dataclass
class LibriSpeechDataCollator:
    whisper_processor: WhisperProcessor
    tokenizer: PreTrainedTokenizer
    separator_token_id: int = 128000

    def __call__(self, batch):
        audios = [sample['audio']['array'] for sample in batch]
        texts = [sample['text'] for sample in batch]

        # all libri speech are 16kHz
        audio_inputs = self.whisper_processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = audio_inputs.input_features  # size [B, 80, 1500]
        input_features = input_features.to(torch.bfloat16)


        batch_size, seq_audio, _ = input_features.shape
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        separator_token = torch.full((batch_size, 1), self.separator_token_id, dtype=input_ids.dtype)
        input_ids_prepended = torch.cat([separator_token, input_ids], dim=1)

        attend_to_separator = torch.full((batch_size, 1), 1, dtype=attention_mask.dtype)
        attention_mask_prepended = torch.cat([attend_to_separator, attention_mask], dim=1)

        labels = input_ids_prepended.clone()

        # print(f"input_features type {input_features.dtype}")
        # print(f"input_ids type {input_ids.dtype}")
        # print(f"attention_mask type {attention_mask.dtype}")
        # print(f"labels type {labels.dtype}")

        return {
            "input_features": input_features,
            "input_ids": input_ids_prepended,
            "attention_mask": attention_mask_prepended,
            "labels": labels
        }


@dataclass
class GPTVoiceAssistantDataCollator:
    whisper_processor: WhisperProcessor
    tokenizer: PreTrainedTokenizer

    resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
    separator_token_id: int = 128000
    base_index: int = 1616
    switch_every: int = 4

    def __call__(self, batch):
        input_features = []

        for i, sample in enumerate(batch):
            waveform = torch.tensor(sample["question_audio"]["array"]).float()

            sample_rate = sample["question_audio"]["sampling_rate"]
            if sample_rate != self.resampler.orig_freq:
                raise ValueError("Data is not sampled at the correct rate")

            # investigate the weird padding behavior when 'padding=True' for whisper_processor
            input_processed = self.whisper_processor(
                self.resampler(waveform),
                sampling_rate=self.resampler.new_freq,
                return_tensors="pt",
            ).input_features

            input_features.append(input_processed.squeeze(0))

        input_features = torch.stack(input_features, dim=0)

        is_task_answer_not_transcribe = (int(batch[0]['index']) - self.base_index) % self.switch_every == 0
        prepend = "Answer: " if is_task_answer_not_transcribe else "Transcribe: "
        column_name = "answer" if is_task_answer_not_transcribe else "question"

        input_ids, attention_mask = self.create_text_tokens(
            prepend_word=prepend,
            batch=batch,
            column_name=column_name
        )

        labels = input_ids.clone()

        return {
            "input_features": input_features,
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def create_text_tokens(self, prepend_word, batch, column_name):
        inputs = [f"{prepend_word}{sample[column_name]}" for sample in batch]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # add separator tokens
        batch_size = input_ids.size(0)
        separator_token = torch.full((batch_size, 1), self.separator_token_id, dtype=input_ids.dtype)
        input_ids_prepended = torch.cat([separator_token, input_ids], dim=1)

        # attend to the separator tokens
        attend_to_separator = torch.full((batch_size, 1), 1, dtype=attention_mask.dtype)
        attention_mask_prepended = torch.cat([attend_to_separator, attention_mask], dim=1)

        return input_ids_prepended, attention_mask_prepended