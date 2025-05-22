from dataclasses import dataclass
from transformers import WhisperProcessor, PreTrainedTokenizer
import torch

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

        return {
            "input_features": input_features,
            "input_ids": input_ids_prepended,
            "attention_mask": attention_mask_prepended,
            "labels": labels
        }


@dataclass
class VoiceAssistantDataCollator:
    whisper_processor: WhisperProcessor
    tokenizer: PreTrainedTokenizer
    separator_token_id: int = 128000

    def __call__(self, batch):
        pass