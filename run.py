from datasets import load_dataset
from models import SpeechToTextModel
from transformers import WhisperProcessor, TrainingArguments, Trainer, AutoTokenizer
from utils import LibriSpeechDataCollator

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Loading models...")

whisper_model_name = "openai/whisper-base"
llama_model_name = "meta-llama/Llama-3.2-3B"
dataset_name = "openslr/librispeech_asr"

model = SpeechToTextModel(
    whisper_model_name=whisper_model_name,
    llama_model_name=llama_model_name,
    hidden_dims=[2048, 1024, 2048, 1024, 2048],
    train_whisper=False,
    train_llama=False
)

print("Loading datasets...")

dataset = load_dataset(dataset_name, 'clean', split='train.100', streaming=True)

processor = WhisperProcessor.from_pretrained(whisper_model_name)
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

print("Training starts...")
model.train()

training_args = TrainingArguments(
    output_dir="./v1-checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    max_steps=10000,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    remove_unused_columns=False,
    learning_rate=1e-6,
    save_safetensors=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=LibriSpeechDataCollator(processor, tokenizer),
)

trainer.train()
trainer.save_model("./v1-checkpoints")