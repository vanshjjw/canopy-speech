from datasets import load_dataset
from models import SpeechToTextModel
from transformers import WhisperProcessor, TrainingArguments, Trainer, AutoTokenizer
from utils import GPTVoiceAssistantDataCollator


# hack
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


llama_model_name = "meta-llama/Llama-3.2-3B"
whisper_model_name = "openai/whisper-base"
ds_name = "gpt-omni/VoiceAssistant-400K"


print("Loading models...")

model = SpeechToTextModel(
    whisper_model_name=whisper_model_name,
    llama_model_name=llama_model_name,
    hidden_dims=[2048, 1024, 2048, 1024, 2048],
    train_whisper=True,
    train_llama=True,
)
processor = WhisperProcessor.from_pretrained(whisper_model_name)
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

print("Loading datasets...")
dataset = load_dataset(ds_name, split='train', streaming=True)

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

collator = GPTVoiceAssistantDataCollator(
    whisper_processor=processor,
    tokenizer=tokenizer,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

trainer.train()
trainer.save_model("./v1-checkpoints")