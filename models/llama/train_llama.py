import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from pathlib import Path

# Path to the dataset
input_file = Path("../../data/corpus/cleaned_corpus.csv")

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load dataset (Ensure it's preprocessed as per Step 4)
dataset = load_dataset(str(input_file))

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load pre-trained Llama model
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")

print("Llama fine-tuning complete!")
