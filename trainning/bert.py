import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
import os

# Load dataset
data_train = pd.read_csv("dataset/raw/data_train_raw.csv")
data_test = pd.read_csv("dataset/raw/data_test_raw.csv")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["url"], padding="max_length", truncation=True)

# Convert pandas DataFrame to Hugging Face Dataset
dataset_train = Dataset.from_pandas(data_train)
dataset_test = Dataset.from_pandas(data_test)

tokenized_train = dataset_train.map(tokenize_function, batched=True)
tokenized_test = dataset_test.map(tokenize_function, batched=True)

tokenized_train = tokenized_train.remove_columns(["url"])
tokenized_test = tokenized_test.remove_columns(["url"])
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")

tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

trainer.train()

# Save model to folder "model"
os.makedirs("model", exist_ok=True)
model.save_pretrained("model")
tokenizer.save_pretrained("model")
