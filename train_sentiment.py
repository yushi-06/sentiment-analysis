import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# ------------------ CONFIG ------------------
model_name = "xlm-roberta-base"
train_file = "datasets/train.csv"
test_file = "datasets/test.csv"
max_length = 128
batch_size = 16
num_epochs = 3
output_dir = "./sentiment_model"

# ------------------ CLEAN CSV ------------------
for f in [train_file, test_file]:
    df = pd.read_csv(f)
    df["sentiment"] = df["sentiment"].replace("", pd.NA)
    df = df.dropna(subset=["sentiment"])
    # Strip whitespace from sentiment values
    df["sentiment"] = df["sentiment"].str.strip()
    # Convert sentiment labels to lowercase and numeric
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["sentiment"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df.to_csv(f, index=False)

# ------------------ LOAD DATASET ------------------
dataset = load_dataset("csv", data_files={"train": train_file, "test": test_file})
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized["labels"] = examples["label"]
    return tokenized

train_dataset = dataset["train"].map(preprocess, batched=True)
eval_dataset = dataset["test"].map(preprocess, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
