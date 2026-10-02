import pandas as pd
from datasets import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, DataCollatorWithPadding
import torch
import evaluate

test_file = input("Enter test dataset path (CSV/Excel): ").strip().strip('"')
if not test_file:
    print("No file path provided. Exiting...")
    exit(1)
model_dir = "./sentiment_model"
max_length = 128
batch_size = 16

label_shift = {-1: 0, 0: 1, 1: 2}
reverse_shift = {0: -1, 1: 0, 2: 1}


# Read test file and auto-detect sentiment/label column
test_df = pd.read_csv(test_file, encoding="ISO-8859-1", on_bad_lines='skip')
if "sentiment" in test_df.columns:
    label_col = "sentiment"
elif "label" in test_df.columns:
    label_col = "label"
else:
    print(f"No 'sentiment' or 'label' column found. Columns: {list(test_df.columns)}")
    raise KeyError("No sentiment or label column found in test file.")
test_df[label_col] = test_df[label_col].replace("", pd.NA)
test_df = test_df.dropna(subset=[label_col])
# Strip whitespace from sentiment/label values
test_df[label_col] = test_df[label_col].astype(str).str.strip().str.lower()
label_map = {"negative": -1, "neutral": 0, "positive": 1}
if test_df[label_col].dtype == object:
    test_df["label"] = test_df[label_col].map(label_map)
test_df = test_df.dropna(subset=["label"])
test_df["label"] = test_df["label"].map(label_shift).astype(int)

test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir)
model = XLMRobertaForSequenceClassification.from_pretrained(model_dir)

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized["labels"] = examples["label"]
    return tokenized

test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove pandas index
if "__index_level_0__" in test_dataset.column_names:
    test_dataset = test_dataset.remove_columns(["__index_level_0__"])

test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    shifted_preds = [reverse_shift[p.item()] for p in predictions]
    shifted_labels = [reverse_shift[l.item()] for l in labels]
    return metric.compute(predictions=shifted_preds, references=shifted_labels)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

results = trainer.evaluate(test_dataset)
print("Test Results:", results)
