import torch
import torch.nn.functional as F
# ------------------------------
# Advanced Loss Functions
# ------------------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
import pandas as pd
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate

# ------------------------------
# Load dataset
# ------------------------------
file_path = input("Enter training dataset path (CSV/Excel): ").strip().strip('"')
if not file_path:
    print("No file path provided. Exiting...")
    exit(1)

df = pd.read_excel(file_path)

# Auto-detect label column
if "sentiment" in df.columns:
    label_col = "sentiment"
elif "label" in df.columns:
    label_col = "label"
else:
    raise ValueError("CSV must contain a 'sentiment' or 'label' column.")

# Clean and normalize labels
df[label_col] = df[label_col].astype(str).str.strip().str.lower()

# Map labels if strings
label_map = {"negative": 0, "neutral": 1, "positive": 2}
if df[label_col].dtype == object or df[label_col].isin(label_map.keys()).all():
    df["label"] = df[label_col].map(label_map)
else:
    df["label"] = df[label_col].astype(int)


# Drop rows with invalid/missing labels
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Drop rows where text is not a string (e.g., NaN or numbers)
df = df[df["text"].apply(lambda x: isinstance(x, str))]

print("Label mapping:", label_map)
print("Label distribution:\n", df["label"].value_counts())

# ------------------------------

# ------------------------------
# Balance the training data by upsampling minority classes
# ------------------------------
from sklearn.utils import resample

# Split first, then balance only the training set
train_df, test_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df["label"]
)

# Upsample minority classes in train_df
max_count = train_df['label'].value_counts().max()
balanced_train_df = []
for label in train_df['label'].unique():
    subset = train_df[train_df['label'] == label]
    upsampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
    balanced_train_df.append(upsampled)
train_df = pd.concat(balanced_train_df)

print("Label distribution after upsampling:")
print(train_df['label'].value_counts())

train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

# ------------------------------
# Tokenizer & Model
# ------------------------------
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

# ------------------------------

# Adaptive Trainer: reweights samples after each epoch based on misclassification
from torch.utils.data import WeightedRandomSampler, DataLoader
from tqdm import tqdm

class AdaptiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = np.ones(len(self.train_dataset), dtype=np.float32)
        self.epoch = 0


    def get_train_dataloader(self):
        sampler = WeightedRandomSampler(self.sample_weights, len(self.train_dataset), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").to(self.args.device)
        outputs = model(**{k: v.to(self.args.device) for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        # Choose loss function: 'focal' or 'cross_entropy'
        loss_type = getattr(self.args, 'loss_type', 'focal')  # default to focal
        if loss_type == 'focal':
            loss_fct = FocalLoss(alpha=1, gamma=2)
            loss = loss_fct(logits, labels)
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def on_epoch_end(self):
        self.model.eval()
        all_logits = []
        all_labels = []
        dataloader = DataLoader(self.train_dataset, batch_size=32)
        for batch in tqdm(dataloader, desc=f"Adaptive weighting epoch {self.epoch}"):
            input_ids = batch["input_ids"].to(self.args.device)
            attention_mask = batch["attention_mask"].to(self.args.device)
            labels = batch["label"].to(self.args.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        preds = torch.argmax(logits, dim=-1)
        incorrect = (preds != labels).numpy()
        self.sample_weights = self.sample_weights * (1.2 * incorrect + 0.8 * (~incorrect))
        self.sample_weights = self.sample_weights / np.sum(self.sample_weights) * len(self.sample_weights)
        self.epoch += 1
        self.model.train()

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        super()._maybe_log_save_evaluate(*args, **kwargs)
        self.on_epoch_end()

# ------------------------------
# Metrics
# ------------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }

# ------------------------------
# Training Arguments (old version compatible)
# ------------------------------
class CustomTrainingArguments(TrainingArguments):
    def __init__(self, *args, loss_type='focal', **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type

training_args = CustomTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",       # old version syntax
    save_strategy="epoch",       # must match eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True if torch.cuda.is_available() else False,
    loss_type='focal'  # or 'cross_entropy'
)

# ------------------------------
# Trainer
# ------------------------------
trainer = AdaptiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ------------------------------
# Train
# ------------------------------
trainer.train()

# ------------------------------
# Save model
# ------------------------------
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")
print("✅ Training complete. Model saved to ./sentiment_model")


