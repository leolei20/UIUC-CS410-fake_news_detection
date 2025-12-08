"""
DistilBERT Fake News Classifier
================================
Compares against baseline TF-IDF + Logistic Regression.

Assumptions:
- Data splits already exist: ../data/train.csv, dev.csv, test.csv
  with columns: ['clean_text', 'title', 'text', 'label']
- label is 0 = real, 1 = fake (as in train_baseline.py)

Outputs:
- Fine-tuned DistilBERT model + tokenizer in ../outputs/models/distilbert/
- JSON metrics in ../outputs/results/distilbert_results.json
- Test predictions in ../outputs/results/distilbert_test_predictions.csv
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# -----------------------------
# Config / Hyperparameters
# -----------------------------
RANDOM_SEED = 721
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 721):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------
class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


# -----------------------------
# Utilities
# -----------------------------
def build_text_column(df: pd.DataFrame) -> list:
    """
    Build input text from title + body.
    You can tweak this (e.g., add special separator) if you want.
    """
    titles = df["title"].fillna("").astype(str)
    bodies = df["text"].fillna("").astype(str)
    texts = (titles + " " + bodies).tolist()
    return texts


def tokenize_texts(tokenizer, texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True,
    )


def evaluate(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, 2]
            probs = torch.softmax(logits, dim=-1)[:, 1]  # prob of class 1 (fake)
            preds = torch.argmax(logits, dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    # Ranking metrics
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(all_labels, all_probs)
    except ValueError:
        pr_auc = float("nan")

    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(f1_macro),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "per_class": {
            "real_0": {
                "precision": float(precision_per_class[0]),
                "recall": float(recall_per_class[0]),
                "f1": float(f1_per_class[0]),
                "support": int(support[0]),
            },
            "fake_1": {
                "precision": float(precision_per_class[1]),
                "recall": float(recall_per_class[1]),
                "f1": float(f1_per_class[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": cm.tolist(),
    }

    return metrics, all_labels, all_preds, all_probs


# -----------------------------
# Main training routine
# -----------------------------
def main():
    set_seed(RANDOM_SEED)

    base_dir = Path(__file__).parent
    data_dir = base_dir / ".." / "data"
    outputs_dir = base_dir / ".." / "outputs"
    models_dir = outputs_dir / "models"
    results_dir = outputs_dir / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DISTILBERT FAKE NEWS CLASSIFIER")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print()

    # -------------------------
    # Load data
    # -------------------------
    train_df = pd.read_csv(data_dir / "train.csv")
    dev_df = pd.read_csv(data_dir / "dev.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # Ensure int labels (0 / 1)
    train_labels = train_df["label"].astype(int).values
    dev_labels = dev_df["label"].astype(int).values
    test_labels = test_df["label"].astype(int).values

    # Build text inputs
    train_texts = build_text_column(train_df)
    dev_texts = build_text_column(dev_df)
    test_texts = build_text_column(test_df)

    print(f"Train size: {len(train_texts)}")
    print(f"Dev size:   {len(dev_texts)}")
    print(f"Test size:  {len(test_texts)}")

    # -------------------------
    # Tokenizer & encodings
    # -------------------------
    print("\n[1] Loading tokenizer and encoding texts...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_encodings = tokenize_texts(tokenizer, train_texts)
    dev_encodings = tokenize_texts(tokenizer, dev_texts)
    test_encodings = tokenize_texts(tokenizer, test_texts)

    train_dataset = FakeNewsDataset(train_encodings, train_labels)
    dev_dataset = FakeNewsDataset(dev_encodings, dev_labels)
    test_dataset = FakeNewsDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # Model, optimizer, scheduler
    # -------------------------
    print("\n[2] Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # -------------------------
    # Training loop
    # -------------------------
    print("\n[3] Training...")
    best_dev_f1 = -1.0
    best_state_dict = None

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"  Step {step + 1}/{len(train_loader)} - loss: {avg_loss:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"\n  Average train loss: {avg_train_loss:.4f}")

        # Evaluate on dev set
        print("  Evaluating on dev set...")
        dev_metrics, _, _, _ = evaluate(model, dev_loader, device)
        print(
            f"  Dev - Acc: {dev_metrics['accuracy']:.4f}, "
            f"Macro-F1: {dev_metrics['macro_f1']:.4f}, "
            f"ROC-AUC: {dev_metrics['roc_auc']:.4f}, "
            f"PR-AUC: {dev_metrics['pr_auc']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            best_state_dict = model.state_dict()
            print("  -> New best model on dev set (saved in memory).")

    # Load best dev model before final test evaluation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # -------------------------
    # Final evaluation on test set
    # -------------------------
    print("\n[4] Final evaluation on test set...")
    test_metrics, test_y, test_preds, test_probs = evaluate(model, test_loader, device)

    print("\nTest metrics:")
    print(f"  Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:     {test_metrics['macro_f1']:.4f}")
    print(f"  ROC-AUC:      {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC (AP):  {test_metrics['pr_auc']:.4f}")
    print("  Confusion Matrix (rows = true, cols = pred):")
    print(np.array(test_metrics["confusion_matrix"]))

    # -------------------------
    # Save model & tokenizer
    # -------------------------
    print("\n[5] Saving model and results...")
    distilbert_dir = models_dir / "distilbert"
    distilbert_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(distilbert_dir)
    tokenizer.save_pretrained(distilbert_dir)
    print(f"  Saved model + tokenizer to: {distilbert_dir}")

    # Save metrics JSON (similar style to baseline_results.json)
    results = {
        "model": f"DistilBERT ({MODEL_NAME})",
        "hyperparameters": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "random_seed": RANDOM_SEED,
        },
        "test_metrics": test_metrics,
    }

    results_path = results_dir / "distilbert_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics to: {results_path}")

    # Save per-example test predictions for further analysis
    test_out_df = test_df.copy()
    test_out_df["predicted"] = test_preds
    test_out_df["probability_fake"] = test_probs
    test_pred_path = results_dir / "distilbert_test_predictions.csv"
    test_out_df.to_csv(test_pred_path, index=False)
    print(f"  Saved test predictions to: {test_pred_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
