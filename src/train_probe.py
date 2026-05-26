"""
Step 1: Fine-tune DeBERTa-v3 as the black-box safety probe.

The probe takes (prompt, response_prefix) and outputs a risk score s_t ∈ [0, 1].
Training uses random prefix sampling to simulate the streaming scenario.

Usage:
    python src/train_probe.py [--model deberta-v3-small] [--epochs 3] [--batch-size 32]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_fineharm, split_train_calibration, PrefixDataset, print_dataset_stats

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].float().to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].float().to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        "loss": -np.mean(all_labels * np.log(all_probs + 1e-8) +
                         (1 - all_labels) * np.log(1 - all_probs + 1e-8)),
    }
    return metrics, all_probs


def main():
    parser = argparse.ArgumentParser(description="Train safety probe")
    parser.add_argument("--model", default="microsoft/deberta-v3-small",
                        help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cal-ratio", type=float, default=0.2,
                        help="Ratio of training data to use as calibration set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # Load data
    print("Loading FineHarm dataset...")
    train_samples = load_fineharm("train")
    val_samples = load_fineharm("val")
    test_samples = load_fineharm("test")

    # Split train into probe training + calibration
    train_probe, cal_samples = split_train_calibration(
        train_samples, cal_ratio=args.cal_ratio, seed=args.seed
    )
    print_dataset_stats(train_probe, "Probe Training")
    print_dataset_stats(cal_samples, "Calibration")

    # Save calibration data indices for later use
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="single_label_classification"
    )
    model = model.to(device)

    # Create datasets and dataloaders
    train_dataset = PrefixDataset(
        train_probe, tokenizer, max_length=args.max_length,
        min_prefix_ratio=0.05, max_prefix_ratio=1.0,
    )
    val_dataset = PrefixDataset(
        val_samples, tokenizer, max_length=args.max_length,
        min_prefix_ratio=0.1, max_prefix_ratio=1.0,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    best_val_f1 = 0
    history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics, _ = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f} "
              f"Loss: {val_metrics['loss']:.4f}")

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model_path = OUTPUT_DIR / "best_probe"
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"  -> Best model saved (F1={best_val_f1:.4f})")

    # Save training history
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on test set (using full responses)
    print("\nFinal evaluation on test set...")
    test_dataset = PrefixDataset(
        test_samples, tokenizer, max_length=args.max_length,
        min_prefix_ratio=0.5, max_prefix_ratio=1.0,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_metrics, _ = evaluate(model, test_loader, device)
    print(f"Test - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, "
          f"AUC: {test_metrics['auc']:.4f}")

    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nDone! Model saved to outputs/best_probe/")


if __name__ == "__main__":
    main()
