"""
Palm Vein Recognition — Single Model Training Script
=====================================================
Usage:
    python3 train_model.py --model ResNet50
    python3 train_model.py --model InceptionV3 --epochs 300 --no_augmentation
    python3 train_model.py --model MobileNetV3Large --epochs 5 --batch_size 16
    python3 train_model.py --model GhostNet_050 --epochs 300 --batch_size 32

Models are trained independently. Run one at a time.
Results saved to training_results/{ModelName}/
"""

import argparse
import csv
import json
import time
import sys
import warnings

# Force UTF-8 output on Windows (avoids UnicodeEncodeError for box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from palm_vein_dataset import create_dataloaders
from model_factory import (
    create_model, get_input_size, get_available_models,
    get_backbone_and_head_params, freeze_backbone, unfreeze_backbone,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ─── Device ──────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)} (CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("Device: Apple MPS")
    else:
        dev = torch.device("cpu")
        print("Device: CPU")
    return dev


# ─── Training One Epoch ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, model_name, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # InceptionV3 returns (output, aux_output) during training
        if model_name == "InceptionV3" and model.training:
            outputs, aux_outputs = model(images)
            loss_main = criterion(outputs, labels)
            loss_aux  = criterion(aux_outputs, labels)
            loss = loss_main + 0.4 * loss_aux
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ─── Test Evaluation (Full Metrics) ─────────────────────────────────────────

@torch.no_grad()
def evaluate_test(model, loader, device, num_classes):
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    t_start = time.time()
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()
        preds   = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.append(probs)
        n_batches += 1

    inference_time = (time.time() - t_start) / n_batches  # per batch

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.vstack(all_probs)

    # Core metrics
    acc   = accuracy_score(all_labels, all_preds)
    prec  = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec   = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1    = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # AUC (one-vs-rest, macro)
    try:
        # Only compute for classes present in test set
        present_classes = np.unique(all_labels)
        if len(present_classes) > 1:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro",
                labels=present_classes,
            )
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    # EER (macro-averaged)
    try:
        eers = []
        for cls in np.unique(all_labels):
            y_bin = (all_labels == cls).astype(int)
            scores = all_probs[:, cls]
            fpr, tpr, _ = roc_curve(y_bin, scores)
            fnr = 1 - tpr
            if len(fpr) > 1:
                eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
                eers.append(eer)
        eer_avg = np.mean(eers) if eers else float("nan")
    except Exception:
        eer_avg = float("nan")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    cls_report = classification_report(
        all_labels, all_preds, zero_division=0, output_dict=False,
    )

    results = {
        "accuracy":       float(acc),
        "precision":      float(prec),
        "recall":         float(rec),
        "f1_score":       float(f1),
        "auc":            float(auc) if not np.isnan(auc) else None,
        "eer":            float(eer_avg) if not np.isnan(eer_avg) else None,
        "inference_time_per_batch_sec": float(inference_time),
        "num_test_samples": int(len(all_labels)),
    }

    return results, cm, cls_report, all_labels, all_preds, all_probs


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_training_curves(log_path, save_dir):
    """Plot training & validation loss/accuracy curves."""
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Skip baris corrupt (null bytes, kosong, atau bukan angka)
                epoch_val = row.get("epoch", "").replace("\x00", "").strip()
                if not epoch_val or not epoch_val.lstrip("-").isdigit():
                    continue
                epochs.append(int(epoch_val))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                train_acc.append(float(row["train_acc"]))
                val_acc.append(float(row["val_acc"]))
            except (ValueError, KeyError):
                continue  # skip baris yang tidak valid

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, val_loss,   label="Val Loss", linewidth=1.5)
    best_epoch = epochs[np.argmin(val_loss)]
    ax1.axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
                label=f"Best (epoch {best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_acc, label="Train Acc", linewidth=1.5)
    ax2.plot(epochs, val_acc,   label="Val Acc", linewidth=1.5)
    ax2.axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
                label=f"Best (epoch {best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, save_dir, num_classes):
    """Plot confusion matrix heatmap."""
    fig_size = max(8, num_classes * 0.12)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    if num_classes > 50:
        # For large number of classes, don't show annotations
        sns.heatmap(cm, cmap="Blues", ax=ax, cbar=True,
                    xticklabels=False, yticklabels=False)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({num_classes} classes)")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(all_labels, all_probs, save_dir, num_classes):
    """Plot macro ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Macro-average ROC
    from sklearn.preprocessing import label_binarize

    present_classes = np.unique(all_labels)
    y_bin = label_binarize(all_labels, classes=present_classes)

    # Compute macro ROC
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)

    for i, cls in enumerate(present_classes):
        if y_bin.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, cls])
        else:
            fpr, tpr, _ = roc_curve(y_bin.ravel(), all_probs[:, cls])
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= len(present_classes)

    try:
        macro_auc = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro",
            labels=present_classes,
        )
        auc_str = f"{macro_auc:.4f}"
    except Exception:
        auc_str = "N/A"

    ax.plot(all_fpr, mean_tpr, linewidth=2,
            label=f"Macro ROC (AUC = {auc_str})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Macro-Average)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─── Main Training Loop ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a CNN model for palm vein recognition")
    parser.add_argument("--model", type=str, required=True,
                        choices=get_available_models(),
                        help="Model name")
    parser.add_argument("--data_dir", type=str,
                        default=str(Path(__file__).resolve().parents[2] / "data" / "preprocessed"),
                        help="Path to preprocessed dataset")
    parser.add_argument("--output_dir", type=str, default="training_results",
                        help="Base output directory")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Epochs to freeze backbone (phase 1); keep short to avoid head overfitting")
    parser.add_argument("--lr_head", type=float, default=3e-4,
                        help="Learning rate for head")
    parser.add_argument("--lr_backbone", type=float, default=5e-5,
                        help="Learning rate for backbone (phase 2)")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="AdamW weight decay (higher = more regularization)")
    parser.add_argument("--label_smoothing", type=float, default=0.2,
                        help="Label smoothing factor")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="LR warmup epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable training augmentation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device    = get_device()
    save_dir  = Path(args.output_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model      : {args.model}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Freeze     : {args.freeze_epochs} epochs")
    print(f"  Augment    : {'OFF' if args.no_augmentation else 'ON (light)'}")
    print(f"  Output     : {save_dir}")
    print(f"{'='*60}\n")

    # Save config
    config = vars(args)
    config["device"] = str(device)
    config["timestamp"] = datetime.now().isoformat()
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Data ──────────────────────────────────────────────────────────────
    input_size = get_input_size(args.model)
    use_aug    = not args.no_augmentation

    train_loader, val_loader, test_loader, data_info = create_dataloaders(
        data_dir=args.data_dir,
        split_path=str(Path(args.output_dir) / "split_info.json"),
        batch_size=args.batch_size,
        input_size=input_size,
        num_workers=args.num_workers,
        use_augmentation=use_aug,
    )

    num_classes = data_info["num_classes"]

    # ── Model ─────────────────────────────────────────────────────────────
    model, _ = create_model(args.model, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Count total params
    total_params = sum(p.numel() for p in model.parameters())
    config["total_params"] = total_params
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Phase 1: Freeze backbone, train head only ─────────────────────────
    print(f"\n── Phase 1: Freeze backbone (epochs 1-{args.freeze_epochs}) ──")
    freeze_backbone(model, args.model)

    # Only head params for phase 1
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)

    # Warmup scheduler for phase 1
    warmup_iters = min(args.warmup_epochs, args.freeze_epochs)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, args.freeze_epochs - warmup_iters), eta_min=1e-5)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_iters])

    # ── Training log ──────────────────────────────────────────────────────
    log_path = save_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "phase", "train_loss", "train_acc",
        "val_loss", "val_acc", "lr", "epoch_time_sec",
    ])
    log_file.flush()

    best_val_loss   = float("inf")
    best_epoch      = 0
    training_start  = time.time()

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):

        # ── Phase transition: unfreeze at freeze_epochs+1 ──
        if epoch == args.freeze_epochs + 1:
            print(f"\n── Phase 2: Unfreeze all (epochs {epoch}-{args.epochs}) ──")
            unfreeze_backbone(model)

            # New optimizer with differential LR
            backbone_params, head_params_list = get_backbone_and_head_params(
                model, args.model)

            optimizer = AdamW([
                {"params": backbone_params, "lr": args.lr_backbone},
                {"params": head_params_list, "lr": args.lr_head},
            ], weight_decay=args.weight_decay)

            remaining = args.epochs - args.freeze_epochs
            warmup2   = min(args.warmup_epochs, remaining)
            warmup_sched2 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup2)
            cosine_sched2 = CosineAnnealingLR(
                optimizer, T_max=max(1, remaining - warmup2), eta_min=1e-6)
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup_sched2, cosine_sched2],
                milestones=[warmup2])

        epoch_start = time.time()
        phase = 1 if epoch <= args.freeze_epochs else 2

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            args.model, args.grad_clip,
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]  # head LR
        epoch_time = time.time() - epoch_start

        # Log
        log_writer.writerow([
            epoch, phase,
            f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_loss:.6f}", f"{val_acc:.6f}",
            f"{current_lr:.8f}", f"{epoch_time:.2f}",
        ])
        log_file.flush()

        # Save best model (by val_loss)
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            torch.save(model.state_dict(), save_dir / "best_model.pth")

        # Print progress
        marker = " *** BEST" if epoch == best_epoch and val_loss <= best_val_loss else ""
        if epoch % 10 == 0 or epoch <= 5 or epoch == args.epochs:
            print(
                f"  Epoch {epoch:>4}/{args.epochs}  P{phase} │ "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} │ "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} │ "
                f"lr={current_lr:.6f}  {epoch_time:.1f}s{marker}"
            )

    log_file.close()

    # Save last model
    torch.save(model.state_dict(), save_dir / "last_model.pth")

    total_time = time.time() - training_start
    print(f"\nTraining completed in {total_time/60:.1f} min")
    print(f"Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # ── Test Evaluation ───────────────────────────────────────────────────
    print(f"\n── Evaluating on test set (best model, epoch {best_epoch}) ──")

    # Load best model
    model.load_state_dict(torch.load(save_dir / "best_model.pth", map_location=device))

    test_results, cm, cls_report, all_labels, all_preds, all_probs = \
        evaluate_test(model, test_loader, device, num_classes)

    test_results["best_epoch"]       = best_epoch
    test_results["best_val_loss"]    = float(best_val_loss)
    test_results["total_params"]     = total_params
    test_results["training_time_min"] = float(total_time / 60)
    test_results["model_name"]       = args.model

    # Print results
    print(f"\n  Test Results ({args.model}):")
    print(f"    Accuracy  : {test_results['accuracy']*100:.2f}%")
    print(f"    Precision : {test_results['precision']:.4f}")
    print(f"    Recall    : {test_results['recall']:.4f}")
    print(f"    F1 Score  : {test_results['f1_score']:.4f}")
    print(f"    AUC       : {test_results['auc']}")
    print(f"    EER       : {test_results['eer']}")
    print(f"    Params    : {total_params:,}")

    # Save results
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Test accuracy: {test_results['accuracy']*100:.2f}%\n\n")
        f.write(cls_report)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_training_curves(log_path, save_dir)
    plot_confusion_matrix(cm, save_dir, num_classes)

    try:
        plot_roc_curve(all_labels, all_probs, save_dir, num_classes)
    except Exception as e:
        print(f"  Warning: ROC plot failed ({e})")

    # ── Also evaluate last model ──────────────────────────────────────────
    print(f"\n── Evaluating last model (epoch {args.epochs}) ──")
    model.load_state_dict(torch.load(save_dir / "last_model.pth", map_location=device))
    last_results, _, _, _, _, _ = evaluate_test(model, test_loader, device, num_classes)
    last_results["model_name"] = args.model
    last_results["epoch"] = args.epochs

    with open(save_dir / "last_model_results.json", "w") as f:
        json.dump(last_results, f, indent=2)

    print(f"    Last model accuracy: {last_results['accuracy']*100:.2f}%")
    print(f"    Best model accuracy: {test_results['accuracy']*100:.2f}%  (epoch {best_epoch})")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE: {args.model}")
    print(f"  Best model  : epoch {best_epoch}, val_loss={best_val_loss:.6f}")
    print(f"  Test acc     : {test_results['accuracy']*100:.2f}% (best) / {last_results['accuracy']*100:.2f}% (last)")
    print(f"  Output dir  : {save_dir}")
    print(f"  Training time: {total_time/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
