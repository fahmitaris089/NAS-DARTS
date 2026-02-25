"""
Retrain — Train Derived Architecture from Scratch
===================================================
After P-DARTS search discovers the optimal cell topology (Genotype),
this script trains the derived network from random initialisation
on the FULL training set.

Training follows teacher's pattern:
  - AdamW optimizer, CosineAnnealing + warmup
  - CrossEntropy with label smoothing
  - Same augmentation + CutOut + DropPath
  - Best model by val_loss, full test evaluation

Usage:
    python retrain.py --genotype nas_results/search/genotype_final.json
    python retrain.py --genotype nas_results/search/genotype_final.json --C_init 24 --epochs 600
    python retrain.py --genotype nas_results/search/genotype_final.json --epochs 5  # quick test
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from nas_config import RETRAIN_CFG, RETRAIN_DIR, NUM_CLASSES, SEED
from genotypes import dict_to_genotype, genotype_to_dict
from model_eval import EvalNetwork, count_parameters, find_optimal_C_init, param_breakdown
from palm_vein_dataset import create_retrain_dataloaders
from utils import set_seed, get_device, setup_logger, AverageMeter


# ─── Training One Epoch ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    auxiliary, aux_weight, grad_clip):
    """Train one epoch with optional auxiliary head loss."""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        output = model(images)

        if auxiliary and isinstance(output, tuple):
            logits, logits_aux = output
            loss = criterion(logits, labels) + aux_weight * criterion(logits_aux, labels)
        else:
            logits = output if not isinstance(output, tuple) else output[0]
            loss = criterion(logits, labels)

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        _, pred = logits.max(1)
        acc = pred.eq(labels).float().mean().item()
        losses.update(loss.item(), images.size(0))
        top1.update(acc, images.size(0))

    return losses.avg, top1.avg


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(images)
        logits = output if not isinstance(output, tuple) else output[0]
        loss = criterion(logits, labels)

        _, pred = logits.max(1)
        acc = pred.eq(labels).float().mean().item()
        losses.update(loss.item(), images.size(0))
        top1.update(acc, images.size(0))

    return losses.avg, top1.avg


# ─── Test Evaluation (Full Metrics) ─────────────────────────────────────────

@torch.no_grad()
def evaluate_test(model, loader, device, num_classes):
    """Full test evaluation — same metrics as Teacher."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score, roc_curve,
    )
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    t_start = time.time()
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        logits = output if not isinstance(output, tuple) else output[0]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.append(probs)
        n_batches += 1

    inference_time = (time.time() - t_start) / max(n_batches, 1)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        present = np.unique(all_labels)
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr",
                            average="macro", labels=present) if len(present) > 1 else float("nan")
    except Exception:
        auc = float("nan")

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

    cm = confusion_matrix(all_labels, all_preds)
    cls_report = classification_report(all_labels, all_preds, zero_division=0, output_dict=False)

    results = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc": float(auc) if not np.isnan(auc) else None,
        "eer": float(eer_avg) if not np.isnan(eer_avg) else None,
        "inference_time_per_batch_sec": float(inference_time),
        "num_test_samples": int(len(all_labels)),
    }

    return results, cm, cls_report, all_labels, all_preds, all_probs


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_training_curves(log_path, save_dir):
    """Plot training curves from CSV log."""
    from utils import plot_training_curves as _plot
    _plot(log_path, save_dir)


def plot_confusion_matrix(cm, save_dir, num_classes):
    """Plot confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_size = max(8, num_classes * 0.12)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    if num_classes > 50:
        sns.heatmap(cm, cmap="Blues", ax=ax, cbar=True,
                    xticklabels=False, yticklabels=False)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"NAS Model — Confusion Matrix ({num_classes} classes)")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(all_labels, all_probs, save_dir, num_classes):
    """Plot macro ROC curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize

    present = np.unique(all_labels)
    y_bin = label_binarize(all_labels, classes=present)
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)

    for i, cls in enumerate(present):
        if y_bin.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, cls])
        else:
            fpr, tpr, _ = roc_curve(y_bin.ravel(), all_probs[:, cls])
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= len(present)

    try:
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr",
                                  average="macro", labels=present)
        auc_str = f"{macro_auc:.4f}"
    except Exception:
        auc_str = "N/A"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(all_fpr, mean_tpr, linewidth=2, label=f"NAS Model ROC (AUC = {auc_str})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — NAS Model (Macro-Average)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrain NAS-derived architecture")
    parser.add_argument("--genotype", type=str, required=True,
                        help="Path to genotype JSON (from search)")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(RETRAIN_DIR))
    parser.add_argument("--C_init", type=int, default=None,
                        help=f"Initial channels (default: auto-select for "
                             f"{RETRAIN_CFG['target_params_min']//1000}k-"
                             f"{RETRAIN_CFG['target_params_max']//1000}k params)")
    parser.add_argument("--num_cells", type=int, default=RETRAIN_CFG["num_cells"])
    parser.add_argument("--epochs", type=int, default=RETRAIN_CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=RETRAIN_CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=RETRAIN_CFG["lr"])
    parser.add_argument("--weight_decay", type=float, default=RETRAIN_CFG["weight_decay"])
    parser.add_argument("--drop_path_prob", type=float, default=RETRAIN_CFG["drop_path_prob"])
    parser.add_argument("--cutout_length", type=int, default=RETRAIN_CFG["cutout_length"])
    parser.add_argument("--auxiliary", action="store_true", default=RETRAIN_CFG["auxiliary"])
    parser.add_argument("--no_auxiliary", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=RETRAIN_CFG["num_workers"])
    args = parser.parse_args()

    use_auxiliary = args.auxiliary and not args.no_auxiliary

    # Setup
    set_seed(args.seed)
    device = get_device()
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("retrain", save_dir / "retrain.log")
    logger.info(f"NAS Retrain Started at {datetime.now().isoformat()}")

    # Load genotype
    genotype_path = Path(args.genotype)
    with open(genotype_path, "r") as f:
        genotype_dict = json.load(f)
    genotype = dict_to_genotype(genotype_dict)
    logger.info(f"Genotype loaded from {genotype_path}")
    logger.info(f"  Normal: {genotype.normal}")
    logger.info(f"  Reduce: {genotype.reduce}")

    # Data
    train_loader, val_loader, test_loader, data_info = create_retrain_dataloaders(
        data_dir=args.data_dir,
        split_path=args.split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=RETRAIN_CFG["use_augmentation"],
        cutout_length=args.cutout_length,
    )
    num_classes = data_info["num_classes"]

    # Determine C_init
    if args.C_init is None:
        C_init, est_params = find_optimal_C_init(
            genotype, args.num_cells, num_classes,
            target_min=RETRAIN_CFG["target_params_min"],
            target_max=RETRAIN_CFG["target_params_max"],
            auxiliary=use_auxiliary,
            dropout=RETRAIN_CFG["dropout"],
        )
        if C_init is None:
            C_init = 24  # fallback
            logger.warning(f"Could not find C_init in range, using default {C_init}")
        logger.info(f"Auto-selected C_init={C_init} (est. {est_params:,} params)")
    else:
        C_init = args.C_init

    # Build model
    model = EvalNetwork(
        genotype=genotype,
        C_init=C_init,
        num_cells=args.num_cells,
        num_classes=num_classes,
        auxiliary=use_auxiliary,
        dropout=RETRAIN_CFG["dropout"],
    ).to(device)

    total_params = count_parameters(model)
    logger.info(f"\nModel Architecture:")
    logger.info(f"  C_init     : {C_init}")
    logger.info(f"  Cells      : {args.num_cells}")
    logger.info(f"  Auxiliary   : {use_auxiliary}")
    logger.info(f"  Parameters : {total_params:,}")
    logger.info(param_breakdown(model))

    # Verify param budget
    if total_params < RETRAIN_CFG["target_params_min"]:
        logger.warning(f"  ⚠ Below target min ({RETRAIN_CFG['target_params_min']:,})")
    elif total_params > RETRAIN_CFG["target_params_max"]:
        logger.warning(f"  ⚠ Above target max ({RETRAIN_CFG['target_params_max']:,})")
    else:
        logger.info(f"  ✓ Within target range [{RETRAIN_CFG['target_params_min']:,}, "
                     f"{RETRAIN_CFG['target_params_max']:,}]")

    # Optimizer & scheduler
    criterion = nn.CrossEntropyLoss(
        label_smoothing=RETRAIN_CFG["label_smoothing"]).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    warmup_epochs = RETRAIN_CFG["warmup_epochs"]
    warmup_sched = LinearLR(optimizer,
                            start_factor=RETRAIN_CFG["warmup_factor"],
                            total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - warmup_epochs),
        eta_min=RETRAIN_CFG["lr_min"],
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    # Save config
    config = {
        **vars(args),
        "C_init": C_init,
        "total_params": total_params,
        "auxiliary": use_auxiliary,
        "genotype": genotype_to_dict(genotype),
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "retrain_cfg": {k: str(v) for k, v in RETRAIN_CFG.items()},
    }

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Training log
    log_path = save_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "lr", "drop_path", "epoch_time_sec",
    ])

    # ─── Training Loop ───────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    training_start = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {args.epochs} epochs")
    logger.info(f"  LR: {args.lr}, Weight Decay: {args.weight_decay}")
    logger.info(f"  Warmup: {warmup_epochs} epochs")
    logger.info(f"  DropPath: 0 → {args.drop_path_prob}")
    logger.info(f"  CutOut: {args.cutout_length}px")
    logger.info(f"  Batch: {args.batch_size}")
    logger.info(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        # Schedule drop path probability
        drop_path = args.drop_path_prob * epoch / args.epochs
        model.set_drop_path_prob(drop_path)

        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            auxiliary=use_auxiliary,
            aux_weight=RETRAIN_CFG["auxiliary_weight"],
            grad_clip=RETRAIN_CFG["grad_clip"],
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Log
        log_writer.writerow([
            epoch,
            f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_loss:.6f}", f"{val_acc:.6f}",
            f"{current_lr:.8f}", f"{drop_path:.4f}", f"{epoch_time:.2f}",
        ])
        log_file.flush()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_model.pth")

        # Print
        marker = " *** BEST" if epoch == best_epoch and val_loss <= best_val_loss else ""
        if epoch % 10 == 0 or epoch <= 5 or epoch == args.epochs:
            logger.info(
                f"  E{epoch:>4}/{args.epochs} │ "
                f"train_loss={train_loss:.4f}  acc={train_acc:.4f} │ "
                f"val_loss={val_loss:.4f}  acc={val_acc:.4f} │ "
                f"lr={current_lr:.6f}  dp={drop_path:.3f}  "
                f"{epoch_time:.1f}s{marker}"
            )

    log_file.close()
    torch.save(model.state_dict(), save_dir / "last_model.pth")

    total_time = time.time() - training_start
    logger.info(f"\nTraining completed in {total_time/60:.1f} min")
    logger.info(f"Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # ─── Test Evaluation ─────────────────────────────────────────────────
    logger.info(f"\n── Evaluating best model (epoch {best_epoch}) ──")
    model.load_state_dict(torch.load(save_dir / "best_model.pth", map_location=device))

    test_results, cm, cls_report, all_labels, all_preds, all_probs = \
        evaluate_test(model, test_loader, device, num_classes)

    test_results["best_epoch"] = best_epoch
    test_results["best_val_loss"] = float(best_val_loss)
    test_results["total_params"] = total_params
    test_results["training_time_min"] = float(total_time / 60)
    test_results["model_name"] = "NAS-PDARTS"
    test_results["C_init"] = C_init
    test_results["num_cells"] = args.num_cells

    # Model efficiency metrics
    from utils import model_size_mb, estimate_flops, measure_latency
    test_results["model_size_mb"] = model_size_mb(model)
    flops, _ = estimate_flops(model, device="cpu")
    if flops:
        test_results["flops"] = flops
        test_results["flops_M"] = flops / 1e6

    try:
        lat_gpu, lat_std = measure_latency(model, device=str(device))
        test_results["latency_gpu_ms"] = lat_gpu
        test_results["latency_gpu_std_ms"] = lat_std
    except Exception:
        pass

    try:
        model_cpu = model.cpu()
        lat_cpu, lat_cpu_std = measure_latency(model_cpu, device="cpu")
        test_results["latency_cpu_ms"] = lat_cpu
        test_results["latency_cpu_std_ms"] = lat_cpu_std
        model.to(device)
    except Exception:
        pass

    logger.info(f"\n  Test Results (NAS-PDARTS):")
    logger.info(f"    Accuracy  : {test_results['accuracy']*100:.2f}%")
    logger.info(f"    Precision : {test_results['precision']:.4f}")
    logger.info(f"    Recall    : {test_results['recall']:.4f}")
    logger.info(f"    F1 Score  : {test_results['f1_score']:.4f}")
    logger.info(f"    AUC       : {test_results.get('auc', 'N/A')}")
    logger.info(f"    EER       : {test_results.get('eer', 'N/A')}")
    logger.info(f"    Params    : {total_params:,}")
    logger.info(f"    Size      : {test_results['model_size_mb']:.2f} MB")
    if flops:
        logger.info(f"    FLOPs     : {flops/1e6:.1f} M")

    # Save
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(f"Model: NAS-PDARTS (C_init={C_init}, cells={args.num_cells})\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Test accuracy: {test_results['accuracy']*100:.2f}%\n\n")
        f.write(cls_report)

    # Plots
    logger.info("\nGenerating plots...")
    plot_training_curves(log_path, save_dir)
    plot_confusion_matrix(cm, save_dir, num_classes)

    try:
        plot_roc_curve(all_labels, all_probs, save_dir, num_classes)
    except Exception as e:
        logger.warning(f"ROC plot failed: {e}")

    # ─── Also evaluate last model ────────────────────────────────────────
    logger.info(f"\n── Evaluating last model (epoch {args.epochs}) ──")
    model.load_state_dict(torch.load(save_dir / "last_model.pth", map_location=device))
    last_results, _, _, _, _, _ = evaluate_test(model, test_loader, device, num_classes)
    last_results["model_name"] = "NAS-PDARTS"
    last_results["epoch"] = args.epochs
    with open(save_dir / "last_model_results.json", "w") as f:
        json.dump(last_results, f, indent=2, default=str)

    logger.info(f"    Last model accuracy: {last_results['accuracy']*100:.2f}%")
    logger.info(f"    Best model accuracy: {test_results['accuracy']*100:.2f}% (epoch {best_epoch})")

    # ─── Comparison with Teacher ─────────────────────────────────────────
    teacher_csv = Path(__file__).resolve().parent.parent / "Teacher" / "training_results" / "comparison_table.csv"
    if teacher_csv.exists():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Comparison with Teacher Models")
        logger.info(f"{'='*60}")
        import csv as csv_module
        with open(teacher_csv, "r") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                t_name = row.get("model", row.get("Model", "?"))
                t_acc = row.get("test_accuracy", row.get("Test Accuracy", "?"))
                t_params = row.get("total_params", row.get("Total Params", "?"))
                logger.info(f"  {t_name:<25} acc={t_acc}  params={t_params}")

        logger.info(f"  {'NAS-PDARTS':<25} acc={test_results['accuracy']*100:.2f}%  "
                     f"params={total_params:,}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  DONE: NAS-PDARTS Retrain")
    logger.info(f"  Params      : {total_params:,}")
    logger.info(f"  Best epoch  : {best_epoch}")
    logger.info(f"  Test acc    : {test_results['accuracy']*100:.2f}%")
    logger.info(f"  Output      : {save_dir}")
    logger.info(f"  Train time  : {total_time/60:.1f} min")
    logger.info(f"{'='*60}")
    logger.info(f"\nNext step: Knowledge Distillation")
    logger.info(f"  python kd_train.py --student_weights {save_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
