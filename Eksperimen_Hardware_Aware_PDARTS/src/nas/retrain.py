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
  - Best model by val_loss, best model by val_acc, and last model evaluation

Usage:
    python retrain.py --genotype nas_results/search/genotype_final.json
    python retrain.py --genotype nas_results/search/genotype_final.json --C_init 24 --epochs 600
    python retrain.py --genotype nas_results/search/genotype_final.json --epochs 5  # quick test
"""

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
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

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from nas_config import RETRAIN_CFG, RETRAIN_DIR, NUM_CLASSES, SEED
from genotypes import dict_to_genotype, genotype_to_dict
from model_eval import EvalNetwork, count_parameters, find_optimal_C_init, param_breakdown
from adaface import replace_linear_with_adaface, replace_linear_with_arcface
from palm_vein_dataset import create_retrain_dataloaders
from utils import set_seed, get_device, setup_logger, AverageMeter


# ─── Training One Epoch ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    auxiliary, aux_weight, grad_clip, loss_mode="ce"):
    """Train one epoch with optional auxiliary head loss."""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if loss_mode in {"adaface", "arcface", "subcenter_arcface"}:
            logits, margin_logits, _ = model.forward_adaface(images, labels)
            loss = criterion(margin_logits, labels)
            output = logits
        else:
            output = model(images)

        if loss_mode in {"adaface", "arcface", "subcenter_arcface"}:
            pass
        elif auxiliary and isinstance(output, tuple):
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
    # Checkpoint selection is always based on ordinary inference CE.  The
    # training criterion may contain label smoothing or a margin head.
    inference_criterion = nn.CrossEntropyLoss()
    losses = AverageMeter()
    top1 = AverageMeter()
    margins = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(images)
        logits = output if not isinstance(output, tuple) else output[0]
        loss = inference_criterion(logits, labels)

        _, pred = logits.max(1)
        acc = pred.eq(labels).float().mean().item()
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        competing_logits = logits.clone()
        competing_logits.scatter_(1, labels.unsqueeze(1), float("-inf"))
        true_class_margin = (true_logits - competing_logits.max(dim=1).values).mean().item()
        losses.update(loss.item(), images.size(0))
        top1.update(acc, images.size(0))
        margins.update(true_class_margin, images.size(0))

    return losses.avg, top1.avg, margins.avg


# ─── Test Evaluation (Full Metrics) ─────────────────────────────────────────

@torch.no_grad()
def evaluate_test(model, loader, device, num_classes):
    """Full test evaluation — same metrics as Teacher."""
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
    parser.add_argument("--stem_downsample", type=int, default=2,
                        help="Stem spatial downsample factor (power of 2). "
                             "2=224->112 (default), 4=224->56 (lower latency).")
    parser.add_argument("--reduction_indices", type=str, default=None,
                        help="Comma-separated cell indices used as reduction cells, "
                             "e.g. '2,5'. Default: [num_cells//3, 2*num_cells//3].")
    parser.add_argument("--epochs", type=int, default=RETRAIN_CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=RETRAIN_CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=RETRAIN_CFG["lr"])
    parser.add_argument("--lr_min", type=float, default=RETRAIN_CFG["lr_min"])
    parser.add_argument("--weight_decay", type=float, default=RETRAIN_CFG["weight_decay"])
    parser.add_argument("--warmup_epochs", type=int, default=RETRAIN_CFG["warmup_epochs"])
    parser.add_argument("--drop_path_prob", type=float, default=RETRAIN_CFG["drop_path_prob"])
    parser.add_argument("--cutout_length", type=int, default=RETRAIN_CFG["cutout_length"])
    parser.add_argument("--augmentation_policy", type=str, default="v1_legacy",
                        choices=["v1_legacy", "v2_multi_distance", "v3_no_flip_light", "v4_robust_light"],
                        help=(
                            "Augmentation policy: v1_legacy (with horizontal flip), "
                            "v2_multi_distance (aggressive no flip), v3_no_flip_light "
                            "(mild no flip), or v4_robust_light (robust brightness/crop no flip)"
                        ))
    parser.add_argument("--auxiliary", action="store_true", default=RETRAIN_CFG["auxiliary"])
    parser.add_argument("--no_auxiliary", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=RETRAIN_CFG["num_workers"])
    parser.add_argument("--loss-mode", choices=[
        "ce", "adaface", "arcface", "subcenter_arcface"
    ], default="ce")
    parser.add_argument("--label-smoothing", type=float,
                        default=float(RETRAIN_CFG["label_smoothing"]))
    parser.add_argument("--adaface-m", type=float, default=0.4)
    parser.add_argument("--adaface-h", type=float, default=0.333)
    parser.add_argument("--adaface-s", type=float, default=64.0)
    parser.add_argument("--adaface-t-alpha", type=float, default=0.01)
    parser.add_argument("--arcface-margin", type=float, default=0.5)
    parser.add_argument("--arcface-scale", type=float, default=64.0)
    parser.add_argument("--arcface-subcenters", type=int, default=None,
                        help="Defaults to 1 for arcface and 2 for subcenter_arcface")
    parser.add_argument("--skip-test-evaluation", action="store_true",
                        help="Stop after validation-based checkpointing during screening")
    parser.add_argument("--train_sampler", choices=["random", "pk"], default="random")
    parser.add_argument("--pk_p", type=int, default=16)
    parser.add_argument("--pk_k", type=int, default=4)
    parser.add_argument("--initial_weights", type=str, default="",
                        help="Common random state shared by controlled scratch ablations")
    parser.add_argument("--checkpoint_epochs", type=str, default="100")
    parser.add_argument("--resume_training_state", type=str, default="")
    args = parser.parse_args()

    margin_modes = {"adaface", "arcface", "subcenter_arcface"}
    use_auxiliary = args.auxiliary and not args.no_auxiliary and args.loss_mode not in margin_modes
    if args.loss_mode in margin_modes and args.label_smoothing != 0.0:
        parser.error("Margin-head runs require --label-smoothing 0.0")

    # Parse reduction indices ("2,5" -> [2, 5]); None keeps the default positions.
    reduction_indices = None
    if args.reduction_indices:
        reduction_indices = [int(x) for x in str(args.reduction_indices).split(",") if x.strip() != ""]

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
        augmentation_policy=args.augmentation_policy,
        sampler_type=args.train_sampler,
        pk_p=args.pk_p,
        pk_k=args.pk_k,
        seed=args.seed,
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
        stem_downsample=args.stem_downsample,
        reduction_indices=reduction_indices,
    ).to(device)
    if args.initial_weights:
        initial_payload = torch.load(args.initial_weights, map_location="cpu", weights_only=False)
        if isinstance(initial_payload, dict) and "student" in initial_payload:
            initial_payload = initial_payload["student"]
        model.load_state_dict(initial_payload, strict=True)
        logger.info(f"  Loaded shared initial state: {args.initial_weights}")
    if args.loss_mode == "adaface":
        replace_linear_with_adaface(
            model, num_classes=num_classes, m=args.adaface_m, h=args.adaface_h,
            s=args.adaface_s, t_alpha=args.adaface_t_alpha,
        )
        model.to(device)
        logger.info(
            f"  Classification head: AdaFace m={args.adaface_m} h={args.adaface_h} "
            f"s={args.adaface_s} t_alpha={args.adaface_t_alpha}"
        )
    elif args.loss_mode in {"arcface", "subcenter_arcface"}:
        default_k = 2 if args.loss_mode == "subcenter_arcface" else 1
        subcenters = args.arcface_subcenters or default_k
        if args.loss_mode == "arcface" and subcenters != 1:
            parser.error("arcface control requires exactly one center per class")
        if args.loss_mode == "subcenter_arcface" and subcenters != 2:
            parser.error("the bounded screening protocol fixes subcenter_arcface to K=2")
        replace_linear_with_arcface(
            model, num_classes=num_classes, m=args.arcface_margin,
            s=args.arcface_scale, num_subcenters=subcenters,
        )
        model.to(device)
        logger.info(
            f"  Classification head: ArcFace m={args.arcface_margin} "
            f"s={args.arcface_scale} K={subcenters}"
        )

    total_params = count_parameters(model)
    if not args.resume_training_state:
        torch.save(model.state_dict(), save_dir / "initial_student.pth")
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
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    warmup_epochs = args.warmup_epochs
    warmup_sched = LinearLR(optimizer,
                            start_factor=RETRAIN_CFG["warmup_factor"],
                            total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - warmup_epochs),
        eta_min=args.lr_min,
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
        "loss_mode": args.loss_mode,
        "adaface_m": args.adaface_m,
        "adaface_h": args.adaface_h,
        "adaface_s": args.adaface_s,
        "adaface_t_alpha": args.adaface_t_alpha,
        "effective_label_smoothing": args.label_smoothing,
        "arcface_margin": args.arcface_margin,
        "arcface_scale": args.arcface_scale,
        "arcface_subcenters": (
            args.arcface_subcenters or (2 if args.loss_mode == "subcenter_arcface" else 1)
        ),
    }

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Training log
    log_path = save_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "val_true_class_margin", "lr", "drop_path", "epoch_time_sec",
    ])

    # ─── Training Loop ───────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_acc = -float("inf")
    best_acc_epoch = 0
    best_checkpoint_margin = -float("inf")
    maximum_val_margin = -float("inf")
    training_start = time.time()
    checkpoint_epochs = {
        int(value) for value in args.checkpoint_epochs.split(",") if value.strip()
    }
    best_screening = None
    best_screening_epoch = 0
    start_epoch = 1

    def file_hash(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    provenance = {
        "genotype_sha256": file_hash(genotype_path),
        "split_sha256": file_hash(args.split_path) if args.split_path else None,
        "initial_weights_sha256": file_hash(args.initial_weights) if args.initial_weights else None,
        "seed": args.seed,
        "sampler": args.train_sampler,
        "pk_p": args.pk_p if args.train_sampler == "pk" else None,
        "pk_k": args.pk_k if args.train_sampler == "pk" else None,
    }

    def rng_state():
        return {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def save_full_state(path, epoch):
        sampler = getattr(train_loader, "batch_sampler", None)
        torch.save({
            "student": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "sampler": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
            "rng": rng_state(),
            "scaler": None,
            "provenance": provenance,
            "best_metrics": {
                "best_val_loss": best_val_loss, "best_epoch": best_epoch,
                "best_val_acc": best_val_acc, "best_acc_epoch": best_acc_epoch,
                "best_screening": best_screening,
                "best_screening_epoch": best_screening_epoch,
            },
        }, path)

    if args.resume_training_state:
        state = torch.load(args.resume_training_state, map_location=device, weights_only=False)
        if state.get("provenance") != provenance:
            raise ValueError("Resume provenance does not match current retraining run")
        model.load_state_dict(state["student"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler = getattr(train_loader, "batch_sampler", None)
        if state.get("sampler") is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(state["sampler"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        metrics = state["best_metrics"]
        best_val_loss, best_epoch = metrics["best_val_loss"], metrics["best_epoch"]
        best_val_acc, best_acc_epoch = metrics["best_val_acc"], metrics["best_acc_epoch"]
        best_screening = tuple(metrics["best_screening"]) if metrics["best_screening"] else None
        best_screening_epoch = metrics["best_screening_epoch"]
        start_epoch = int(state["epoch"]) + 1
    else:
        # Model/dataloader construction consumes RNG. Reset before epoch 1 so
        # controlled ablations with the same seed see the same stochastic stream.
        set_seed(args.seed)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {args.epochs} epochs")
    logger.info(f"  LR: {args.lr}, Weight Decay: {args.weight_decay}")
    logger.info(f"  Warmup: {warmup_epochs} epochs")
    logger.info(f"  DropPath: 0 → {args.drop_path_prob}")
    logger.info(f"  CutOut: {args.cutout_length}px")
    logger.info(f"  Batch: {args.batch_size}")
    logger.info(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        # Schedule drop path probability
        drop_path = args.drop_path_prob * epoch / args.epochs
        model.set_drop_path_prob(drop_path)
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch - 1)

        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            auxiliary=use_auxiliary,
            aux_weight=RETRAIN_CFG["auxiliary_weight"],
            grad_clip=RETRAIN_CFG["grad_clip"],
            loss_mode=args.loss_mode,
        )

        # Validate
        val_loss, val_acc, val_margin = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Log
        log_writer.writerow([
            epoch,
            f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_loss:.6f}", f"{val_acc:.6f}",
            f"{val_margin:.6f}",
            f"{current_lr:.8f}", f"{drop_path:.4f}", f"{epoch_time:.2f}",
        ])
        log_file.flush()

        # Save checkpoints by complementary validation criteria. Keep
        # best_model.pth as val_loss-based for backward compatibility.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_checkpoint_margin = val_margin
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            torch.save(model.state_dict(), save_dir / "best_by_val_loss.pth")

        maximum_val_margin = max(maximum_val_margin, val_margin)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_val_acc_model.pth")
            torch.save(model.state_dict(), save_dir / "best_by_val_acc.pth")

        screening = (round((1.0 - val_acc) * data_info["val_size"]), val_loss, -val_margin)
        if best_screening is None or screening < best_screening:
            best_screening = screening
            best_screening_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_screening.pth")
            save_full_state(save_dir / "training_state_best.pth", epoch)
        save_full_state(save_dir / "training_state_last.pth", epoch)
        if epoch in checkpoint_epochs:
            checkpoint_dir = save_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch:03d}.pth")
            save_full_state(checkpoint_dir / f"training_state_epoch_{epoch:03d}.pth", epoch)

        # Print
        markers = []
        if epoch == best_epoch and val_loss <= best_val_loss:
            markers.append("BEST_LOSS")
        if epoch == best_acc_epoch and val_acc >= best_val_acc:
            markers.append("BEST_ACC")
        marker = f" *** {'/'.join(markers)}" if markers else ""
        if epoch % 10 == 0 or epoch <= 5 or epoch == args.epochs:
            logger.info(
                f"  E{epoch:>4}/{args.epochs} │ "
                f"train_loss={train_loss:.4f}  acc={train_acc:.4f} │ "
                f"val_loss={val_loss:.4f}  acc={val_acc:.4f}  margin={val_margin:.4f} │ "
                f"lr={current_lr:.6f}  dp={drop_path:.3f}  "
                f"{epoch_time:.1f}s{marker}"
            )

    log_file.close()
    torch.save(model.state_dict(), save_dir / "last_model.pth")

    total_time = time.time() - training_start
    logger.info(f"\nTraining completed in {total_time/60:.1f} min")
    logger.info(f"Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")
    logger.info(f"Best val_acc : {best_val_acc:.6f} at epoch {best_acc_epoch}")

    def load_eval_model(weights_path):
        eval_model = EvalNetwork(
            genotype=genotype,
            C_init=C_init,
            num_cells=args.num_cells,
            num_classes=num_classes,
            auxiliary=use_auxiliary,
            dropout=RETRAIN_CFG["dropout"],
            stem_downsample=args.stem_downsample,
            reduction_indices=reduction_indices,
        ).to(device)
        if args.loss_mode == "adaface":
            replace_linear_with_adaface(
                eval_model, num_classes=num_classes, m=args.adaface_m, h=args.adaface_h,
                s=args.adaface_s, t_alpha=args.adaface_t_alpha,
            )
            eval_model.to(device)
        elif args.loss_mode in {"arcface", "subcenter_arcface"}:
            replace_linear_with_arcface(
                eval_model, num_classes=num_classes, m=args.arcface_margin,
                s=args.arcface_scale,
                num_subcenters=(
                    args.arcface_subcenters
                    or (2 if args.loss_mode == "subcenter_arcface" else 1)
                ),
            )
            eval_model.to(device)
        state_dict = torch.load(weights_path, map_location="cpu")
        eval_model.load_state_dict(state_dict)
        eval_model.to(device)
        eval_model.eval()
        return eval_model

    if args.skip_test_evaluation:
        screening = {
            "status": "screening_complete_test_not_evaluated",
            "best_epoch": best_epoch,
            "best_validation_loss": float(best_val_loss),
            "best_validation_accuracy": float(best_val_acc),
            "best_validation_accuracy_epoch": best_acc_epoch,
            "best_checkpoint_true_class_margin": float(best_checkpoint_margin),
            "maximum_validation_true_class_margin": float(maximum_val_margin),
            "checkpoint_selection": "lexicographic_validation_errors_loss_margin",
            "best_checkpoint": str(save_dir / "best_screening.pth"),
            "best_screening_epoch": best_screening_epoch,
            "best_screening_key": {
                "validation_errors": int(best_screening[0]),
                "validation_ce_loss": float(best_screening[1]),
                "negative_true_class_margin": float(best_screening[2]),
            },
            "loss_mode": args.loss_mode,
        }
        with open(save_dir / "screening_results.json", "w") as f:
            json.dump(screening, f, indent=2)
        plot_training_curves(log_path, save_dir)
        logger.info("Test evaluation skipped; screening stopped after validation checkpointing.")
        logger.info(json.dumps(screening, indent=2))
        return screening

    # ─── Test Evaluation ─────────────────────────────────────────────────
    logger.info(f"\n── Evaluating best model (epoch {best_epoch}) ──")
    eval_model = load_eval_model(save_dir / "best_model.pth")

    test_results, cm, cls_report, all_labels, all_preds, all_probs = \
        evaluate_test(eval_model, test_loader, device, num_classes)

    test_results["best_epoch"] = best_epoch
    test_results["best_val_loss"] = float(best_val_loss)
    test_results["best_val_acc_epoch"] = best_acc_epoch
    test_results["best_val_acc"] = float(best_val_acc)
    test_results["checkpoint_selection"] = "val_loss"
    test_results["total_params"] = total_params
    test_results["training_time_min"] = float(total_time / 60)
    test_results["model_name"] = "NAS-PDARTS"
    test_results["C_init"] = C_init
    test_results["num_cells"] = args.num_cells

    # Model efficiency metrics
    from utils import model_size_mb, estimate_flops, measure_latency
    test_results["model_size_mb"] = model_size_mb(eval_model)
    flops, _ = estimate_flops(eval_model, device="cpu")
    if flops:
        test_results["flops"] = flops
        test_results["flops_M"] = flops / 1e6

    try:
        lat_gpu, lat_std = measure_latency(eval_model, device=str(device))
        test_results["latency_gpu_ms"] = lat_gpu
        test_results["latency_gpu_std_ms"] = lat_std
    except Exception:
        pass

    try:
        model_cpu = copy.deepcopy(eval_model).cpu()
        lat_cpu, lat_cpu_std = measure_latency(model_cpu, device="cpu")
        test_results["latency_cpu_ms"] = lat_cpu
        test_results["latency_cpu_std_ms"] = lat_cpu_std
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
        f.write(f"Best val_loss epoch: {best_epoch}\n")
        f.write(f"Best val_acc epoch: {best_acc_epoch}\n")
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
    last_eval_model = load_eval_model(save_dir / "last_model.pth")
    last_results, _, _, _, _, _ = evaluate_test(last_eval_model, test_loader, device, num_classes)
    last_results["model_name"] = "NAS-PDARTS"
    last_results["epoch"] = args.epochs
    with open(save_dir / "last_model_results.json", "w") as f:
        json.dump(last_results, f, indent=2, default=str)

    logger.info(f"    Last model accuracy: {last_results['accuracy']*100:.2f}%")
    logger.info(f"    Best val_loss model accuracy: {test_results['accuracy']*100:.2f}% (epoch {best_epoch})")

    # ─── Also evaluate best-val-accuracy model ──────────────────────────
    logger.info(f"\n── Evaluating best val_acc model (epoch {best_acc_epoch}) ──")
    best_acc_eval_model = load_eval_model(save_dir / "best_val_acc_model.pth")
    best_acc_results, _, _, _, _, _ = evaluate_test(
        best_acc_eval_model, test_loader, device, num_classes
    )
    best_acc_results["model_name"] = "NAS-PDARTS"
    best_acc_results["epoch"] = best_acc_epoch
    best_acc_results["best_val_acc"] = float(best_val_acc)
    best_acc_results["checkpoint_selection"] = "val_acc"
    with open(save_dir / "best_val_acc_model_results.json", "w") as f:
        json.dump(best_acc_results, f, indent=2, default=str)

    logger.info(f"    Best val_acc model accuracy: {best_acc_results['accuracy']*100:.2f}%")
    logger.info(f"    Best val_acc checkpoint    : {save_dir / 'best_val_acc_model.pth'}")

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
    logger.info(f"  Best val_loss epoch : {best_epoch}")
    logger.info(f"  Best val_acc epoch  : {best_acc_epoch}")
    logger.info(f"  Best val_loss test  : {test_results['accuracy']*100:.2f}%")
    logger.info(f"  Best val_acc test   : {best_acc_results['accuracy']*100:.2f}%")
    logger.info(f"  Last model test     : {last_results['accuracy']*100:.2f}%")
    logger.info(f"  Output      : {save_dir}")
    logger.info(f"  Train time  : {total_time/60:.1f} min")
    logger.info(f"{'='*60}")
    logger.info(f"\nNext step: Knowledge Distillation")
    logger.info(f"  # val_loss checkpoint")
    logger.info(f"  python kd_train.py --student_weights {save_dir / 'best_model.pth'}")
    logger.info(f"  # val_acc checkpoint")
    logger.info(f"  python kd_train.py --student_weights {save_dir / 'best_val_acc_model.pth'}")
    logger.info(f"  # final epoch checkpoint")
    logger.info(f"  python kd_train.py --student_weights {save_dir / 'last_model.pth'}")


if __name__ == "__main__":
    main()
