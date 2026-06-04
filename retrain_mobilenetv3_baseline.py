#!/usr/bin/env python3
"""
MobileNetV3Large Baseline — 2-class palm vein fine-tune.

Trains MobileNetV3Large (pretrained ImageNet) on the same multi-distance
2-class dataset used by NAS retrain, with identical hyperparameters.

Purpose:
  Architecture efficiency comparison only (FLOPs, latency on Pi 5).
  Accuracy comparison is secondary — NAS and MobileNetV3 solve the
  same 2-class task but with very different model complexity.

Usage:
    python retrain_mobilenetv3_baseline.py
    python retrain_mobilenetv3_baseline.py --epochs 300 \\
        --output-dir nas_results/baseline_mobilenetv3
    python retrain_mobilenetv3_baseline.py \\
        --split_path dataset_multi_distance/split_info.json \\
        --output-dir nas_results/baseline_mobilenetv3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "dataset_multi_distance" / "split_info.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "nas_results" / "baseline_mobilenetv3"

# ─── Reuse helpers from existing scripts ─────────────────────────────────────
from retrain_run7_robust import (
    validate_split_file,
    convert_split_to_retrain_format,
    build_data_dir_structure,
)
from retrain import evaluate_test as evaluate_model
from palm_vein_dataset import create_retrain_dataloaders
from utils import set_seed, get_device, setup_logger, model_size_mb, estimate_flops, measure_latency


# ─── Model ───────────────────────────────────────────────────────────────────

def build_mobilenetv3(num_classes: int) -> nn.Module:
    """
    MobileNetV3Large with pretrained ImageNet weights.
    Replace final classifier head for num_classes.
    """
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    # classifier: Sequential(Linear(960,1280), Hardswish, Dropout(0.2), Linear(1280,1000))
    in_features = model.classifier[-1].in_features  # 1280
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ─── Training One Epoch ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MobileNetV3Large baseline retrain")
    p.add_argument("--split_path", type=Path, default=DEFAULT_SPLIT_PATH,
                   help="Path to multi-distance split_info.json")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Output directory for results")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("mobilenetv3_baseline",
                          output_dir / "training.log")
    logger.info(f"MobileNetV3Large Baseline — {datetime.now().isoformat()}")
    logger.info(f"Output: {output_dir}")

    # ── Dataset setup (reuse retrain_run7_robust helpers) ────────────────────
    split_info = validate_split_file(args.split_path)

    converted_split_path = output_dir / "split_info_converted.json"
    convert_split_to_retrain_format(split_info, converted_split_path)

    data_dir = build_data_dir_structure(split_info, output_dir)

    train_loader, val_loader, test_loader, ds_info = create_retrain_dataloaders(
        data_dir=data_dir,
        split_path=converted_split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=True,
        cutout_length=16,
        augmentation_policy="v2_multi_distance",
    )
    num_classes = ds_info["num_classes"]
    logger.info(f"Classes: {num_classes}  |  Train: {ds_info['train_size']}  |  "
                f"Val: {ds_info['val_size']}  |  Test: {ds_info['test_size']}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_mobilenetv3(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"MobileNetV3Large  params={total_params:,}  trainable={trainable_params:,}")

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer,
                                         T_max=args.epochs - args.warmup_epochs,
                                         eta_min=1e-6)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[args.warmup_epochs])

    # ── Training Loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    log_rows = []
    t_start = time.time()

    log_csv = output_dir / "training_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device, args.grad_clip)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.4f}",
                             f"{vl_loss:.6f}", f"{vl_acc:.4f}", f"{lr:.6f}"])

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d}/{args.epochs}  "
                        f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
                        f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}  "
                        f"lr={lr:.6f}")

    training_time_min = (time.time() - t_start) / 60
    logger.info(f"Training done in {training_time_min:.1f} min  |  best epoch={best_epoch}")

    # ── Evaluate on test set ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(output_dir / "best_model.pth",
                                     map_location=device))
    test_results, cm, cls_report, _, _, _ = evaluate_model(model, test_loader, device)
    test_results["best_epoch"] = best_epoch
    test_results["best_val_loss"] = float(best_val_loss)
    test_results["total_params"] = total_params
    test_results["training_time_min"] = float(training_time_min)
    test_results["model_name"] = "MobileNetV3Large"

    # ── FLOPs & Latency ──────────────────────────────────────────────────────
    eval_model = model.to("cpu").eval()
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

    lat_cpu, lat_cpu_std = measure_latency(eval_model, device="cpu", warmup=5, repeats=20)
    test_results["latency_cpu_ms"] = lat_cpu
    test_results["latency_cpu_std_ms"] = lat_cpu_std

    # ── Save results ─────────────────────────────────────────────────────────
    (output_dir / "test_results.json").write_text(
        json.dumps(test_results, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(cls_report, encoding="utf-8")

    # Save config
    config = vars(args)
    config["split_path"] = str(config["split_path"])
    config["output_dir"] = str(config["output_dir"])
    config["num_classes"] = num_classes
    config["timestamp"] = datetime.now().isoformat()
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info(f"  Accuracy  : {test_results['accuracy']:.4f}")
    logger.info(f"  EER       : {test_results.get('eer', 'N/A')}")
    logger.info(f"  Params    : {total_params:,}")
    logger.info(f"  FLOPs     : {test_results.get('flops_M', 'N/A'):.1f} M")
    logger.info(f"  CPU lat   : {test_results.get('latency_cpu_ms', 'N/A'):.1f} ms")
    logger.info(f"  GPU lat   : {test_results.get('latency_gpu_ms', 'N/A'):.1f} ms")
    logger.info(f"  Model size: {test_results['model_size_mb']:.2f} MB")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
