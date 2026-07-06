"""
Metric-learning fine-tune for NAS C12 palm-vein student.

This script fine-tunes an existing EvalNetwork checkpoint with CE + ArcFace.
The ArcFace head is used only during training; saved checkpoints contain only
the original student state_dict, so existing ONNX/export/evaluation scripts
continue to work unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from palm_vein_dataset import create_retrain_dataloaders
from utils import get_device, setup_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class ArcMarginProduct(nn.Module):
    """ArcFace classification head for training-time metric supervision."""

    def __init__(self, in_features: int, out_features: int, s: float = 16.0, m: float = 0.2) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s


def freeze_batchnorm(model: nn.Module) -> int:
    n_bn = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            n_bn += 1
            for p in module.parameters():
                p.requires_grad = False
    return n_bn


def keep_batchnorm_frozen(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def parse_reduction_indices(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return None


def load_student(args, num_classes: int, device: torch.device, logger) -> EvalNetwork:
    with open(args.student_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    genotype = dict_to_genotype(cfg["genotype"])
    c_init = int(cfg.get("C_init", 8))
    num_cells = int(cfg.get("num_cells", 8))
    stem_downsample = int(cfg.get("stem_downsample", 2))
    reduction_indices = parse_reduction_indices(cfg.get("reduction_indices"))
    dropout = float(cfg.get("dropout", 0.3))

    model = EvalNetwork(
        genotype=genotype,
        C_init=c_init,
        num_cells=num_cells,
        num_classes=num_classes,
        auxiliary=False,
        dropout=dropout,
        stem_downsample=stem_downsample,
        reduction_indices=reduction_indices,
    )

    logger.info(
        f"Student arch: C_init={c_init}, num_cells={num_cells}, "
        f"stem_downsample={stem_downsample}, reduction_indices={reduction_indices}, dropout={dropout}"
    )
    logger.info(f"Loading student weights: {args.student_weights}")
    state = torch.load(args.student_weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    aux_keys = [k for k in unexpected if "_auxiliary_head" in k]
    other_unexpected = [k for k in unexpected if "_auxiliary_head" not in k]
    if aux_keys:
        logger.info(f"Auxiliary head keys skipped: {len(aux_keys)}")
    if other_unexpected:
        logger.warning(f"Unexpected keys: {other_unexpected[:5]}{'...' if len(other_unexpected) > 5 else ''}")

    return model.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, compute_auc: bool = False) -> dict:
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses = AverageMeter()
    correct = 0
    total = 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = ce(logits, labels)
        preds = logits.argmax(dim=1)

        losses.update(loss.item(), images.size(0))
        correct += preds.eq(labels).sum().item()
        total += labels.numel()

        if compute_auc:
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    result = {"loss": losses.avg, "acc": correct / max(total, 1), "correct": correct, "total": total}
    if compute_auc and all_probs:
        probs = np.vstack(all_probs)
        labels_np = np.concatenate(all_labels)
        try:
            present = np.unique(labels_np)
            result["auc"] = roc_auc_score(
                labels_np, probs, multi_class="ovr", average="macro", labels=present
            )
        except Exception:
            result["auc"] = None
    return result


@torch.no_grad()
def compute_eer(model: nn.Module, loader, device: torch.device) -> float | None:
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

    probs = np.vstack(all_probs)
    labels_np = np.concatenate(all_labels)
    eers = []
    for cls in np.unique(labels_np):
        y_bin = (labels_np == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, cls])
        fnr = 1 - tpr
        if len(fpr) > 1:
            try:
                eers.append(brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0))
            except Exception:
                pass
    return float(np.mean(eers)) if eers else None


def build_scheduler(optimizer, args):
    warmup_epochs = min(args.warmup_epochs, max(args.epochs, 0))
    if args.epochs <= 0:
        return None
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - warmup_epochs),
            eta_min=args.lr_min,
        )
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min)


def train_one_epoch(model, metric_head, loader, optimizer, scaler, device, args):
    model.train()
    if args.freeze_bn:
        keep_batchnorm_frozen(model)
    metric_head.train()

    ce_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    metric_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=args.amp and device.type == "cuda"):
            logits, embeddings = model.forward_with_embeddings(images)
            if args.method != "arcface":
                raise ValueError(f"Unsupported method for v1: {args.method}")
            metric_logits = metric_head(embeddings, labels)
            ce_loss = ce_fn(logits, labels)
            metric_loss = ce_fn(metric_logits, labels)
            loss = args.ce_weight * ce_loss + args.metric_weight * metric_loss

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(metric_head.parameters()), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        acc = preds.eq(labels).float().mean().item()
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        ce_meter.update(ce_loss.item(), n)
        metric_meter.update(metric_loss.item(), n)
        acc_meter.update(acc, n)

    return loss_meter.avg, ce_meter.avg, metric_meter.avg, acc_meter.avg


def save_checkpoint(model: nn.Module, output_dir: Path, epoch: int, val_acc: float, is_best: bool) -> None:
    torch.save(model.state_dict(), output_dir / "last_model.pth")
    metadata = {"epoch": epoch, "val_acc": val_acc, "timestamp": datetime.now().isoformat()}
    with open(output_dir / "last_checkpoint_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if is_best:
        torch.save(model.state_dict(), output_dir / "best_model.pth")
        with open(output_dir / "best_checkpoint_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def get_args():
    p = argparse.ArgumentParser(description="Metric-learning fine-tune for NAS C12")
    p.add_argument("--student_config", default="nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json")
    p.add_argument("--student_weights", default="nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth")
    p.add_argument("--data_dir", default="preprocessed_results")
    p.add_argument("--split_path", default="split_info.json")
    p.add_argument("--output_dir", default="metric_results/arcface_C12_L005_s16_m02_ce05_w05_lr5e6_freezebn")

    p.add_argument("--method", choices=["arcface"], default="arcface")
    p.add_argument("--ce_weight", type=float, default=0.5)
    p.add_argument("--metric_weight", type=float, default=0.5)
    p.add_argument("--arc_s", type=float, default=16.0)
    p.add_argument("--arc_m", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--lr_min", type=float, default=5e-7)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--drop_path", type=float, default=0.0)
    p.add_argument("--freeze_bn", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--cutout_length", type=int, default=16)
    p.add_argument("--augmentation_policy", default="v1_legacy")
    return p.parse_args()


def main() -> None:
    args = get_args()
    args.amp = not args.no_amp
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("metric_finetune", output_dir / "metric_finetune.log")
    device = get_device()

    logger.info("=" * 70)
    logger.info("Metric Fine-Tune Config")
    logger.info("=" * 70)
    logger.info(f"Student config : {args.student_config}")
    logger.info(f"Student weights: {args.student_weights}")
    logger.info(f"Method         : {args.method}")
    logger.info(f"CE/Metric      : {args.ce_weight} / {args.metric_weight}")
    logger.info(f"ArcFace        : s={args.arc_s} m={args.arc_m}")
    logger.info(f"Epochs         : {args.epochs}")
    logger.info(f"LR             : {args.lr} -> {args.lr_min}")
    logger.info(f"Freeze BN      : {args.freeze_bn}")
    logger.info(f"AMP            : {args.amp}")
    logger.info(f"Output dir     : {output_dir}")
    logger.info(f"Device         : {device}")
    if device.type == "cuda":
        logger.info(f"GPU            : {torch.cuda.get_device_name(0)}")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    train_loader, val_loader, test_loader, data_info = create_retrain_dataloaders(
        data_dir=args.data_dir,
        split_path=args.split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=True,
        cutout_length=args.cutout_length,
        augmentation_policy=args.augmentation_policy,
    )
    num_classes = int(data_info["num_classes"])

    model = load_student(args, num_classes, device, logger)
    model.set_drop_path_prob(args.drop_path)
    if args.freeze_bn:
        n_bn = freeze_batchnorm(model)
        logger.info(f"Freeze BN enabled: {n_bn} BatchNorm layers fixed")

    embedding_dim = int(model.classifier.in_features)
    metric_head = ArcMarginProduct(embedding_dim, num_classes, s=args.arc_s, m=args.arc_m).to(device)
    logger.info(f"Embedding dim: {embedding_dim} | classes: {num_classes}")

    logger.info("Evaluasi initial student sebelum metric fine-tune...")
    initial_val = evaluate(model, val_loader, device)
    initial_test = evaluate(model, test_loader, device)
    logger.info(f"Initial VAL  : acc={initial_val['acc']*100:.2f}% loss={initial_val['loss']:.4f}")
    logger.info(f"Initial TEST : acc={initial_test['acc']*100:.2f}% loss={initial_test['loss']:.4f}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(metric_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    csv_path = output_dir / "training_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_ce", "train_metric", "train_acc",
            "val_loss", "val_acc", "lr", "time_s",
        ])

    best_val_acc = 0.0
    best_epoch = 0
    stale_low_val = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.set_drop_path_prob(args.drop_path * epoch / max(args.epochs, 1))
        train_loss, train_ce, train_metric, train_acc = train_one_epoch(
            model, metric_head, train_loader, optimizer, scaler, device, args
        )
        if scheduler is not None:
            scheduler.step()

        val = evaluate(model, val_loader, device)
        val_acc = val["acc"]
        val_loss = val["loss"]
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
        save_checkpoint(model, output_dir, epoch, val_acc, is_best)

        if val_acc <= 0.9820:
            stale_low_val += 1
        else:
            stale_low_val = 0

        logger.info(
            f"E {epoch:3d}/{args.epochs} | loss={train_loss:.4f} "
            f"ce={train_ce:.4f} metric={train_metric:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"{'** BEST' if is_best else ''} | lr={lr:.2e} {elapsed:.1f}s"
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, round(train_loss, 6), round(train_ce, 6),
                round(train_metric, 6), round(train_acc, 4),
                round(val_loss, 6), round(val_acc, 4), lr, round(elapsed, 1),
            ])

        if epoch >= 15 and stale_low_val >= 5:
            logger.info("Early stop: val_acc <= 98.20% for 5 consecutive epochs after epoch 15.")
            break

    logger.info("=" * 70)
    logger.info(f"Training selesai. Best epoch={best_epoch} best_val_acc={best_val_acc:.4f}")
    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        logger.info("Memuat best_model.pth untuk evaluasi test...")
        model.load_state_dict(torch.load(best_path, map_location=device))

    logger.info("Evaluasi TEST set...")
    test = evaluate(model, test_loader, device, compute_auc=True)
    eer = compute_eer(model, test_loader, device)
    test["eer_pct"] = round(eer * 100, 4) if eer is not None else None

    logger.info("=" * 70)
    logger.info(f"TEST ACCURACY : {test['acc']*100:.2f}% ({test['correct']}/{test['total']})")
    logger.info(f"TEST LOSS     : {test['loss']:.4f}")
    logger.info(f"TEST AUC      : {test.get('auc')}")
    logger.info(f"TEST EER      : {test.get('eer_pct')}%")
    logger.info("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": args.method,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "initial_val_acc": round(initial_val["acc"], 4),
        "initial_val_loss": round(initial_val["loss"], 4),
        "initial_test_acc": round(initial_test["acc"], 4),
        "initial_test_loss": round(initial_test["loss"], 4),
        "test_acc": round(test["acc"], 4),
        "test_correct": test["correct"],
        "test_total": test["total"],
        "test_loss": round(test["loss"], 6),
        "test_auc": test.get("auc"),
        "test_eer_pct": test.get("eer_pct"),
        "config": vars(args),
    }
    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Output disimpan di: {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
