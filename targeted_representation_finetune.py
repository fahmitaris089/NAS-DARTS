"""Targeted representation fine-tune for hard class-pair confusions.

This post-KD experiment uses train-split samples from selected confusing
classes, plus random background classes, to strengthen student embeddings with
CE + teacher KD + anchor KL + supervised contrastive loss. Test images are not
used for training or sample selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "knowledge_distilation"))

from knowledge_distilation.kd_config import KD_CFG
from knowledge_distilation.kd_loss import HintonKDLoss
from knowledge_distilation.kd_train import (
    build_scheduler,
    compute_eer,
    evaluate,
    freeze_batchnorm,
    load_student,
    load_teacher,
    set_seed,
)
from palm_vein_dataset import build_label_map, create_retrain_dataloaders, load_split


@dataclass
class TargetSamplerStats:
    total_train: int
    target_count: int
    normal_count: int
    target_subjects: list[str]
    target_labels: list[int]
    target_ratio: float


class TargetedBatchSampler(Sampler[list[int]]):
    """Mixes target-class train samples with normal train samples per batch."""

    def __init__(
        self,
        target_indices: list[int],
        normal_indices: list[int],
        batch_size: int,
        target_ratio: float,
        num_batches: int,
        seed: int = 42,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not target_indices:
            raise ValueError("target_indices is empty")
        if not normal_indices:
            raise ValueError("normal_indices is empty")
        self.target_indices = list(target_indices)
        self.normal_indices = list(normal_indices)
        self.batch_size = int(batch_size)
        self.target_ratio = float(min(max(target_ratio, 0.0), 1.0))
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        target_per_batch = max(2, min(self.batch_size - 1, round(self.batch_size * self.target_ratio)))
        normal_per_batch = self.batch_size - target_per_batch
        for _ in range(self.num_batches):
            batch = []
            batch.extend(rng.choices(self.target_indices, k=target_per_batch))
            batch.extend(rng.choices(self.normal_indices, k=normal_per_batch))
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("targeted_representation_finetune")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(output_dir / "targeted_representation_finetune.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted representation fine-tune for NAS KD checkpoints")
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--student_weights", required=True)
    parser.add_argument("--anchor_weights", default="")
    parser.add_argument("--teacher_arch", default="efficientnet_v2_m")
    parser.add_argument("--teacher_weights", default=str(PROJECT_ROOT / "Teacher/training_results/EfficientNetV2M/best_model.pth"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "preprocessed_results"))
    parser.add_argument("--split_path", default=str(PROJECT_ROOT / "split_info.json"))
    parser.add_argument("--target_subjects", default="277,43,330,504,483,469,485")
    parser.add_argument("--target_ratio", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--kd_weight", type=float, default=0.3)
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--anchor_temperature", type=float, default=2.0)
    parser.add_argument("--contrast_weight", type=float, default=0.05)
    parser.add_argument("--contrast_temperature", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--lr_min", type=float, default=5e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--scheduler", choices=["cosine", "sgdr"], default="cosine")
    parser.add_argument("--sgdr_T0", type=int, default=50)
    parser.add_argument("--sgdr_T_mult", type=int, default=2)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--freeze_bn", action="store_true")
    parser.add_argument("--cutout_length", type=int, default=16)
    parser.add_argument(
        "--augmentation_policy",
        choices=["v1_legacy", "v2_multi_distance", "v3_no_flip_light", "v4_robust_light"],
        default="v1_legacy",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--early_stop_val_acc", type=float, default=0.9856)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--min_initial_test_acc", type=float, default=0.9970)
    return parser.parse_args()


def make_kd_cfg(args: argparse.Namespace):
    cfg = deepcopy(KD_CFG)
    cfg.teacher_arch = args.teacher_arch
    cfg.teacher_weights = args.teacher_weights
    cfg.student_config_path = args.student_config
    cfg.student_weights = args.student_weights
    cfg.output_dir = args.output_dir
    cfg.data_dir = args.data_dir
    cfg.split_path = args.split_path
    cfg.temperature = args.temperature
    cfg.alpha = 1.0
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.lr_min = min(args.lr_min, args.lr)
    cfg.weight_decay = args.weight_decay
    cfg.warmup_epochs = args.warmup_epochs
    cfg.scheduler = args.scheduler
    cfg.sgdr_T0 = args.sgdr_T0
    cfg.sgdr_T_mult = args.sgdr_T_mult
    cfg.drop_path_prob = args.drop_path
    cfg.freeze_bn = args.freeze_bn
    cfg.cutout_length = args.cutout_length
    cfg.augmentation_policy = args.augmentation_policy
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.amp = not args.no_amp
    cfg.mixup_alpha = 0.0
    cfg.cutmix_alpha = 0.0
    cfg.label_smoothing = 0.0
    cfg.log_interval = args.log_interval
    return cfg


def parse_subjects(text: str) -> list[str]:
    subjects = [item.strip() for item in text.split(",") if item.strip()]
    if not subjects:
        raise ValueError("--target_subjects must contain at least one subject id")
    return subjects


def find_target_indices(args: argparse.Namespace) -> tuple[list[int], list[int], TargetSamplerStats]:
    split = load_split(args.split_path)
    label_map = build_label_map(split["subjects"])
    target_subjects = parse_subjects(args.target_subjects)
    missing = [s for s in target_subjects if s not in label_map]
    if missing:
        raise ValueError(f"Target subjects not found in split_info subjects: {missing}")

    target_labels = {label_map[s] for s in target_subjects}
    target_indices: list[int] = []
    normal_indices: list[int] = []
    for idx, (_subject, _filename) in enumerate(split["train"]):
        label = label_map[_subject]
        if label in target_labels:
            target_indices.append(idx)
        else:
            normal_indices.append(idx)

    stats = TargetSamplerStats(
        total_train=len(split["train"]),
        target_count=len(target_indices),
        normal_count=len(normal_indices),
        target_subjects=target_subjects,
        target_labels=sorted(target_labels),
        target_ratio=float(args.target_ratio),
    )
    if not target_indices:
        raise RuntimeError("No target samples found in train split.")
    if not normal_indices:
        raise RuntimeError("No normal samples found in train split.")
    return target_indices, normal_indices, stats


def create_targeted_train_loader(args: argparse.Namespace, target_indices: list[int], normal_indices: list[int]):
    train_loader, _, _, _ = create_retrain_dataloaders(
        data_dir=args.data_dir,
        split_path=args.split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=224,
        use_augmentation=True,
        cutout_length=args.cutout_length,
        augmentation_policy=args.augmentation_policy,
        sampler_type="random",
        seed=args.seed,
    )
    sampler = TargetedBatchSampler(
        target_indices=target_indices,
        normal_indices=normal_indices,
        batch_size=args.batch_size,
        target_ratio=args.target_ratio,
        num_batches=len(train_loader),
        seed=args.seed,
    )
    return DataLoader(
        train_loader.dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def anchor_kl_loss(logits_student: torch.Tensor, logits_anchor: torch.Tensor, temperature: float) -> torch.Tensor:
    log_p = F.log_softmax(logits_student / temperature, dim=1)
    q = F.softmax(logits_anchor / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.t()
    logits = logits / max(float(temperature), 1e-6)
    batch_size = targets.size(0)
    eye = torch.eye(batch_size, dtype=torch.bool, device=targets.device)
    positive_mask = targets.view(-1, 1).eq(targets.view(1, -1)) & ~eye
    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        return embeddings.new_tensor(0.0)

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~eye).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(1.0)
    return -mean_log_prob_pos[valid].mean()


def save_config(output_dir: Path, args: argparse.Namespace, cfg, sampler_stats: TargetSamplerStats) -> None:
    config = {
        "timestamp": datetime.now().isoformat(),
        "method": "targeted_representation_finetune",
        "student_config_path": args.student_config,
        "student_weights": args.student_weights,
        "anchor_weights": args.anchor_weights or args.student_weights,
        "teacher_arch": args.teacher_arch,
        "teacher_weights": args.teacher_weights,
        "target_subjects": sampler_stats.target_subjects,
        "target_labels": sampler_stats.target_labels,
        "target_ratio": args.target_ratio,
        "temperature": args.temperature,
        "ce_weight": args.ce_weight,
        "kd_weight": args.kd_weight,
        "anchor_weight": args.anchor_weight,
        "anchor_temperature": args.anchor_temperature,
        "contrast_weight": args.contrast_weight,
        "contrast_temperature": args.contrast_temperature,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_min": cfg.lr_min,
        "weight_decay": args.weight_decay,
        "drop_path_prob": args.drop_path,
        "freeze_bn": args.freeze_bn,
        "cutout_length": args.cutout_length,
        "augmentation_policy": args.augmentation_policy,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "amp": cfg.amp,
        "sampler_stats": asdict(sampler_stats),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    anchor: nn.Module,
    loader,
    optimizer,
    scheduler,
    scaler,
    hinton_loss: HintonKDLoss,
    args: argparse.Namespace,
    cfg,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    anchor.eval()
    if args.freeze_bn:
        freeze_batchnorm(student)

    total_loss = total_ce = total_kd = total_anchor = total_contrast = 0.0
    correct = n_samples = 0
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            logits_teacher = teacher(images)
            logits_anchor = anchor(images)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=cfg.amp):
            logits_student, embeddings = student.forward_with_embeddings(images)
            _kd_total, kd_breakdown = hinton_loss(logits_student, logits_teacher, targets)
            ce = F.cross_entropy(logits_student, targets)
            kd_only = torch.as_tensor(kd_breakdown["loss_kd"], device=device, dtype=logits_student.dtype)
            anchor_loss_value = anchor_kl_loss(logits_student, logits_anchor, args.anchor_temperature)
            contrast = supervised_contrastive_loss(embeddings, targets, args.contrast_temperature)
            loss = (
                args.ce_weight * ce
                + args.kd_weight * kd_only
                + args.anchor_weight * anchor_loss_value
                + args.contrast_weight * contrast
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            pred = logits_student.argmax(dim=1)
            correct += int(pred.eq(targets).sum().item())
            n_samples += targets.size(0)

        total_loss += float(loss.detach().item())
        total_ce += float(ce.detach().item())
        total_kd += float(kd_only.detach().item())
        total_anchor += float(anchor_loss_value.detach().item())
        total_contrast += float(contrast.detach().item())

        if (batch_idx + 1) % args.log_interval == 0:
            logger.debug(
                f"  E{epoch:3d} [{batch_idx+1:4d}/{len(loader)}] "
                f"loss={loss.detach().item():.4f} ce={ce.detach().item():.4f} "
                f"kd={kd_only.detach().item():.4f} anchor={anchor_loss_value.detach().item():.4f} "
                f"contrast={contrast.detach().item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    n_batches = len(loader)
    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
        "kd": total_kd / n_batches,
        "anchor": total_anchor / n_batches,
        "contrast": total_contrast / n_batches,
        "acc": correct / n_samples if n_samples else 0.0,
    }


def save_checkpoint(student: nn.Module, output_dir: Path, is_best: bool) -> None:
    torch.save(student.state_dict(), output_dir / "last_model.pth")
    if is_best:
        torch.save(student.state_dict(), output_dir / "best_model.pth")


def main() -> None:
    args = parse_args()
    cfg = make_kd_cfg(args)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("  Targeted Representation Fine-tune")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Student weights : {args.student_weights}")
    logger.info(f"Teacher         : {args.teacher_arch} | {args.teacher_weights}")
    logger.info(f"Target subjects : {args.target_subjects} | target_ratio={args.target_ratio:.2f}")
    logger.info(
        f"Loss weights    : CE={args.ce_weight} KD={args.kd_weight} "
        f"anchor={args.anchor_weight} contrast={args.contrast_weight}"
    )

    logger.info("Loading deterministic loaders for initial eval...")
    _, val_loader, test_loader, _ = create_retrain_dataloaders(
        data_dir=args.data_dir,
        split_path=args.split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=cfg.input_size,
        use_augmentation=False,
        sampler_type="random",
        seed=args.seed,
    )

    logger.info("Loading models...")
    teacher = load_teacher(cfg, device, logger)
    student = load_student(cfg, device, logger)
    anchor_cfg = deepcopy(cfg)
    anchor_cfg.student_weights = args.anchor_weights or args.student_weights
    anchor = load_student(anchor_cfg, device, logger)
    anchor.eval()
    for param in anchor.parameters():
        param.requires_grad_(False)

    if args.freeze_bn:
        n_bn = freeze_batchnorm(student)
        logger.info(f"Freeze BN enabled: {n_bn} BatchNorm layers fixed")

    logger.info("Initial evaluation...")
    initial_val = evaluate(student, val_loader, device)
    initial_test = evaluate(student, test_loader, device, compute_auc=True)
    logger.info(f"Initial VAL  : acc={initial_val['acc']*100:.2f}% loss={initial_val['loss']:.4f}")
    logger.info(f"Initial TEST : acc={initial_test['acc']*100:.2f}% loss={initial_test['loss']:.4f}")
    if initial_test["acc"] < args.min_initial_test_acc:
        raise RuntimeError(
            f"Initial test accuracy {initial_test['acc']*100:.2f}% is below "
            f"required {args.min_initial_test_acc*100:.2f}%. Wrong starting checkpoint?"
        )

    target_indices, normal_indices, sampler_stats = find_target_indices(args)
    logger.info(
        f"Target sampler: target_samples={sampler_stats.target_count} "
        f"normal_samples={sampler_stats.normal_count} total_train={sampler_stats.total_train}"
    )
    save_config(output_dir, args, cfg, sampler_stats)

    train_loader = create_targeted_train_loader(args, target_indices, normal_indices)
    logger.info(
        f"Targeted train loader: batches={len(train_loader)} "
        f"target_ratio={args.target_ratio:.2f} batch_size={args.batch_size}"
    )

    hinton_loss = HintonKDLoss(
        temperature=args.temperature,
        alpha=0.5,
        label_smoothing=0.0,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler("cuda", enabled=cfg.amp)

    csv_path = output_dir / "training_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_ce", "train_kd", "train_anchor",
            "train_contrast", "train_acc", "val_loss", "val_acc", "lr", "time_s",
        ])

    best_val_acc = 0.0
    best_epoch = 0
    low_val_streak = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        dp_prob = args.drop_path * epoch / max(args.epochs, 1)
        student.set_drop_path_prob(dp_prob)

        train_stats = train_one_epoch(
            student, teacher, anchor, train_loader, optimizer, scheduler, scaler,
            hinton_loss, args, cfg, device, epoch, logger,
        )
        if args.freeze_bn:
            freeze_batchnorm(student)
        val = evaluate(student, val_loader, device)
        is_best = val["acc"] > best_val_acc
        if is_best:
            best_val_acc = val["acc"]
            best_epoch = epoch
        save_checkpoint(student, output_dir, is_best)

        if epoch > 10 and val["acc"] < args.early_stop_val_acc:
            low_val_streak += 1
        else:
            low_val_streak = 0

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  E {epoch:3d}/{args.epochs} | "
            f"loss={train_stats['loss']:.4f} ce={train_stats['ce']:.4f} "
            f"kd={train_stats['kd']:.4f} anchor={train_stats['anchor']:.4f} "
            f"contrast={train_stats['contrast']:.4f} train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} "
            f"{'** BEST' if is_best else ''} | lr={lr_now:.2e} {elapsed:.1f}s"
        )

        row = [
            epoch,
            round(train_stats["loss"], 6),
            round(train_stats["ce"], 6),
            round(train_stats["kd"], 6),
            round(train_stats["anchor"], 6),
            round(train_stats["contrast"], 6),
            round(train_stats["acc"], 6),
            round(val["loss"], 6),
            round(val["acc"], 6),
            round(lr_now, 10),
            round(elapsed, 1),
        ]
        history.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        if low_val_streak >= args.early_stop_patience:
            logger.info(
                f"Early stop: val_acc < {args.early_stop_val_acc:.4f} "
                f"for {args.early_stop_patience} consecutive epochs after epoch 10."
            )
            break

    logger.info("=" * 70)
    logger.info(f"Training selesai. Best val epoch={best_epoch} acc={best_val_acc:.4f}")

    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        student.load_state_dict(torch.load(best_path, map_location=device))
    else:
        torch.save(student.state_dict(), best_path)

    test = evaluate(student, test_loader, device, compute_auc=True)
    try:
        eer = compute_eer(student, test_loader, device)
    except Exception:
        eer = None

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "targeted_representation_finetune",
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 6),
        "initial_val_acc": round(initial_val["acc"], 6),
        "initial_test_acc": round(initial_test["acc"], 6),
        "test_acc": round(test["acc"], 6),
        "test_loss": round(test["loss"], 6),
        "test_auc": test.get("auc"),
        "test_eer_pct": round(eer * 100, 4) if eer is not None else None,
        "sampler_stats": asdict(sampler_stats),
        "config": json.loads((output_dir / "config.json").read_text(encoding="utf-8")),
        "history": history,
    }
    (output_dir / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info(f"TEST ACC  : {test['acc']*100:.2f}% loss={test['loss']:.4f}")
    logger.info(f"TEST AUC  : {test.get('auc')}")
    logger.info(f"TEST EER  : {round(eer * 100, 4) if eer is not None else None}%")
    logger.info(f"Output    : {output_dir}")


if __name__ == "__main__":
    main()
