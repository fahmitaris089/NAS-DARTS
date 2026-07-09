"""Teacher-guided hard-mining fine-tune for NAS KD checkpoints.

This script is a post-KD fine-tuning stage. It mines difficult samples from the
train split only, then fine-tunes the student with CE + teacher KD + anchor KL +
margin loss. Test errors are never used for mining or training.
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
from palm_vein_dataset import create_retrain_dataloaders


@dataclass
class HardMiningStats:
    total_train: int
    hard_count: int
    normal_count: int
    student_wrong_count: int
    low_margin_count: int
    teacher_disagree_count: int
    teacher_confident_count: int
    hard_margin_threshold: float
    teacher_conf_threshold: float
    fallback_lowest_margin_count: int
    student_acc_on_train_scan: float
    teacher_acc_on_train_scan: float


class HardMiningBatchSampler(Sampler[list[int]]):
    """Batch sampler that mixes hard and normal sample indices."""

    def __init__(
        self,
        hard_indices: list[int],
        normal_indices: list[int],
        batch_size: int,
        hard_ratio: float,
        num_batches: int,
        seed: int = 42,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not hard_indices:
            raise ValueError("hard_indices is empty; cannot hard-mine")
        if not normal_indices:
            raise ValueError("normal_indices is empty; cannot build mixed batches")
        self.hard_indices = list(hard_indices)
        self.normal_indices = list(normal_indices)
        self.batch_size = int(batch_size)
        self.hard_ratio = float(min(max(hard_ratio, 0.0), 1.0))
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        hard_per_batch = max(1, min(self.batch_size - 1, round(self.batch_size * self.hard_ratio)))
        normal_per_batch = self.batch_size - hard_per_batch

        for _ in range(self.num_batches):
            batch = []
            batch.extend(rng.choices(self.hard_indices, k=hard_per_batch))
            batch.extend(rng.choices(self.normal_indices, k=normal_per_batch))
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("hard_mining_finetune")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(output_dir / "hard_mining_finetune.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-guided hard-mining fine-tune")
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--student_weights", required=True)
    parser.add_argument("--anchor_weights", default="")
    parser.add_argument("--teacher_arch", default="efficientnet_v2_m")
    parser.add_argument("--teacher_weights", default=str(PROJECT_ROOT / "Teacher/training_results/EfficientNetV2M/best_model.pth"))
    parser.add_argument("--teacher2_arch", default="")
    parser.add_argument("--teacher2_weights", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "preprocessed_results"))
    parser.add_argument("--split_path", default=str(PROJECT_ROOT / "split_info.json"))
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--kd_weight", type=float, default=0.3)
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--anchor_temperature", type=float, default=2.0)
    parser.add_argument("--margin_weight", type=float, default=0.1)
    parser.add_argument("--margin_m", type=float, default=0.5)
    parser.add_argument("--hard_margin_threshold", type=float, default=0.20)
    parser.add_argument("--teacher_conf_threshold", type=float, default=0.30)
    parser.add_argument("--hard_ratio", type=float, default=0.50)
    parser.add_argument("--hard_top_fraction", type=float, default=0.10,
                        help="Fallback fraction of train samples with the lowest true-vs-best-wrong margins")
    parser.add_argument("--min_hard_samples", type=int, default=512,
                        help="Minimum hard samples; filled from lowest-margin samples if explicit hard mining is sparse")
    parser.add_argument("--epochs", type=int, default=30)
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


def save_config(output_dir: Path, args: argparse.Namespace, cfg, hard_stats: HardMiningStats | None = None) -> None:
    config = {
        "timestamp": datetime.now().isoformat(),
        "method": "teacher_guided_hard_mining_finetune",
        "student_config_path": args.student_config,
        "student_weights": args.student_weights,
        "anchor_weights": args.anchor_weights or args.student_weights,
        "teacher_arch": args.teacher_arch,
        "teacher_weights": args.teacher_weights,
        "teacher2_arch": args.teacher2_arch,
        "teacher2_weights": args.teacher2_weights,
        "temperature": args.temperature,
        "ce_weight": args.ce_weight,
        "kd_weight": args.kd_weight,
        "anchor_weight": args.anchor_weight,
        "anchor_temperature": args.anchor_temperature,
        "margin_weight": args.margin_weight,
        "margin_m": args.margin_m,
        "hard_margin_threshold": args.hard_margin_threshold,
        "teacher_conf_threshold": args.teacher_conf_threshold,
        "hard_ratio": args.hard_ratio,
        "hard_top_fraction": args.hard_top_fraction,
        "min_hard_samples": args.min_hard_samples,
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
        "hard_mining_stats": asdict(hard_stats) if hard_stats is not None else None,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def anchor_kl_loss(logits_student: torch.Tensor, logits_anchor: torch.Tensor, temperature: float) -> torch.Tensor:
    log_p = F.log_softmax(logits_student / temperature, dim=1)
    q = F.softmax(logits_anchor / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


def margin_loss(logits: torch.Tensor, targets: torch.Tensor, margin_m: float) -> torch.Tensor:
    true_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    wrong_logits = logits.masked_fill(
        F.one_hot(targets, num_classes=logits.size(1)).bool(),
        float("-inf"),
    )
    best_wrong = wrong_logits.max(dim=1).values
    return F.relu(margin_m - (true_logits - best_wrong)).mean()


@torch.no_grad()
def mine_hard_samples(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[list[int], list[int], HardMiningStats]:
    student.eval()
    teacher.eval()

    hard_indices: list[int] = []
    normal_indices: list[int] = []
    hard_index_set: set[int] = set()
    margin_records: list[tuple[float, int]] = []
    student_wrong = low_margin = teacher_disagree = teacher_confident = 0
    fallback_lowest_margin = 0
    student_correct = teacher_correct = total = 0
    offset = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        logits_student = student(images)
        logits_teacher = teacher(images)

        pred_student = logits_student.argmax(dim=1)
        pred_teacher = logits_teacher.argmax(dim=1)
        probs_teacher = F.softmax(logits_teacher, dim=1)
        teacher_true_conf = probs_teacher.gather(1, targets.view(-1, 1)).squeeze(1)

        true_logits = logits_student.gather(1, targets.view(-1, 1)).squeeze(1)
        wrong_logits = logits_student.masked_fill(
            F.one_hot(targets, num_classes=logits_student.size(1)).bool(),
            float("-inf"),
        )
        best_wrong = wrong_logits.max(dim=1).values
        margins = true_logits - best_wrong

        reason_student_wrong = pred_student.ne(targets)
        reason_low_margin = margins.lt(args.hard_margin_threshold)
        reason_teacher_disagree = pred_teacher.ne(pred_student)
        reason_teacher_confident = pred_teacher.eq(targets) & teacher_true_conf.ge(args.teacher_conf_threshold)
        hard_mask = (
            reason_student_wrong
            | reason_low_margin
            | reason_teacher_disagree
            | (reason_teacher_confident & margins.lt(args.margin_m))
        )

        student_wrong += int(reason_student_wrong.sum().item())
        low_margin += int(reason_low_margin.sum().item())
        teacher_disagree += int(reason_teacher_disagree.sum().item())
        teacher_confident += int(reason_teacher_confident.sum().item())
        student_correct += int(pred_student.eq(targets).sum().item())
        teacher_correct += int(pred_teacher.eq(targets).sum().item())
        total += batch_size

        hard_list = hard_mask.cpu().tolist()
        margin_list = margins.detach().cpu().tolist()
        for i, is_hard in enumerate(hard_list):
            idx = offset + i
            margin_records.append((float(margin_list[i]), idx))
            if is_hard:
                hard_indices.append(idx)
                hard_index_set.add(idx)
            else:
                normal_indices.append(idx)
        offset += batch_size

    min_from_fraction = int(round(total * max(args.hard_top_fraction, 0.0)))
    target_hard = max(int(args.min_hard_samples), min_from_fraction)
    target_hard = min(target_hard, total)
    if len(hard_indices) < target_hard:
        margin_records.sort(key=lambda item: item[0])
        normal_set = set(normal_indices)
        for _margin, idx in margin_records:
            if len(hard_indices) >= target_hard:
                break
            if idx in hard_index_set:
                continue
            hard_indices.append(idx)
            hard_index_set.add(idx)
            if idx in normal_set:
                normal_set.remove(idx)
            fallback_lowest_margin += 1
        normal_indices = sorted(normal_set)

    stats = HardMiningStats(
        total_train=total,
        hard_count=len(hard_indices),
        normal_count=len(normal_indices),
        student_wrong_count=student_wrong,
        low_margin_count=low_margin,
        teacher_disagree_count=teacher_disagree,
        teacher_confident_count=teacher_confident,
        hard_margin_threshold=args.hard_margin_threshold,
        teacher_conf_threshold=args.teacher_conf_threshold,
        fallback_lowest_margin_count=fallback_lowest_margin,
        student_acc_on_train_scan=student_correct / total if total else 0.0,
        teacher_acc_on_train_scan=teacher_correct / total if total else 0.0,
    )
    logger.info(
        "Hard mining scan: "
        f"hard={stats.hard_count}/{stats.total_train} normal={stats.normal_count} | "
        f"student_wrong={stats.student_wrong_count} low_margin={stats.low_margin_count} "
        f"teacher_disagree={stats.teacher_disagree_count} teacher_confident={stats.teacher_confident_count} | "
        f"fallback_lowest_margin={stats.fallback_lowest_margin_count} | "
        f"student_train_acc={stats.student_acc_on_train_scan:.4f} "
        f"teacher_train_acc={stats.teacher_acc_on_train_scan:.4f}"
    )
    if not hard_indices:
        raise RuntimeError("Hard mining found zero hard samples. Relax thresholds.")
    if not normal_indices:
        logger.warning("Hard mining marked all samples hard; using hard pool as normal pool fallback.")
        normal_indices = hard_indices[:]
    return hard_indices, normal_indices, stats


def create_hard_mining_train_loader(args: argparse.Namespace, hard_indices: list[int], normal_indices: list[int]):
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
    sampler = HardMiningBatchSampler(
        hard_indices=hard_indices,
        normal_indices=normal_indices,
        batch_size=args.batch_size,
        hard_ratio=args.hard_ratio,
        num_batches=len(train_loader),
        seed=args.seed,
    )
    return DataLoader(
        train_loader.dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )


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
) -> dict:
    student.train()
    teacher.eval()
    anchor.eval()
    if args.freeze_bn:
        freeze_batchnorm(student)

    total_loss = total_ce = total_kd = total_anchor = total_margin = 0.0
    correct = n_samples = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            logits_teacher = teacher(images)
            logits_anchor = anchor(images)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=cfg.amp):
            logits_student = student(images)
            kd_base, kd_breakdown = hinton_loss(logits_student, logits_teacher, targets)
            ce = F.cross_entropy(logits_student, targets)
            kd_only = kd_breakdown["loss_kd"]
            anchor_loss_value = anchor_kl_loss(logits_student, logits_anchor, args.anchor_temperature)
            margin = margin_loss(logits_student, targets, args.margin_m)
            loss = (
                args.ce_weight * ce
                + args.kd_weight * kd_only
                + args.anchor_weight * anchor_loss_value
                + args.margin_weight * margin
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
        total_kd += float(kd_only)
        total_anchor += float(anchor_loss_value.detach().item())
        total_margin += float(margin.detach().item())

        if (batch_idx + 1) % args.log_interval == 0:
            logger.debug(
                f"  E{epoch:3d} [{batch_idx+1:4d}/{len(loader)}] "
                f"loss={loss.detach().item():.4f} ce={ce.detach().item():.4f} "
                f"kd={kd_only:.4f} anchor={anchor_loss_value.detach().item():.4f} "
                f"margin={margin.detach().item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    n_batches = len(loader)
    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
        "kd": total_kd / n_batches,
        "anchor": total_anchor / n_batches,
        "margin": total_margin / n_batches,
        "acc": correct / n_samples,
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
    logger.info("  Teacher-Guided Hard-Mining Fine-tune")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Student weights : {args.student_weights}")
    logger.info(f"Teacher         : {args.teacher_arch} | {args.teacher_weights}")
    logger.info(
        f"Loss weights    : CE={args.ce_weight} KD={args.kd_weight} "
        f"anchor={args.anchor_weight} margin={args.margin_weight}"
    )

    logger.info("Loading deterministic loaders for initial eval and mining...")
    scan_loader, val_loader, test_loader, _ = create_retrain_dataloaders(
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

    hard_indices, normal_indices, hard_stats = mine_hard_samples(
        student, teacher, scan_loader, args, device, logger
    )
    save_config(output_dir, args, cfg, hard_stats)

    train_loader = create_hard_mining_train_loader(args, hard_indices, normal_indices)
    logger.info(
        f"Hard-mining train loader: batches={len(train_loader)} "
        f"hard_ratio={args.hard_ratio:.2f} batch_size={args.batch_size}"
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
            "train_margin", "train_acc", "val_loss", "val_acc", "lr", "time_s",
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

        if val["acc"] < args.early_stop_val_acc:
            low_val_streak += 1
        else:
            low_val_streak = 0

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  E {epoch:3d}/{args.epochs} | "
            f"loss={train_stats['loss']:.4f} ce={train_stats['ce']:.4f} "
            f"kd={train_stats['kd']:.4f} anchor={train_stats['anchor']:.4f} "
            f"margin={train_stats['margin']:.4f} train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} "
            f"{'** BEST' if is_best else ''} | lr={lr_now:.2e} {elapsed:.1f}s"
        )

        row = [
            epoch,
            round(train_stats["loss"], 6),
            round(train_stats["ce"], 6),
            round(train_stats["kd"], 6),
            round(train_stats["anchor"], 6),
            round(train_stats["margin"], 6),
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
                f"for {args.early_stop_patience} consecutive epochs."
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
        "method": "teacher_guided_hard_mining_finetune",
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 6),
        "initial_val_acc": round(initial_val["acc"], 6),
        "initial_test_acc": round(initial_test["acc"], 6),
        "test_acc": round(test["acc"], 6),
        "test_loss": round(test["loss"], 6),
        "test_auc": test.get("auc"),
        "test_eer_pct": round(eer * 100, 4) if eer is not None else None,
        "hard_mining_stats": asdict(hard_stats),
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
