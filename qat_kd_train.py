"""
Quantization-Aware KD fine-tuning for NAS students.

This is a lightweight QAT-style stage designed to improve robustness to the
existing ONNX Runtime PTQ/QDQ export path. It trains the FP32 student under
fake-quantized Conv/Linear weights and Conv activations, then saves the normal
FP32 state_dict. Export/INT8 quantization stays handled by export_kd_onnx_int8.py.
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
from datetime import datetime
from pathlib import Path
from types import MethodType

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

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


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("qat_kd_train")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(output_dir / "qat_kd_train.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def fake_quant_weight_per_channel_symmetric(weight: torch.Tensor, ch_axis: int = 0) -> torch.Tensor:
    """Per-output-channel symmetric int8 fake quant with STE."""
    if weight.numel() == 0:
        return weight

    reduce_dims = [i for i in range(weight.ndim) if i != ch_axis]
    max_abs = weight.detach().abs().amax(dim=reduce_dims)
    scale = (max_abs / 127.0).clamp(min=1e-8).to(weight.device)
    zero_point = torch.zeros_like(scale, dtype=torch.int32, device=weight.device)
    return torch.fake_quantize_per_channel_affine(
        weight, scale.float(), zero_point, ch_axis, -127, 127
    )


def fake_quant_activation_uint8(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor affine uint8 fake quant for activations with STE."""
    if x.numel() == 0:
        return x
    x_detached = x.detach()
    x_min = x_detached.amin()
    x_max = x_detached.amax()
    scale = ((x_max - x_min) / 255.0).clamp(min=1e-8)
    zero_point = torch.round(-x_min / scale).clamp(0, 255).to(torch.int32)
    return torch.fake_quantize_per_tensor_affine(x, scale.float(), zero_point, 0, 255)


class FakeQuantStudent(nn.Module):
    """
    Runtime-only fake-quant wrapper.

    The wrapped student keeps its original FP32 parameters. Conv2d/Linear forward
    calls use fake-quantized weights, and Conv2d outputs are fake-quantized as a
    proxy for activation quantization. Linear logits are not activation-quantized
    to avoid over-penalizing the final decision layer.
    """

    def __init__(self, student: nn.Module, quantize_activations: bool = True):
        super().__init__()
        self.student = student
        self.quantize_activations = quantize_activations
        self._patched: list[tuple[nn.Module, object]] = []
        self._patch_modules()

    def _patch_modules(self) -> None:
        for module in self.student.modules():
            if isinstance(module, nn.Conv2d):
                original_forward = module.forward

                def conv_forward(mod, input_tensor):
                    q_weight = fake_quant_weight_per_channel_symmetric(mod.weight, ch_axis=0)
                    out = F.conv2d(
                        input_tensor,
                        q_weight,
                        mod.bias,
                        mod.stride,
                        mod.padding,
                        mod.dilation,
                        mod.groups,
                    )
                    if self.quantize_activations:
                        out = fake_quant_activation_uint8(out)
                    return out

                module.forward = MethodType(conv_forward, module)
                self._patched.append((module, original_forward))

            elif isinstance(module, nn.Linear):
                original_forward = module.forward

                def linear_forward(mod, input_tensor):
                    q_weight = fake_quant_weight_per_channel_symmetric(mod.weight, ch_axis=0)
                    return F.linear(input_tensor, q_weight, mod.bias)

                module.forward = MethodType(linear_forward, module)
                self._patched.append((module, original_forward))

    def restore(self) -> None:
        for module, original_forward in self._patched:
            module.forward = original_forward
        self._patched.clear()

    def forward(self, x):
        return self.student(x)

    def set_drop_path_prob(self, prob: float) -> None:
        self.student.set_drop_path_prob(prob)


def anchor_kl_loss(
    logits_student: torch.Tensor,
    logits_anchor: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    log_p = F.log_softmax(logits_student / temperature, dim=1)
    q = F.softmax(logits_anchor / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAT-style quantization-aware KD fine-tuning")
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--student_weights", required=True)
    parser.add_argument("--teacher_arch", default="efficientnet_v2_m")
    parser.add_argument("--teacher_weights", default=str(PROJECT_ROOT / "Teacher/training_results/EfficientNetV2M/best_model.pth"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "preprocessed_results"))
    parser.add_argument("--split_path", default=str(PROJECT_ROOT / "split_info.json"))
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--anchor_temperature", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lr_min", type=float, default=1e-7)
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
    parser.add_argument("--no_activation_fake_quant", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
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
    cfg.alpha = args.alpha
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.lr_min = args.lr_min
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
    if cfg.lr_min > cfg.lr:
        cfg.lr_min = cfg.lr * 0.1
    return cfg


def save_config(output_dir: Path, args: argparse.Namespace, cfg) -> None:
    config = {
        "timestamp": datetime.now().isoformat(),
        "method": "qat_kd_fake_quant",
        "student_config_path": args.student_config,
        "student_weights": args.student_weights,
        "teacher_arch": args.teacher_arch,
        "teacher_weights": args.teacher_weights,
        "num_classes": cfg.num_classes,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "anchor_weight": args.anchor_weight,
        "anchor_temperature": args.anchor_temperature,
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
        "fake_quant": {
            "weight": "per_channel_symmetric_int8_ste_conv_linear",
            "activation": "per_tensor_uint8_ste_conv_outputs",
            "activation_enabled": not args.no_activation_fake_quant,
        },
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate_qat(fake_student: nn.Module, loader, device, compute_auc: bool = False):
    return evaluate(fake_student, loader, device, compute_auc=compute_auc)


def train_one_epoch_qat(
    student: nn.Module,
    fake_student: FakeQuantStudent,
    teacher: nn.Module,
    anchor: nn.Module,
    loader,
    optimizer,
    scheduler,
    scaler,
    hinton_loss: HintonKDLoss,
    args: argparse.Namespace,
    cfg,
    device,
    epoch: int,
    logger: logging.Logger,
):
    student.train()
    fake_student.train()
    teacher.eval()
    anchor.eval()
    if args.freeze_bn:
        freeze_batchnorm(student)

    total_loss = total_ce = total_kd = total_anchor = 0.0
    correct = n_samples = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            logits_teacher = teacher(images)
            logits_anchor = anchor(images)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=cfg.amp):
            logits_student_q = fake_student(images)
            kd_loss, breakdown = hinton_loss(logits_student_q, logits_teacher, targets)
            a_loss = anchor_kl_loss(
                logits_student_q,
                logits_anchor,
                temperature=args.anchor_temperature,
            )
            loss = kd_loss + args.anchor_weight * a_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            pred = logits_student_q.argmax(dim=1)
            correct += (pred == targets).sum().item()
            n_samples += targets.size(0)

        total_loss += float(loss.detach().item())
        total_ce += breakdown["loss_ce"]
        total_kd += breakdown["loss_kd"]
        total_anchor += float(a_loss.detach().item())

        if (batch_idx + 1) % args.log_interval == 0:
            logger.debug(
                f"  E{epoch:3d} [{batch_idx+1:4d}/{len(loader)}] "
                f"loss={loss.detach().item():.4f} ce={breakdown['loss_ce']:.4f} "
                f"kd={breakdown['loss_kd']:.4f} anchor={a_loss.detach().item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    n_batches = len(loader)
    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
        "kd": total_kd / n_batches,
        "anchor": total_anchor / n_batches,
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
    save_config(output_dir, args, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("  QAT / Quantization-Aware KD Fine-tune")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Student weights : {args.student_weights}")
    logger.info(f"Teacher         : {args.teacher_arch} | {args.teacher_weights}")
    logger.info(f"KD              : T={args.temperature} alpha={args.alpha}")
    logger.info(f"Anchor          : weight={args.anchor_weight} T={args.anchor_temperature}")
    logger.info(
        f"Fake quant      : weights=Conv/Linear per-channel int8, "
        f"activations={'off' if args.no_activation_fake_quant else 'Conv output uint8'}"
    )

    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader, _ = create_retrain_dataloaders(
        data_dir=args.data_dir,
        split_path=args.split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=cfg.input_size,
        use_augmentation=True,
        cutout_length=args.cutout_length,
        augmentation_policy=args.augmentation_policy,
        sampler_type="random",
        seed=args.seed,
    )
    logger.info(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | "
        f"Test batches: {len(test_loader)}"
    )

    logger.info("Loading models...")
    teacher = load_teacher(cfg, device, logger)
    student = load_student(cfg, device, logger)
    anchor = load_student(cfg, device, logger)
    anchor.eval()
    for param in anchor.parameters():
        param.requires_grad_(False)

    fake_student = FakeQuantStudent(
        student,
        quantize_activations=not args.no_activation_fake_quant,
    ).to(device)

    if args.freeze_bn:
        n_bn = freeze_batchnorm(student)
        logger.info(f"Freeze BN enabled: {n_bn} BatchNorm layers fixed")

    logger.info("Initial evaluation...")
    initial_fp32_val = evaluate(student, val_loader, device)
    initial_fp32_test = evaluate(student, test_loader, device)
    initial_q_val = evaluate_qat(fake_student, val_loader, device)
    initial_q_test = evaluate_qat(fake_student, test_loader, device)
    logger.info(
        f"Initial FP32 VAL  : acc={initial_fp32_val['acc']*100:.2f}% "
        f"loss={initial_fp32_val['loss']:.4f}"
    )
    logger.info(
        f"Initial FP32 TEST : acc={initial_fp32_test['acc']*100:.2f}% "
        f"loss={initial_fp32_test['loss']:.4f}"
    )
    logger.info(
        f"Initial fakeQ VAL : acc={initial_q_val['acc']*100:.2f}% "
        f"loss={initial_q_val['loss']:.4f}"
    )
    logger.info(
        f"Initial fakeQ TEST: acc={initial_q_test['acc']*100:.2f}% "
        f"loss={initial_q_test['loss']:.4f}"
    )

    hinton_loss = HintonKDLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        label_smoothing=0.0,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler("cuda", enabled=cfg.amp)

    csv_path = output_dir / "training_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_ce", "train_kd", "train_anchor",
            "train_acc", "fakeq_val_loss", "fakeq_val_acc",
            "fp32_val_loss", "fp32_val_acc", "lr", "time_s",
        ])

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        dp_prob = args.drop_path * epoch / max(args.epochs, 1)
        student.set_drop_path_prob(dp_prob)

        train_stats = train_one_epoch_qat(
            student, fake_student, teacher, anchor, train_loader,
            optimizer, scheduler, scaler, hinton_loss,
            args, cfg, device, epoch, logger,
        )

        if args.freeze_bn:
            freeze_batchnorm(student)
        fakeq_val = evaluate_qat(fake_student, val_loader, device)
        fp32_val = evaluate(student, val_loader, device)

        val_acc = fakeq_val["acc"]
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
        save_checkpoint(student, output_dir, is_best)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  E {epoch:3d}/{args.epochs} | "
            f"loss={train_stats['loss']:.4f} ce={train_stats['ce']:.4f} "
            f"kd={train_stats['kd']:.4f} anchor={train_stats['anchor']:.4f} "
            f"train_acc={train_stats['acc']:.4f} | "
            f"fakeQ_val_loss={fakeq_val['loss']:.4f} fakeQ_val_acc={fakeq_val['acc']:.4f} | "
            f"fp32_val_loss={fp32_val['loss']:.4f} fp32_val_acc={fp32_val['acc']:.4f} "
            f"{'** BEST' if is_best else ''} | lr={lr_now:.2e} {elapsed:.1f}s"
        )

        row = [
            epoch,
            round(train_stats["loss"], 6),
            round(train_stats["ce"], 6),
            round(train_stats["kd"], 6),
            round(train_stats["anchor"], 6),
            round(train_stats["acc"], 6),
            round(fakeq_val["loss"], 6),
            round(fakeq_val["acc"], 6),
            round(fp32_val["loss"], 6),
            round(fp32_val["acc"], 6),
            round(lr_now, 10),
            round(elapsed, 1),
        ]
        history.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    logger.info("=" * 70)
    logger.info(f"Training selesai. Best fakeQ val epoch={best_epoch} acc={best_val_acc:.4f}")

    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        student.load_state_dict(torch.load(best_path, map_location=device))
    else:
        torch.save(student.state_dict(), best_path)

    fp32_test = evaluate(student, test_loader, device, compute_auc=True)
    fakeq_test = evaluate_qat(fake_student, test_loader, device, compute_auc=True)
    try:
        fp32_eer = compute_eer(student, test_loader, device)
    except Exception:
        fp32_eer = None

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "qat_kd_fake_quant",
        "best_epoch": best_epoch,
        "best_fakeq_val_acc": round(best_val_acc, 6),
        "initial_fp32_val_acc": round(initial_fp32_val["acc"], 6),
        "initial_fp32_test_acc": round(initial_fp32_test["acc"], 6),
        "initial_fakeq_val_acc": round(initial_q_val["acc"], 6),
        "initial_fakeq_test_acc": round(initial_q_test["acc"], 6),
        "fp32_test_acc": round(fp32_test["acc"], 6),
        "fp32_test_loss": round(fp32_test["loss"], 6),
        "fp32_test_auc": fp32_test.get("auc"),
        "fp32_test_eer_pct": round(fp32_eer * 100, 4) if fp32_eer is not None else None,
        "fakeq_test_acc": round(fakeq_test["acc"], 6),
        "fakeq_test_loss": round(fakeq_test["loss"], 6),
        "fakeq_test_auc": fakeq_test.get("auc"),
        "config": json.loads((output_dir / "config.json").read_text(encoding="utf-8")),
        "history": history,
    }
    (output_dir / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info(f"FP32 TEST  : acc={fp32_test['acc']*100:.2f}% loss={fp32_test['loss']:.4f}")
    logger.info(f"fakeQ TEST : acc={fakeq_test['acc']*100:.2f}% loss={fakeq_test['loss']:.4f}")
    logger.info(f"Output     : {output_dir}")
    logger.info("Next: python export_kd_onnx_int8.py --model-dir \"{}\" --num-calib 834 --eval-accuracy".format(output_dir))

    fake_student.restore()


if __name__ == "__main__":
    main()
