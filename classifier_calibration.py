"""Classifier calibration for a trained NAS EvalNetwork.

This script is intentionally separate from retrain/KD. It starts from a
trained checkpoint, keeps the NAS backbone fixed, and calibrates only the
final classifier by either:
  - classifier_only: train the existing linear classifier with CE
  - prototype: replace classifier weights with class embedding prototypes

The saved checkpoint remains a normal EvalNetwork state_dict and can be used
by analyze_prediction_overlap.py and the existing ONNX export scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from palm_vein_dataset import create_retrain_dataloaders
from utils import AverageMeter, get_device, set_seed, setup_logger


@dataclass
class CalibrationConfig:
    student_config: str
    student_weights: str
    method: str
    output_dir: str
    data_dir: str = "preprocessed_results"
    split_path: str = "split_info.json"
    epochs: int = 40
    batch_size: int = 64
    lr: float = 5e-5
    lr_min: float = 5e-6
    weight_decay: float = 0.0
    freeze_bn: bool = False
    num_workers: int = 0
    cutout_length: int = 0
    augmentation_policy: str = "v3_no_flip_light"
    seed: int = 42
    input_size: int = 224


def parse_reduction_indices(raw_value) -> list[int] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return [int(x) for x in raw_value]
    if isinstance(raw_value, str):
        return [int(x.strip()) for x in raw_value.split(",") if x.strip()]
    raise TypeError(f"Unsupported reduction_indices type: {type(raw_value)}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_model(config_path: Path, weights_path: Path, num_classes: int, device: torch.device) -> EvalNetwork:
    cfg = load_json(config_path)
    genotype = dict_to_genotype(cfg["genotype"])
    retrain_cfg = cfg.get("retrain_cfg", {})
    dropout = float(retrain_cfg.get("dropout", cfg.get("dropout", 0.3)))

    model = EvalNetwork(
        genotype=genotype,
        C_init=int(cfg["C_init"]),
        num_cells=int(cfg["num_cells"]),
        num_classes=num_classes,
        auxiliary=False,
        dropout=dropout,
        stem_downsample=int(cfg.get("stem_downsample", 2)),
        reduction_indices=parse_reduction_indices(cfg.get("reduction_indices")),
    )

    state_dict = torch.load(weights_path, map_location="cpu")
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_auxiliary_head")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    return model.to(device)


def set_backbone_frozen(model: EvalNetwork, freeze_bn: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    if freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False


def set_train_mode_classifier_only(model: EvalNetwork, freeze_bn: bool) -> None:
    model.eval()
    model.classifier.train()
    if not freeze_bn:
        # Keep non-BN backbone deterministic because backbone parameters are frozen.
        model.classifier.train()


@torch.no_grad()
def evaluate(model: EvalNetwork, loader, device: torch.device) -> dict:
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.numel()
        losses.update(loss.item(), labels.numel())

    return {
        "loss": float(losses.avg),
        "acc": correct / total if total else 0.0,
        "correct": int(correct),
        "total": int(total),
    }


def train_classifier_only(
    model: EvalNetwork,
    train_loader,
    val_loader,
    test_loader,
    cfg: CalibrationConfig,
    output_dir: Path,
    logger,
    device: torch.device,
) -> dict:
    set_backbone_frozen(model, cfg.freeze_bn)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.classifier.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr_min)

    best_val_acc = -math.inf
    best_epoch = 0
    best_state = None
    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_sec"])

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            set_train_mode_classifier_only(model, cfg.freeze_bn)
            losses = AverageMeter()
            top1 = AverageMeter()

            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    embeddings = model.forward_features(images)
                logits = model.classifier(embeddings)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                acc = preds.eq(labels).float().mean().item()
                losses.update(loss.item(), labels.size(0))
                top1.update(acc, labels.size(0))

            scheduler.step()
            val = evaluate(model, val_loader, device)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            writer.writerow([epoch, losses.avg, top1.avg, val["loss"], val["acc"], lr, elapsed])
            f.flush()

            marker = ""
            if val["acc"] > best_val_acc:
                best_val_acc = val["acc"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, output_dir / "best_model.pth")
                marker = " ** BEST"

            logger.info(
                f"E {epoch:3d}/{cfg.epochs} | loss={losses.avg:.4f} "
                f"train_acc={top1.avg:.4f} | val_loss={val['loss']:.4f} "
                f"val_acc={val['acc']:.4f} | lr={lr:.2e} {elapsed:.1f}s{marker}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "last_model.pth")
    test = evaluate(model, test_loader, device)
    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test": test,
    }


@torch.no_grad()
def apply_prototype_classifier(
    model: EvalNetwork,
    train_loader,
    val_loader,
    test_loader,
    cfg: CalibrationConfig,
    output_dir: Path,
    logger,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        embeddings = model.forward_features(images)
        embeddings = F.normalize(embeddings, dim=1)
        if sums is None:
            sums = torch.zeros(num_classes, embeddings.size(1), dtype=torch.float32, device=device)
        sums.index_add_(0, labels, embeddings)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

    if sums is None:
        raise RuntimeError("No training embeddings collected.")
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).view(-1).tolist()
        raise RuntimeError(f"Missing train samples for classes: {missing[:10]}")

    prototypes = sums / counts.unsqueeze(1).clamp_min(1.0)
    prototypes = F.normalize(prototypes, dim=1)
    old_weight_norm = model.classifier.weight.detach().norm(dim=1).mean().item()
    scale = old_weight_norm if old_weight_norm > 0 else 1.0

    model.classifier.weight.copy_(prototypes * scale)
    model.classifier.bias.zero_()

    val = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)
    torch.save(model.state_dict(), output_dir / "best_model.pth")
    torch.save(model.state_dict(), output_dir / "last_model.pth")
    logger.info(
        f"Prototype classifier | val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} "
        f"| test_acc={test['acc']:.4f}"
    )
    return {
        "best_epoch": 0,
        "best_val_acc": val["acc"],
        "test": test,
        "prototype_scale": scale,
    }


def parse_args() -> CalibrationConfig:
    parser = argparse.ArgumentParser(description="Calibrate final classifier of a trained NAS model")
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--student_weights", required=True)
    parser.add_argument("--method", choices=["classifier_only", "prototype"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="preprocessed_results")
    parser.add_argument("--split_path", default="split_info.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_min", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--freeze_bn", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cutout_length", type=int, default=0)
    parser.add_argument(
        "--augmentation_policy",
        choices=["v1_legacy", "v2_multi_distance", "v3_no_flip_light", "v4_robust_light"],
        default="v3_no_flip_light",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return CalibrationConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    output_dir = resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"classifier_calibration_{output_dir.name}", output_dir / "calibration.log")
    device = get_device()

    logger.info("=" * 70)
    logger.info("Classifier Calibration")
    logger.info("=" * 70)
    logger.info(f"Method      : {cfg.method}")
    logger.info(f"Weights     : {cfg.student_weights}")
    logger.info(f"Output      : {output_dir}")
    logger.info(f"Device      : {device}")

    train_loader, val_loader, test_loader, info = create_retrain_dataloaders(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        batch_size=cfg.batch_size,
        input_size=cfg.input_size,
        num_workers=cfg.num_workers,
        use_augmentation=(cfg.method == "classifier_only"),
        cutout_length=cfg.cutout_length if cfg.method == "classifier_only" else 0,
        augmentation_policy=cfg.augmentation_policy,
        sampler_type="random",
        seed=cfg.seed,
    )

    model = build_model(
        resolve_path(cfg.student_config),
        resolve_path(cfg.student_weights),
        num_classes=info["num_classes"],
        device=device,
    )

    initial_val = evaluate(model, val_loader, device)
    initial_test = evaluate(model, test_loader, device)
    logger.info(
        f"Initial VAL  : acc={initial_val['acc']*100:.2f}% "
        f"loss={initial_val['loss']:.4f}"
    )
    logger.info(
        f"Initial TEST : acc={initial_test['acc']*100:.2f}% "
        f"({initial_test['correct']}/{initial_test['total']}) "
        f"loss={initial_test['loss']:.4f}"
    )

    save_json(output_dir / "config.json", asdict(cfg) | {
        "created_at": datetime.now().isoformat(),
        "num_classes": info["num_classes"],
        "initial_val": initial_val,
        "initial_test": initial_test,
    })

    if cfg.method == "classifier_only":
        results = train_classifier_only(model, train_loader, val_loader, test_loader, cfg, output_dir, logger, device)
    else:
        results = apply_prototype_classifier(
            model, train_loader, val_loader, test_loader, cfg, output_dir, logger, device, info["num_classes"]
        )

    test = results["test"]
    logger.info("=" * 70)
    logger.info(f"Best epoch   : {results['best_epoch']}")
    logger.info(f"Best val acc : {results['best_val_acc']*100:.2f}%")
    logger.info(f"TEST ACC     : {test['acc']*100:.2f}% ({test['correct']}/{test['total']})")
    logger.info(f"TEST LOSS    : {test['loss']:.4f}")
    logger.info(f"Output       : {output_dir}")
    logger.info("=" * 70)

    save_json(output_dir / "test_results.json", {
        "method": cfg.method,
        "best_epoch": results["best_epoch"],
        "best_val_acc": results["best_val_acc"],
        "test_accuracy": test["acc"],
        "test_correct": test["correct"],
        "test_total": test["total"],
        "test_loss": test["loss"],
        "initial_test_accuracy": initial_test["acc"],
        "initial_test_correct": initial_test["correct"],
        "initial_test_total": initial_test["total"],
        **({"prototype_scale": results["prototype_scale"]} if "prototype_scale" in results else {}),
    })


if __name__ == "__main__":
    main()
