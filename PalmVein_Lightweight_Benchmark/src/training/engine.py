from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.evaluation.metrics import evaluate_classifier
from src.models.factory import get_classifier_parameters


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    classifier_ids = {id(parameter) for parameter in get_classifier_parameters(model)}
    for parameter in model.parameters():
        if id(parameter) not in classifier_ids:
            parameter.requires_grad = trainable


def build_optimizer(model: nn.Module, protocol: dict):
    if protocol["protocol"] == "scratch":
        return torch.optim.AdamW(
            model.parameters(), lr=float(protocol["learning_rate"]), weight_decay=float(protocol["weight_decay"])
        )
    classifier = get_classifier_parameters(model)
    classifier_ids = {id(parameter) for parameter in classifier}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in classifier_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(protocol["backbone_learning_rate"]), "group_name": "backbone"},
            {"params": classifier, "lr": float(protocol["classifier_learning_rate"]), "group_name": "classifier"},
        ],
        weight_decay=float(protocol["weight_decay"]),
    )


def scheduled_learning_rates(protocol: dict, epoch: int) -> dict[str, float]:
    warmup = int(protocol["warmup_epochs"])
    epochs = int(protocol["epochs"])
    start = float(protocol["warmup_start_factor"])
    minimum = float(protocol["minimum_learning_rate"])
    bases = {
        "default": float(protocol.get("learning_rate", protocol.get("classifier_learning_rate"))),
        "classifier": float(protocol.get("classifier_learning_rate", protocol.get("learning_rate"))),
        "backbone": float(protocol.get("backbone_learning_rate", protocol.get("learning_rate"))),
    }
    result = {}
    for name, base in bases.items():
        if epoch < warmup:
            factor = start + (1.0 - start) * (epoch + 1) / max(1, warmup)
            result[name] = base * factor
        else:
            progress = (epoch - warmup) / max(1, epochs - warmup - 1)
            result[name] = minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))
    return result


def apply_learning_rates(optimizer, rates: dict[str, float]) -> None:
    for group in optimizer.param_groups:
        group["lr"] = rates.get(group.get("group_name", "default"), rates["default"])


def train_one_epoch(model, loader, criterion, optimizer, device, gradient_clip_norm: float, freeze_backbone: bool = False) -> dict[str, float]:
    model.train()
    if freeze_backbone:
        # Frozen affine parameters alone are insufficient: BatchNorm running
        # statistics are buffers and would otherwise continue changing.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started = time.perf_counter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        if isinstance(outputs, tuple):
            logits, auxiliary = outputs
            loss = criterion(logits, targets) + 0.4 * criterion(auxiliary, targets)
        else:
            logits = outputs
            loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch = targets.size(0)
        total_loss += float(loss.item()) * batch
        total_correct += int((logits.argmax(1) == targets).sum().item())
        total_samples += batch
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "seconds": time.perf_counter() - started,
    }


def write_epoch_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, *, model, optimizer, epoch: int, best_val_loss: float, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def run_training(model, loaders, protocol: dict, device, checkpoint_dir: Path, result_dir: Path, metadata: dict, resume: Path | None = None):
    criterion = nn.CrossEntropyLoss(label_smoothing=float(protocol["label_smoothing"]))
    optimizer = build_optimizer(model, protocol)
    start_epoch = 0
    best_val_loss = float("inf")
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_val_loss = float(state["best_val_loss"])
    model.to(device)
    log_path = result_dir / "training_log.csv"
    freeze_epochs = int(protocol.get("freeze_backbone_epochs", 0))
    for epoch in range(start_epoch, int(protocol["epochs"])):
        if protocol["protocol"] == "pretrained":
            set_backbone_trainable(model, epoch >= freeze_epochs)
        rates = scheduled_learning_rates(protocol, epoch)
        apply_learning_rates(optimizer, rates)
        if hasattr(model, "set_drop_path_prob"):
            model.set_drop_path_prob(0.0)
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, float(protocol["gradient_clip_norm"]),
            freeze_backbone=protocol["protocol"] == "pretrained" and epoch < freeze_epochs,
        )
        val_metrics = evaluate_classifier(model, loaders["val"], criterion, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr_backbone": rates["backbone"],
            "lr_classifier": rates["classifier"],
            "epoch_seconds": train_metrics["seconds"],
        }
        write_epoch_log(log_path, row)
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            save_checkpoint(best_path, model=model, optimizer=optimizer, epoch=epoch, best_val_loss=best_val_loss, metadata=metadata)
        save_checkpoint(last_path, model=model, optimizer=optimizer, epoch=epoch, best_val_loss=best_val_loss, metadata=metadata)
        print(
            f"epoch={epoch + 1}/{protocol['epochs']} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.4%}",
            flush=True,
        )
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics = evaluate_classifier(model, loaders["test"], criterion, device)
    return {"best_epoch": int(best["epoch"]) + 1, "best_validation_loss": best_val_loss, "test": test_metrics, "best_checkpoint": str(best_path)}
