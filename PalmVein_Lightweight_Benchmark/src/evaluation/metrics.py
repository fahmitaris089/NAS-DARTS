from __future__ import annotations

import math

import torch


@torch.inference_mode()
def evaluate_classifier(model, loader, criterion, device) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, targets)
        batch = targets.size(0)
        total_loss += float(loss.item()) * batch
        total_correct += int((logits.argmax(1) == targets).sum().item())
        total_samples += batch
    if total_samples == 0:
        raise ValueError("Evaluation loader is empty")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "correct": total_correct,
        "samples": total_samples,
    }


def sample_standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
