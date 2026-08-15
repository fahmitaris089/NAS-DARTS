"""Independent adaptive center-relation distillation components.

The design is inspired by AdaDistill and CoupleFace, but it is neither an
official implementation nor an equation-level reproduction of either work.
Only training-set teacher embeddings may be used to initialize the center bank.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class AdaptiveCenterRelationLoss(nn.Module):
    """CE plus adaptive class-center, feature, and batch-relation losses."""

    method_label = "Adaptive Center–Relation Distillation (inspired by AdaDistill and CoupleFace)"

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_classes: int,
        initial_centers: torch.Tensor,
        *,
        center_weight: float = 0.5,
        feature_weight: float = 0.1,
        relation_weight: float = 0.05,
        scale: float = 64.0,
        margin: float = 0.35,
        topk_negatives: int = 8,
        difference_threshold: float = 0.02,
        warmup_epochs: int = 20,
        label_smoothing: float = 0.2,
    ):
        super().__init__()
        if initial_centers.shape != (num_classes, teacher_dim):
            raise ValueError(
                f"center shape {tuple(initial_centers.shape)} != {(num_classes, teacher_dim)}"
            )
        if min(student_dim, teacher_dim, num_classes) <= 0:
            raise ValueError("embedding dimensions and num_classes must be positive")
        self.adapter = nn.Linear(student_dim, teacher_dim, bias=False)
        self.register_buffer("centers", F.normalize(initial_centers.float(), dim=1))
        self.num_classes = int(num_classes)
        self.center_weight = float(center_weight)
        self.feature_weight = float(feature_weight)
        self.relation_weight = float(relation_weight)
        self.scale = float(scale)
        self.margin = float(margin)
        self.topk_negatives = int(topk_negatives)
        self.difference_threshold = float(difference_threshold)
        self.warmup_epochs = int(warmup_epochs)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _relation_loss(self, student: torch.Tensor, teacher: torch.Tensor,
                       labels: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch = labels.numel()
        if batch < 2:
            return self._zero(student), 0, 0
        student_sim = student @ student.t()
        teacher_sim = teacher @ teacher.t()
        eye = torch.eye(batch, dtype=torch.bool, device=labels.device)
        same = labels[:, None].eq(labels[None, :]) & ~eye
        different = ~labels[:, None].eq(labels[None, :])
        learnable = (teacher_sim - student_sim).abs() >= self.difference_threshold

        positive_mask = same & learnable
        positive = F.smooth_l1_loss(
            student_sim[positive_mask], teacher_sim[positive_mask], reduction="mean"
        ) if positive_mask.any() else self._zero(student)

        mined_pairs = []
        for anchor in range(batch):
            candidates = torch.where(different[anchor] & learnable[anchor])[0]
            if candidates.numel() == 0:
                continue
            k = min(self.topk_negatives, candidates.numel())
            chosen = candidates[torch.topk(teacher_sim[anchor, candidates], k=k).indices]
            mined_pairs.extend((anchor, int(other)) for other in chosen)
        if mined_pairs:
            rows = torch.tensor([x[0] for x in mined_pairs], device=labels.device)
            cols = torch.tensor([x[1] for x in mined_pairs], device=labels.device)
            negative = F.smooth_l1_loss(
                student_sim[rows, cols], teacher_sim[rows, cols], reduction="mean"
            )
        else:
            negative = self._zero(student)
        return positive + negative, int(positive_mask.sum()), len(mined_pairs)

    @torch.no_grad()
    def _adaptive_targets_and_update(self, projected: torch.Tensor,
                                     teacher: torch.Tensor,
                                     labels: torch.Tensor) -> torch.Tensor:
        old = self.centers[labels]
        student_ability = (projected * teacher).sum(1).clamp(0.0, 1.0)
        sample_quality = (teacher * old).sum(1).clamp(0.0, 1.0)
        alpha = (student_ability * sample_quality).detach()
        candidates = F.normalize(
            alpha[:, None] * old + (1.0 - alpha[:, None]) * teacher, dim=1
        )
        targets = candidates.clone()
        # Aggregate all samples of a class before one order-invariant update.
        for label in labels.unique(sorted=True):
            mask = labels.eq(label)
            self.centers[label] = F.normalize(candidates[mask].mean(0), dim=0)
        return targets

    def forward(self, logits_student: torch.Tensor, student_embeddings: torch.Tensor,
                teacher_embeddings: torch.Tensor, targets: torch.Tensor,
                *, epoch: int = 1) -> tuple[torch.Tensor, dict]:
        if targets.ndim != 1 or targets.numel() != logits_student.shape[0]:
            raise ValueError("targets must be [B] and aligned with logits")
        teacher = F.normalize(teacher_embeddings.detach(), dim=1)
        projected = F.normalize(self.adapter(student_embeddings), dim=1)
        adaptive_targets = self._adaptive_targets_and_update(projected.detach(), teacher, targets)

        center_logits = projected @ self.centers.detach().t()
        target_cosine = (projected * adaptive_targets.detach()).sum(1)
        center_logits = center_logits.clone()
        center_logits.scatter_(1, targets[:, None], (target_cosine - self.margin)[:, None])
        loss_center = F.cross_entropy(self.scale * center_logits, targets)
        loss_feature = (1.0 - (projected * teacher).sum(1)).mean()
        loss_relation, positive_count, negative_count = self._relation_loss(
            projected, teacher, targets
        )
        loss_ce = self.ce(logits_student, targets)
        ramp = 1.0 if self.warmup_epochs == 0 else min(float(epoch) / self.warmup_epochs, 1.0)
        auxiliary = (
            self.center_weight * loss_center
            + self.feature_weight * loss_feature
            + self.relation_weight * loss_relation
        )
        total = loss_ce + ramp * auxiliary
        return total, {
            "loss_total": float(total.detach()),
            "loss_ce": float(loss_ce.detach()),
            "loss_kd": float((ramp * auxiliary).detach()),
            "loss_center": float(loss_center.detach()),
            "loss_embedding": float(loss_feature.detach()),
            "loss_relation": float(loss_relation.detach()),
            "adaptive_ramp": ramp,
            "positive_pairs": positive_count,
            "mined_negative_pairs": negative_count,
        }


def save_center_cache(path: str | Path, centers: torch.Tensor, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"centers": F.normalize(centers.cpu().float(), dim=1), "metadata": metadata}, path)


def load_center_cache(path: str | Path, expected_metadata: dict) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Stale/incompatible teacher-center cache: {mismatches}")
    centers = payload["centers"].float()
    if not torch.isfinite(centers).all():
        raise ValueError("Teacher-center cache contains non-finite values")
    if not torch.allclose(centers.norm(dim=1), torch.ones(centers.shape[0]), atol=1e-4):
        raise ValueError("Teacher centers are not L2-normalized")
    return centers
