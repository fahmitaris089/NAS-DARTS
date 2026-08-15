"""Independent adaptive center-relation distillation components.

The design is inspired by AdaDistill and CoupleFace, but it is neither an
official implementation nor an equation-level reproduction of either work.
Only training-set teacher embeddings may be used to initialize the center bank.
"""

from __future__ import annotations

import hashlib
import json
import statistics
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
        progressive_staging: bool = False,
        center_start_epoch: int = 101,
        relation_start_epoch: int = 201,
        calibration_batches: int = 10,
        center_grad_ratio: float = 0.10,
        feature_grad_ratio: float = 0.05,
        relation_grad_ratio: float = 0.05,
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
        self.progressive_staging = bool(progressive_staging)
        self.center_start_epoch = int(center_start_epoch)
        self.relation_start_epoch = int(relation_start_epoch)
        self.calibration_batches = int(calibration_batches)
        self.center_grad_ratio = float(center_grad_ratio)
        self.feature_grad_ratio = float(feature_grad_ratio)
        self.relation_grad_ratio = float(relation_grad_ratio)
        self.register_buffer("calibrated_center_weight", torch.tensor(float(center_weight)))
        self.register_buffer("calibrated_feature_weight", torch.tensor(float(feature_weight)))
        self.register_buffer("calibrated_relation_weight", torch.tensor(float(relation_weight)))
        self.register_buffer("center_feature_calibrated", torch.tensor(not progressive_staging))
        self.register_buffer(
            "relation_calibrated", torch.tensor((not progressive_staging) or relation_weight == 0)
        )
        self._center_candidates: list[float] = []
        self._feature_candidates: list[float] = []
        self._relation_candidates: list[float] = []

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

    @staticmethod
    def _gradient_norm(loss: torch.Tensor, embeddings: torch.Tensor) -> float:
        gradient = torch.autograd.grad(
            loss, embeddings, retain_graph=True, create_graph=False, allow_unused=True
        )[0]
        return 0.0 if gradient is None else float(gradient.detach().float().norm())

    @staticmethod
    def _candidate_weight(target_ratio: float, ce_norm: float, component_norm: float) -> float:
        if component_norm <= 1e-12 or ce_norm <= 1e-12:
            return 0.0
        return max(1e-6, min(10.0, target_ratio * ce_norm / component_norm))

    def _progressive_weights(
        self,
        *,
        epoch: int,
        batch_index: int,
        loss_ce: torch.Tensor,
        loss_center: torch.Tensor,
        loss_feature: torch.Tensor,
        loss_relation: torch.Tensor,
        student_embeddings: torch.Tensor,
    ) -> tuple[float, float, float, dict]:
        diagnostics = {
            "grad_norm_ce": 0.0,
            "grad_norm_center": 0.0,
            "grad_norm_feature": 0.0,
            "grad_norm_relation": 0.0,
        }
        if not self.progressive_staging:
            ramp = 1.0 if self.warmup_epochs == 0 else min(epoch / self.warmup_epochs, 1.0)
            return (
                ramp * self.center_weight,
                ramp * self.feature_weight,
                ramp * self.relation_weight,
                diagnostics,
            )

        if epoch < self.center_start_epoch:
            return 0.0, 0.0, 0.0, diagnostics

        calibrating_center = (
            not bool(self.center_feature_calibrated.item())
            and epoch == self.center_start_epoch
            and batch_index < self.calibration_batches
        )
        calibrating_relation = (
            self.relation_weight > 0
            and not bool(self.relation_calibrated.item())
            and epoch == self.relation_start_epoch
            and batch_index < self.calibration_batches
        )
        if calibrating_center or calibrating_relation:
            diagnostics["grad_norm_ce"] = self._gradient_norm(loss_ce, student_embeddings)
        if calibrating_center:
            diagnostics["grad_norm_center"] = self._gradient_norm(loss_center, student_embeddings)
            diagnostics["grad_norm_feature"] = self._gradient_norm(loss_feature, student_embeddings)
            self._center_candidates.append(self._candidate_weight(
                self.center_grad_ratio, diagnostics["grad_norm_ce"], diagnostics["grad_norm_center"]
            ))
            self._feature_candidates.append(self._candidate_weight(
                self.feature_grad_ratio, diagnostics["grad_norm_ce"], diagnostics["grad_norm_feature"]
            ))
            if len(self._center_candidates) >= self.calibration_batches:
                self.calibrated_center_weight.fill_(statistics.median(self._center_candidates))
                self.calibrated_feature_weight.fill_(statistics.median(self._feature_candidates))
                self.center_feature_calibrated.fill_(True)
        if calibrating_relation:
            diagnostics["grad_norm_relation"] = self._gradient_norm(loss_relation, student_embeddings)
            self._relation_candidates.append(self._candidate_weight(
                self.relation_grad_ratio, diagnostics["grad_norm_ce"], diagnostics["grad_norm_relation"]
            ))
            if len(self._relation_candidates) >= self.calibration_batches:
                self.calibrated_relation_weight.fill_(statistics.median(self._relation_candidates))
                self.relation_calibrated.fill_(True)

        center_ramp = min(
            max(epoch - self.center_start_epoch + 1, 0) / max(self.warmup_epochs, 1), 1.0
        )
        relation_ramp = min(
            max(epoch - self.relation_start_epoch + 1, 0) / max(self.warmup_epochs, 1), 1.0
        )
        center_weight = (
            float(self.calibrated_center_weight) * center_ramp
            if bool(self.center_feature_calibrated.item()) else 0.0
        )
        feature_weight = (
            float(self.calibrated_feature_weight) * center_ramp
            if bool(self.center_feature_calibrated.item()) else 0.0
        )
        relation_weight = (
            float(self.calibrated_relation_weight) * relation_ramp
            if bool(self.relation_calibrated.item()) and epoch >= self.relation_start_epoch else 0.0
        )
        return center_weight, feature_weight, relation_weight, diagnostics

    def forward(self, logits_student: torch.Tensor, student_embeddings: torch.Tensor,
                teacher_embeddings: torch.Tensor | None, targets: torch.Tensor,
                *, epoch: int = 1, batch_index: int = 0) -> tuple[torch.Tensor, dict]:
        if targets.ndim != 1 or targets.numel() != logits_student.shape[0]:
            raise ValueError("targets must be [B] and aligned with logits")
        loss_ce = self.ce(logits_student, targets)
        if self.progressive_staging and epoch < self.center_start_epoch:
            return loss_ce, {
                "loss_total": float(loss_ce.detach()), "loss_ce": float(loss_ce.detach()),
                "loss_kd": 0.0, "loss_center": 0.0, "loss_embedding": 0.0,
                "loss_relation": 0.0, "weighted_center": 0.0,
                "weighted_feature": 0.0, "weighted_relation": 0.0,
                "center_weight_effective": 0.0, "feature_weight_effective": 0.0,
                "relation_weight_effective": 0.0, "adaptive_stage": 1,
                "positive_pairs": 0, "mined_negative_pairs": 0,
                "grad_norm_ce": 0.0, "grad_norm_center": 0.0,
                "grad_norm_feature": 0.0, "grad_norm_relation": 0.0,
            }
        if teacher_embeddings is None:
            raise ValueError("teacher embeddings are required after the CE-only stage")
        teacher = F.normalize(teacher_embeddings.detach(), dim=1)
        projected = F.normalize(self.adapter(student_embeddings), dim=1)
        adaptive_targets = self._adaptive_targets_and_update(projected.detach(), teacher, targets)

        center_logits = projected @ self.centers.detach().t()
        target_cosine = (projected * adaptive_targets.detach()).sum(1)
        center_logits = center_logits.clone()
        center_logits.scatter_(1, targets[:, None], (target_cosine - self.margin)[:, None])
        loss_center = F.cross_entropy(self.scale * center_logits, targets)
        loss_feature = (1.0 - (projected * teacher).sum(1)).mean()
        if (not self.progressive_staging) or epoch >= self.relation_start_epoch:
            loss_relation, positive_count, negative_count = self._relation_loss(
                projected, teacher, targets
            )
        else:
            loss_relation, positive_count, negative_count = self._zero(projected), 0, 0
        center_w, feature_w, relation_w, diagnostics = self._progressive_weights(
            epoch=epoch, batch_index=batch_index, loss_ce=loss_ce,
            loss_center=loss_center, loss_feature=loss_feature,
            loss_relation=loss_relation, student_embeddings=student_embeddings,
        )
        weighted_center = center_w * loss_center
        weighted_feature = feature_w * loss_feature
        weighted_relation = relation_w * loss_relation
        auxiliary = weighted_center + weighted_feature + weighted_relation
        total = loss_ce + auxiliary
        return total, {
            "loss_total": float(total.detach()),
            "loss_ce": float(loss_ce.detach()),
            "loss_kd": float(auxiliary.detach()),
            "loss_center": float(loss_center.detach()),
            "loss_embedding": float(loss_feature.detach()),
            "loss_relation": float(loss_relation.detach()),
            "weighted_center": float(weighted_center.detach()),
            "weighted_feature": float(weighted_feature.detach()),
            "weighted_relation": float(weighted_relation.detach()),
            "center_weight_effective": center_w,
            "feature_weight_effective": feature_w,
            "relation_weight_effective": relation_w,
            "adaptive_stage": (
                1 if epoch < self.center_start_epoch else
                2 if epoch < self.relation_start_epoch else 3
            ) if self.progressive_staging else 0,
            "positive_pairs": positive_count,
            "mined_negative_pairs": negative_count,
            **diagnostics,
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
