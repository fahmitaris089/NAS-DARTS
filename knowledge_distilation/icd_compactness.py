"""ICD-Face-inspired intra-class compactness distillation.

This is an independent palm-vein adaptation of Yu et al. (ICCV 2023), not an
official reproduction.  The feature banks and projection layer are training
state only; the deployable student remains unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SynchronizedClassFeatureBank(nn.Module):
    """Paired teacher/student FIFO-like class banks with step validity.

    Teacher and student features always occupy the same class/slot.  When a
    class bank is full, the least-valid (oldest) slot is replaced, matching the
    feature-bank rule described by ICD-Face.
    """

    def __init__(self, num_classes: int, bank_size: int, student_dim: int,
                 teacher_dim: int, valid_steps: int) -> None:
        super().__init__()
        if min(num_classes, bank_size, student_dim, teacher_dim, valid_steps) <= 0:
            raise ValueError("Feature-bank dimensions and valid_steps must be positive")
        self.num_classes = int(num_classes)
        self.bank_size = int(bank_size)
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.valid_steps = int(valid_steps)
        self.register_buffer(
            "student_bank", torch.zeros(num_classes, bank_size, student_dim)
        )
        self.register_buffer(
            "teacher_bank", torch.zeros(num_classes, bank_size, teacher_dim)
        )
        self.register_buffer(
            "validity", torch.zeros(num_classes, bank_size, dtype=torch.long)
        )

    @torch.no_grad()
    def update(self, student_features: torch.Tensor, teacher_features: torch.Tensor,
               labels: torch.Tensor) -> None:
        student = F.normalize(student_features.detach().float(), dim=1)
        teacher = F.normalize(teacher_features.detach().float(), dim=1)
        labels = labels.detach().long()
        label_values = labels.cpu().tolist()
        if student.shape[0] != teacher.shape[0] or labels.numel() != student.shape[0]:
            raise ValueError("Teacher, student, and label batch sizes must match")
        if student.shape[1] != self.student_dim or teacher.shape[1] != self.teacher_dim:
            raise ValueError("Feature dimensions do not match the configured banks")

        for index, class_index in enumerate(label_values):
            if not 0 <= class_index < self.num_classes:
                raise ValueError(f"Class index out of range: {class_index}")
            class_validity = self.validity[class_index]
            invalid = torch.where(class_validity <= 0)[0]
            slot = int(invalid[0]) if invalid.numel() else int(class_validity.argmin())
            self.student_bank[class_index, slot].copy_(student[index])
            self.teacher_bank[class_index, slot].copy_(teacher[index])
            self.validity[class_index, slot] = self.valid_steps

        # Algorithm 1 decrements all entries after inserting the current batch.
        self.validity.sub_(1).clamp_(min=0)

    def positive_similarities(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        student = F.normalize(student_features.float(), dim=1)
        teacher = F.normalize(teacher_features.detach().float(), dim=1)
        label_values = labels.detach().cpu().tolist()
        student_scores: list[torch.Tensor] = []
        teacher_scores: list[torch.Tensor] = []
        for index, class_index in enumerate(label_values):
            valid = self.validity[class_index] > 0
            if not bool(valid.any()):
                continue
            student_scores.append(
                self.student_bank[class_index, valid] @ student[index]
            )
            teacher_scores.append(
                self.teacher_bank[class_index, valid] @ teacher[index]
            )
        if not student_scores:
            zero_s = student_features.sum() * 0.0
            zero_t = teacher_features.detach().sum() * 0.0
            return zero_s.reshape(0), zero_t.reshape(0)
        return torch.cat(student_scores), torch.cat(teacher_scores)

    @property
    def occupancy(self) -> float:
        return float((self.validity > 0).float().mean().item())


class ICDCompactnessLoss(nn.Module):
    """FCD plus optional similarity-distribution consistency and ArcFace."""

    method_label = (
        "ICD-inspired palm-vein distillation "
        "(independent closed-set adaptation of ICD-Face)"
    )

    def __init__(
        self,
        *,
        student_dim: int,
        teacher_dim: int,
        num_classes: int,
        mode: str = "full",
        bank_size: int = 5,
        valid_steps: int = 200,
        delta: float = 0.001,
        gamma: float = 50.0,
        sdc_start_epoch: int = 76,
        sdc_weight: float = 0.5,
        classification_weight: float = 0.1,
        temperature: float = 20.0,
        logit_kd_weight: float = 0.0,
        logit_warmup_epochs: int = 20,
    ) -> None:
        super().__init__()
        if mode not in {"fcd", "full"}:
            raise ValueError("ICD mode must be 'fcd' or 'full'")
        if not 0 < delta <= 2:
            raise ValueError("delta must be in (0, 2]")
        if gamma <= 0 or sdc_start_epoch <= 0:
            raise ValueError("gamma and sdc_start_epoch must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if logit_kd_weight < 0 or logit_warmup_epochs < 0:
            raise ValueError("logit KD weight and warm-up cannot be negative")
        self.mode = mode
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.sdc_start_epoch = int(sdc_start_epoch)
        self.sdc_weight = float(sdc_weight)
        self.classification_weight = float(classification_weight)
        self.temperature = float(temperature)
        self.logit_kd_weight = float(logit_kd_weight)
        self.logit_warmup_epochs = int(logit_warmup_epochs)
        self.projector = nn.Linear(student_dim, teacher_dim, bias=False)
        nn.init.orthogonal_(self.projector.weight)
        self.bank = SynchronizedClassFeatureBank(
            num_classes, bank_size, student_dim, teacher_dim, valid_steps
        )
        bins = int(round(2.0 / self.delta)) + 1
        actual_delta = 2.0 / (bins - 1)
        if not math.isclose(actual_delta, self.delta, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("delta must divide the [-1, 1] interval exactly")
        self.register_buffer("histogram_nodes", torch.linspace(-1.0, 1.0, bins))

    def _soft_distribution(self, similarities: torch.Tensor) -> torch.Tensor:
        if similarities.numel() == 0:
            return self.histogram_nodes.new_full(
                self.histogram_nodes.shape, 1.0 / self.histogram_nodes.numel()
            )
        similarities = similarities.float().clamp(-1.0, 1.0)
        nodes = self.histogram_nodes.float()
        histogram = torch.exp(
            -self.gamma * (similarities[:, None] - nodes[None, :]).square()
        ).mean(dim=0)
        return histogram / histogram.sum().clamp_min(1e-12)

    def forward(
        self,
        *,
        inference_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        classification_logits: torch.Tensor,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        projected_student = F.normalize(self.projector(student_embeddings.float()), dim=1)
        normalized_teacher = F.normalize(teacher_embeddings.detach().float(), dim=1)
        fcd = 0.5 * (projected_student - normalized_teacher).square().sum(dim=1).mean()
        classification = F.cross_entropy(classification_logits.float(), targets)

        # ArcFace inference logits are scaled cosine similarities. Temperature
        # softening is therefore applied to the inference logits, never to the
        # label-dependent margin logits used by the classification objective.
        if teacher_logits.shape != inference_logits.shape:
            raise ValueError(
                "teacher and student logits must have identical [batch, classes] shape"
            )
        student_log_prob = F.log_softmax(
            inference_logits.float() / self.temperature, dim=1
        )
        teacher_prob = F.softmax(
            teacher_logits.detach().float() / self.temperature, dim=1
        )
        logit_kd = F.kl_div(
            student_log_prob, teacher_prob, reduction="batchmean"
        ) * (self.temperature ** 2)
        if self.logit_warmup_epochs == 0:
            ramp = 1.0
        else:
            ramp = min(max(float(epoch), 0.0) / self.logit_warmup_epochs, 1.0)
        effective_logit_weight = self.logit_kd_weight * ramp

        # Updating before positive-pair construction follows Algorithm 1.
        self.bank.update(student_embeddings, teacher_embeddings, targets)
        student_scores, teacher_scores = self.bank.positive_similarities(
            student_embeddings, teacher_embeddings, targets
        )
        sdc_active = self.mode == "full" and epoch >= self.sdc_start_epoch
        if sdc_active and student_scores.numel():
            student_distribution = self._soft_distribution(student_scores)
            teacher_distribution = self._soft_distribution(teacher_scores).detach()
            sdc = F.kl_div(
                student_distribution.clamp_min(1e-12).log(),
                teacher_distribution.clamp_min(1e-12),
                reduction="sum",
            )
        else:
            sdc = student_embeddings.sum() * 0.0

        total = (
            fcd
            + self.classification_weight * classification
            + (self.sdc_weight * sdc if sdc_active else 0.0)
            + effective_logit_weight * logit_kd
        )
        with torch.no_grad():
            true_logits = inference_logits.gather(1, targets[:, None]).squeeze(1)
            masked = inference_logits.detach().clone()
            masked.scatter_(1, targets[:, None], float("-inf"))
            mean_margin = (true_logits - masked.max(dim=1).values).mean()
        breakdown = {
            "loss_total": float(total.detach()),
            "loss_ce": float(classification.detach()),
            "loss_kd": float((
                fcd
                + (self.sdc_weight * sdc if sdc_active else 0.0)
                + effective_logit_weight * logit_kd
            ).detach()),
            "loss_embedding": float(fcd.detach()),
            "loss_fcd": float(fcd.detach()),
            "loss_arcface_raw": float(classification.detach()),
            "loss_arcface_weighted": float(
                (self.classification_weight * classification).detach()
            ),
            "loss_logit_kd": float(logit_kd.detach()),
            "loss_logit_kd_weighted": float(
                (effective_logit_weight * logit_kd).detach()
            ),
            "logit_kd_effective_weight": float(effective_logit_weight),
            "loss_relation": float(sdc.detach()),
            "icd_sdc_active": float(sdc_active),
            "icd_positive_pairs": float(student_scores.numel()),
            "icd_bank_occupancy": self.bank.occupancy,
            "icd_true_class_margin": float(mean_margin.detach()),
        }
        return total, breakdown
