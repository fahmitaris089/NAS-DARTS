"""
Knowledge Distillation Loss Functions
=======================================

Rekomendasi (dipakai di kd_train.py):
    → HintonKDLoss — implementasi paper asli Hinton et al. (2015)

Pilihan tersedia:
    1. HintonKDLoss      ← RECOMMENDED — KL divergence soft targets + CE hard targets
    2. SoftCEKDLoss      — CE dengan label campuran teacher/hard (tanpa temperatur ganda)
    3. KDLossWithAuxiliary  — versi HintonKD dengan auxiliary head (untuk retrain KD)

─────────────────────────────────────────────────────────────────
Mengapa KL divergence lebih baik dari pure Cross-Entropy untuk KD?
─────────────────────────────────────────────────────────────────

CE biasa: L = -sum(y_hard * log(p_student))
  → hanya melihat kelas benar, buang info inter-class dari teacher

KL divergence: L = sum(p_teacher * log(p_teacher / p_student))
  → mengukur "jarak" distribusi student dari teacher
  → memaksa student mereplikasi SELURUH distribusi kepercayaan teacher
  → misalnya: teacher yakin 70% kelas A, 20% kelas B, 10% kelas C
    → student belajar bahwa A dan B mirip (informasi dark knowledge)
  → sangat efektif untuk 834 kelas palm vein yang memiliki
    banyak kelas yang secara visual serupa

Formula Hinton KD (equation 4 di paper):
  L_total = α · CE(z_s, y) + (1-α) · T² · KL(σ(z_s/T) ‖ σ(z_t/T))

  Faktor T² muncul karena gradien KL loss terskala dengan 1/T²
  sehingga tanpa faktor ini kontribusi KD mengecil saat T besar.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. Hinton KD Loss (RECOMMENDED) ─────────────────────────────────────────

class HintonKDLoss(nn.Module):
    """
    Implementasi persis Hinton et al. "Distilling the Knowledge in a Neural Network" (2015).

    L = alpha * CE(logits_student, hard_labels)
      + (1 - alpha) * T^2 * KL(softmax(logits_student/T) || softmax(logits_teacher/T))

    Args:
        temperature (float): τ — temperature untuk melembutkan distribusi.
                             Recommended: 4.0 untuk dataset fine-grained.
        alpha (float):        bobot CE loss (hard targets). Range [0, 1].
                             alpha=0.3 → 70% dark knowledge + 30% hard label.
        label_smoothing (float): factor label smoothing pada CE loss. 0.0 = disabled.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.3,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits_student: torch.Tensor,   # [B, C] — raw logits dari student
        logits_teacher: torch.Tensor,   # [B, C] — raw logits dari teacher (no_grad)
        targets: torch.Tensor,          # [B]    — integer hard labels
        mix_targets: tuple | None = None,  # (targets_a, targets_b, lam) untuk MixUp/CutMix
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            loss_total : scalar loss untuk backward
            breakdown  : dict dengan komponen loss untuk logging

        Jika mix_targets diberikan, CE loss dihitung sebagai:
            lam * CE(logits, targets_a) + (1-lam) * CE(logits, targets_b)
        """
        # Hard target loss (CE dengan label smoothing)
        if mix_targets is not None:
            targets_a, targets_b, lam = mix_targets
            loss_ce = lam * self.ce(logits_student, targets_a) + \
                      (1.0 - lam) * self.ce(logits_student, targets_b)
        else:
            loss_ce = self.ce(logits_student, targets)

        # Soft target loss (KL divergence dengan temperature scaling)
        # F.kl_div mengharapkan input dalam log-space → pakai log_softmax untuk student
        soft_student = F.log_softmax(logits_student / self.T, dim=1)
        soft_teacher = F.softmax(logits_teacher   / self.T, dim=1)

        # reduction='batchmean': dibagi N (batch size) — paper yang merekomendasikan ini
        loss_kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean")

        # Skala dengan T² untuk mengkompensasi gradien yang terskala 1/T²
        loss_kd = (self.T ** 2) * loss_kl

        # Total loss
        loss_total = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd

        breakdown = {
            "loss_ce"    : loss_ce.item(),
            "loss_kl"    : loss_kl.item(),
            "loss_kd"    : loss_kd.item(),  # loss_kl * T^2
            "loss_total" : loss_total.item(),
        }

        return loss_total, breakdown


# ─── 2. Soft-CE KD Loss (alternatif sederhana) ────────────────────────────────

class SoftCEKDLoss(nn.Module):
    """
    Alternatif: Cross-Entropy dengan soft labels (campuran teacher + one-hot).

    L = CE(logits_student, alpha * one_hot(y) + (1-alpha) * softmax(logits_teacher))

    Kelebihan  : lebih simple, tidak perlu T^2 scaling.
    Kekurangan : tidak ada temperature scaling → inter-class similarity kurang detail.
                 Tidak setara dengan Hinton KD saat T > 1.

    Pakai ini jika teacher accuracy sudah hampir 100% dan distribusinya terlalu "peaky",
    sehingga temperatur tidak terlalu membantu (jarang terjadi).
    """

    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        num_classes = logits_student.size(1)
        device      = logits_student.device

        # One-hot hard targets
        one_hot = torch.zeros(targets.size(0), num_classes, device=device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Soft targets dari teacher
        soft_labels = F.softmax(logits_teacher, dim=1)

        # Mixed labels
        mixed = self.alpha * one_hot + (1.0 - self.alpha) * soft_labels

        # CE dengan mixed labels
        log_probs  = F.log_softmax(logits_student, dim=1)
        loss_total = -(mixed * log_probs).sum(dim=1).mean()

        breakdown = {
            "loss_ce"    : F.cross_entropy(logits_student, targets).item(),
            "loss_total" : loss_total.item(),
        }
        return loss_total, breakdown


# ─── 3. Biometric representation KD losses ──────────────────────────────────

class PairwiseRelationKDLoss(nn.Module):
    """
    Match teacher/student pairwise cosine-similarity structure within a batch.

    This is dimension-safe: student and teacher embeddings may have different
    dimensionality because only their [B, B] similarity matrices are compared.
    """

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        student_norm = F.normalize(student_embeddings, p=2, dim=1)
        teacher_norm = F.normalize(teacher_embeddings, p=2, dim=1)

        student_sim = student_norm @ student_norm.t()
        teacher_sim = teacher_norm @ teacher_norm.t()

        return F.mse_loss(student_sim, teacher_sim)


class ProjectedEmbeddingKDLoss(nn.Module):
    """
    Project student embeddings to teacher dimensionality and match normalized
    embeddings. The projection is trainable and should be included in optimizer.
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.projection = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.projection(student_embeddings)
        projected = F.normalize(projected, p=2, dim=1)
        teacher_norm = F.normalize(teacher_embeddings, p=2, dim=1)
        return F.mse_loss(projected, teacher_norm)


class HybridBiometricKDLoss(nn.Module):
    """
    CE + optional pairwise relation KD + optional projected embedding KD +
    optional logit KD. Intended for biometric/fine-grained identity models.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        relation_weight: float = 0.05,
        embedding_weight: float = 0.0,
        logit_kd_weight: float = 0.0,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        student_dim: int | None = None,
        teacher_dim: int | None = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.relation_weight = relation_weight
        self.embedding_weight = embedding_weight
        self.logit_kd_weight = logit_kd_weight

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.relation = PairwiseRelationKDLoss()
        self.logit_kd = HintonKDLoss(
            temperature=temperature,
            alpha=0.0,
            label_smoothing=0.0,
        )

        if embedding_weight > 0:
            if student_dim is None or teacher_dim is None:
                raise ValueError("student_dim and teacher_dim are required when embedding_weight > 0")
            self.embedding = ProjectedEmbeddingKDLoss(student_dim, teacher_dim)
        else:
            self.embedding = None

    def forward(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        targets: torch.Tensor,
        mix_targets: tuple | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if mix_targets is not None:
            targets_a, targets_b, lam = mix_targets
            loss_ce = lam * self.ce(logits_student, targets_a) + \
                      (1.0 - lam) * self.ce(logits_student, targets_b)
        else:
            loss_ce = self.ce(logits_student, targets)

        zero = logits_student.new_tensor(0.0)
        loss_relation = self.relation(student_embeddings, teacher_embeddings) \
            if self.relation_weight > 0 else zero

        loss_embedding = self.embedding(student_embeddings, teacher_embeddings) \
            if self.embedding is not None and self.embedding_weight > 0 else zero

        if self.logit_kd_weight > 0:
            loss_logit_kd, logit_breakdown = self.logit_kd(logits_student, logits_teacher, targets)
            loss_logit_kd = loss_logit_kd
            loss_kl = logits_student.new_tensor(logit_breakdown.get("loss_kl", 0.0))
        else:
            loss_logit_kd = zero
            loss_kl = zero

        loss_total = (
            self.ce_weight * loss_ce +
            self.relation_weight * loss_relation +
            self.embedding_weight * loss_embedding +
            self.logit_kd_weight * loss_logit_kd
        )

        breakdown = {
            "loss_ce": loss_ce.item(),
            "loss_relation": loss_relation.item(),
            "loss_embedding": loss_embedding.item(),
            "loss_logit_kd": loss_logit_kd.item(),
            "loss_kl": loss_kl.item(),
            "loss_kd": (
                self.relation_weight * loss_relation +
                self.embedding_weight * loss_embedding +
                self.logit_kd_weight * loss_logit_kd
            ).item(),
            "loss_total": loss_total.item(),
        }
        return loss_total, breakdown


class HardTopKMarginKDLoss(nn.Module):
    """
    Hard-sample KD for fine-grained identity boundaries.

    The teacher provides a compact top-k distribution, while a margin-ranking
    term pushes the true class logit above the student's best wrong class.
    Hard samples are weighted online from the current student batch.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        topk_k: int = 5,
        topk_weight: float = 0.05,
        margin_weight: float = 0.10,
        margin_m: float = 0.10,
        hard_weight: float = 2.0,
        hard_margin_threshold: float = 0.20,
        teacher_conf_threshold: float = 0.50,
        temperature: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if topk_k <= 0:
            raise ValueError("topk_k must be positive")
        self.ce_weight = float(ce_weight)
        self.topk_k = int(topk_k)
        self.topk_weight = float(topk_weight)
        self.margin_weight = float(margin_weight)
        self.margin_m = float(margin_m)
        self.hard_weight = float(hard_weight)
        self.hard_margin_threshold = float(hard_margin_threshold)
        self.teacher_conf_threshold = float(teacher_conf_threshold)
        self.temperature = float(temperature)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def forward(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        num_classes = logits_student.size(1)
        k = min(self.topk_k, num_classes)
        batch_indices = torch.arange(targets.size(0), device=targets.device)

        with torch.no_grad():
            teacher_probs_full = F.softmax(logits_teacher, dim=1)
            teacher_conf, _ = teacher_probs_full.max(dim=1)
            teacher_top_values, teacher_top_indices = torch.topk(logits_teacher, k=k, dim=1)

            student_pred = logits_student.argmax(dim=1)
            sorted_student = torch.argsort(logits_student, dim=1, descending=True)
            true_rank = (sorted_student == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1

        topk_student_logits = logits_student.gather(1, teacher_top_indices)
        teacher_top_dist = F.softmax(teacher_top_values / self.temperature, dim=1)
        student_top_log_dist = F.log_softmax(topk_student_logits / self.temperature, dim=1)
        topk_kd_per_sample = F.kl_div(
            student_top_log_dist,
            teacher_top_dist,
            reduction="none",
        ).sum(dim=1) * (self.temperature ** 2)

        true_logits = logits_student[batch_indices, targets]
        masked_logits = logits_student.masked_fill(
            F.one_hot(targets, num_classes=num_classes).bool(),
            float("-inf"),
        )
        best_wrong_logits = masked_logits.max(dim=1).values
        student_margin = true_logits - best_wrong_logits
        margin_per_sample = F.relu(self.margin_m - student_margin)

        ce_per_sample = self.ce(logits_student, targets)

        hard_mask = (
            (student_pred != targets)
            | (true_rank > 1)
            | (student_margin < self.hard_margin_threshold)
        )
        teacher_valid = teacher_conf >= self.teacher_conf_threshold
        hard_mask = hard_mask & teacher_valid

        weights = torch.ones_like(ce_per_sample)
        weights = torch.where(hard_mask, weights * self.hard_weight, weights)

        loss_ce = self._weighted_mean(ce_per_sample, weights)
        loss_topk = self._weighted_mean(topk_kd_per_sample, weights)
        loss_margin = self._weighted_mean(margin_per_sample, weights)

        loss_total = (
            self.ce_weight * loss_ce
            + self.topk_weight * loss_topk
            + self.margin_weight * loss_margin
        )

        breakdown = {
            "loss_ce": loss_ce.item(),
            "loss_topk": loss_topk.item(),
            "loss_margin": loss_margin.item(),
            "hard_ratio": hard_mask.float().mean().item(),
            "avg_true_rank": true_rank.float().mean().item(),
            "loss_kd": (self.topk_weight * loss_topk + self.margin_weight * loss_margin).item(),
            "loss_total": loss_total.item(),
        }
        return loss_total, breakdown


class ConservativeAnchorKDLoss(nn.Module):
    """
    Conservative KD for an already strong student.

    The student learns gently from the high-accuracy teacher while a frozen
    anchor student (usually the original retrain checkpoint) prevents broad
    decision-boundary drift. This is useful when the baseline is already
    99%+ and aggressive KD tends to trade old errors for new errors.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        topk_k: int = 5,
        topk_weight: float = 0.01,
        margin_weight: float = 0.05,
        margin_m: float = 1.0,
        anchor_weight: float = 0.75,
        temperature: float = 2.0,
        anchor_temperature: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if topk_k <= 0:
            raise ValueError("topk_k must be positive")
        if anchor_weight <= 0:
            raise ValueError("anchor_weight must be positive for conservative KD")

        self.ce_weight = float(ce_weight)
        self.topk_k = int(topk_k)
        self.topk_weight = float(topk_weight)
        self.margin_weight = float(margin_weight)
        self.margin_m = float(margin_m)
        self.anchor_weight = float(anchor_weight)
        self.temperature = float(temperature)
        self.anchor_temperature = float(anchor_temperature)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @staticmethod
    def _temperature_kl(
        logits_student: torch.Tensor,
        logits_target: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        log_probs_student = F.log_softmax(logits_student / temperature, dim=1)
        probs_target = F.softmax(logits_target / temperature, dim=1)
        return F.kl_div(
            log_probs_student,
            probs_target,
            reduction="batchmean",
        ) * (temperature ** 2)

    def forward(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        logits_anchor: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        num_classes = logits_student.size(1)
        k = min(self.topk_k, num_classes)
        batch_indices = torch.arange(targets.size(0), device=targets.device)

        loss_ce = self.ce(logits_student, targets)

        with torch.no_grad():
            teacher_top_values, teacher_top_indices = torch.topk(logits_teacher, k=k, dim=1)

        topk_student_logits = logits_student.gather(1, teacher_top_indices)
        teacher_top_dist = F.softmax(teacher_top_values / self.temperature, dim=1)
        student_top_log_dist = F.log_softmax(topk_student_logits / self.temperature, dim=1)
        loss_topk = F.kl_div(
            student_top_log_dist,
            teacher_top_dist,
            reduction="batchmean",
        ) * (self.temperature ** 2)

        true_logits = logits_student[batch_indices, targets]
        masked_logits = logits_student.masked_fill(
            F.one_hot(targets, num_classes=num_classes).bool(),
            float("-inf"),
        )
        best_wrong_logits = masked_logits.max(dim=1).values
        loss_margin = F.relu(self.margin_m - (true_logits - best_wrong_logits)).mean()

        loss_anchor = self._temperature_kl(
            logits_student,
            logits_anchor,
            self.anchor_temperature,
        )

        loss_total = (
            self.ce_weight * loss_ce
            + self.topk_weight * loss_topk
            + self.margin_weight * loss_margin
            + self.anchor_weight * loss_anchor
        )

        breakdown = {
            "loss_ce": loss_ce.item(),
            "loss_topk": loss_topk.item(),
            "loss_margin": loss_margin.item(),
            "loss_anchor": loss_anchor.item(),
            "loss_kd": (
                self.topk_weight * loss_topk
                + self.margin_weight * loss_margin
                + self.anchor_weight * loss_anchor
            ).item(),
            "loss_total": loss_total.item(),
        }
        return loss_total, breakdown


class ConservativeMultiTeacherKDLoss(nn.Module):
    """
    Conservative KD with two complementary teachers.

    Teacher 1 is a stable high-accuracy teacher. Teacher 2 is a complementary
    teacher whose top-k signal is selectively weighted by confidence and
    agreement with the hard label. A frozen anchor student keeps the model close
    to the known-good checkpoint.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        topk_k: int = 5,
        teacher1_weight: float = 0.01,
        teacher2_weight: float = 0.05,
        teacher2_conf_threshold: float = 0.05,
        teacher_agree_bonus: float = 1.5,
        teacher_disagree_policy: str = "teacher2_only",
        anchor_weight: float = 0.5,
        temperature: float = 2.0,
        anchor_temperature: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if topk_k <= 0:
            raise ValueError("topk_k must be positive")
        if teacher_disagree_policy not in {"conservative", "teacher2_only", "weighted"}:
            raise ValueError("teacher_disagree_policy must be conservative, teacher2_only, or weighted")

        self.ce_weight = float(ce_weight)
        self.topk_k = int(topk_k)
        self.teacher1_weight = float(teacher1_weight)
        self.teacher2_weight = float(teacher2_weight)
        self.teacher2_conf_threshold = float(teacher2_conf_threshold)
        self.teacher_agree_bonus = float(teacher_agree_bonus)
        self.teacher_disagree_policy = teacher_disagree_policy
        self.anchor_weight = float(anchor_weight)
        self.temperature = float(temperature)
        self.anchor_temperature = float(anchor_temperature)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @staticmethod
    def _temperature_kl(
        logits_student: torch.Tensor,
        logits_target: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        return F.kl_div(
            F.log_softmax(logits_student / temperature, dim=1),
            F.softmax(logits_target / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def _topk_kd_per_sample(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
    ) -> torch.Tensor:
        k = min(self.topk_k, logits_student.size(1))
        teacher_top_values, teacher_top_indices = torch.topk(logits_teacher, k=k, dim=1)
        topk_student_logits = logits_student.gather(1, teacher_top_indices)
        teacher_top_dist = F.softmax(teacher_top_values / self.temperature, dim=1)
        student_top_log_dist = F.log_softmax(topk_student_logits / self.temperature, dim=1)
        return F.kl_div(
            student_top_log_dist,
            teacher_top_dist,
            reduction="none",
        ).sum(dim=1) * (self.temperature ** 2)

    def forward(
        self,
        logits_student: torch.Tensor,
        logits_teacher1: torch.Tensor,
        logits_teacher2: torch.Tensor,
        logits_anchor: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        loss_ce = self.ce(logits_student, targets)
        loss_anchor = self._temperature_kl(
            logits_student,
            logits_anchor,
            self.anchor_temperature,
        )

        teacher1_topk = self._topk_kd_per_sample(logits_student, logits_teacher1)
        teacher2_topk = self._topk_kd_per_sample(logits_student, logits_teacher2)

        with torch.no_grad():
            teacher1_pred = logits_teacher1.argmax(dim=1)
            teacher2_probs = F.softmax(logits_teacher2, dim=1)
            teacher2_conf, teacher2_pred = teacher2_probs.max(dim=1)

            teacher1_correct = teacher1_pred == targets
            teacher2_correct = teacher2_pred == targets
            teacher2_valid = teacher2_conf >= self.teacher2_conf_threshold
            teachers_agree_correct = teacher1_correct & teacher2_correct

            teacher2_weights = torch.zeros_like(teacher2_conf)
            if self.teacher_disagree_policy == "conservative":
                teacher2_weights = torch.where(teachers_agree_correct, torch.ones_like(teacher2_weights), teacher2_weights)
            elif self.teacher_disagree_policy == "teacher2_only":
                teacher2_weights = torch.where(teacher2_valid, torch.ones_like(teacher2_weights), teacher2_weights)
            else:
                teacher2_weights = torch.where(teacher2_valid, teacher2_conf, teacher2_weights)

            teacher2_weights = torch.where(
                teachers_agree_correct & teacher2_valid,
                teacher2_weights * self.teacher_agree_bonus,
                teacher2_weights,
            )

        loss_teacher1 = teacher1_topk.mean()
        loss_teacher2 = self._weighted_mean(teacher2_topk, teacher2_weights)
        teacher2_active = (teacher2_weights > 0).float().mean()

        loss_total = (
            self.ce_weight * loss_ce
            + self.anchor_weight * loss_anchor
            + self.teacher1_weight * loss_teacher1
            + self.teacher2_weight * loss_teacher2
        )

        breakdown = {
            "loss_ce": loss_ce.item(),
            "loss_anchor": loss_anchor.item(),
            "loss_teacher1_kd": loss_teacher1.item(),
            "loss_teacher2_kd": loss_teacher2.item(),
            "teacher2_active": teacher2_active.item(),
            "teacher2_correct_ratio": teacher2_correct.float().mean().item(),
            "teacher_agree_correct_ratio": teachers_agree_correct.float().mean().item(),
            "loss_kd": (
                self.anchor_weight * loss_anchor
                + self.teacher1_weight * loss_teacher1
                + self.teacher2_weight * loss_teacher2
            ).item(),
            "loss_total": loss_total.item(),
        }
        return loss_total, breakdown


class TopKDLoss(nn.Module):
    """
    Top-scaled logit KD for high-class-count fine-grained recognition.

    Lite mode uses CE + a Top-K decoupled logit alignment:
      - teacher Top-K logits are amplified by rank-dependent scaling,
      - the ground-truth class is forced into the Top-K set when missing,
      - student aligns to the scaled Top-K distribution plus the non-Top-K
        probability mass.

    Full mode adds symmetric logit-level contrastive alignment between each
    student sample and its matching teacher sample in the batch.
    """

    def __init__(
        self,
        mode: str = "lite",
        topkd_k: int = 20,
        ce_weight: float = 1.0,
        tdl_weight: float = 0.5,
        contrast_weight: float = 0.05,
        scale: float = 2.0,
        temperature: float = 20.0,
        include_gt: bool = True,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if mode not in {"lite", "full"}:
            raise ValueError("TopKDLoss mode must be 'lite' or 'full'")
        if topkd_k <= 0:
            raise ValueError("topkd_k must be positive")
        if scale < 1.0:
            raise ValueError("topkd_scale should be >= 1.0")

        self.mode = mode
        self.topkd_k = int(topkd_k)
        self.ce_weight = float(ce_weight)
        self.tdl_weight = float(tdl_weight)
        self.contrast_weight = float(contrast_weight)
        self.scale = float(scale)
        self.temperature = float(temperature)
        self.include_gt = bool(include_gt)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _scaled_teacher_topk(
        self,
        logits_teacher: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_classes = logits_teacher.size(1)
        k = min(self.topkd_k, num_classes)
        top_values, top_indices = torch.topk(logits_teacher, k=k, dim=1)

        if self.include_gt:
            with torch.no_grad():
                contains_gt = (top_indices == targets.unsqueeze(1)).any(dim=1)
                missing_gt = ~contains_gt
            if missing_gt.any():
                top_indices = top_indices.clone()
                top_values = top_values.clone()
                top_indices[missing_gt, -1] = targets[missing_gt]
                top_values[missing_gt, -1] = logits_teacher[missing_gt, targets[missing_gt]]

        rank_weights = torch.linspace(
            self.scale,
            1.0,
            steps=k,
            device=logits_teacher.device,
            dtype=logits_teacher.dtype,
        ).unsqueeze(0)
        scaled_top_values = top_values * rank_weights

        top_mask = torch.zeros_like(logits_teacher, dtype=torch.bool)
        top_mask.scatter_(1, top_indices, True)

        scaled_teacher_full = logits_teacher.clone()
        scaled_teacher_full.scatter_(1, top_indices, scaled_top_values)
        return top_indices, scaled_top_values, top_mask, scaled_teacher_full

    def _tdl_loss(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        top_indices, scaled_top_values, top_mask, scaled_teacher_full = self._scaled_teacher_topk(
            logits_teacher, targets
        )

        top_student_logits = logits_student.gather(1, top_indices)
        teacher_top_dist = F.softmax(scaled_top_values / self.temperature, dim=1)
        student_top_log_dist = F.log_softmax(top_student_logits / self.temperature, dim=1)
        loss_topk_kl = F.kl_div(
            student_top_log_dist,
            teacher_top_dist,
            reduction="batchmean",
        ) * (self.temperature ** 2)

        loss_topk_cos = (
            1.0 - F.cosine_similarity(top_student_logits, scaled_top_values, dim=1)
        ).mean()

        student_probs = F.softmax(logits_student / self.temperature, dim=1)
        teacher_probs = F.softmax(scaled_teacher_full / self.temperature, dim=1)
        non_top_mask = ~top_mask
        student_non_top_mass = (student_probs * non_top_mask).sum(dim=1)
        teacher_non_top_mass = (teacher_probs * non_top_mask).sum(dim=1)
        loss_non_top = F.mse_loss(student_non_top_mass, teacher_non_top_mass)

        loss_tdl = loss_topk_kl + loss_topk_cos + loss_non_top
        breakdown = {
            "loss_topk": loss_topk_kl.item(),
            "loss_topk_cos": loss_topk_cos.item(),
            "loss_non_top": loss_non_top.item(),
            "loss_tdl": loss_tdl.item(),
        }
        return loss_tdl, breakdown, scaled_teacher_full

    def _contrastive_loss(
        self,
        logits_student: torch.Tensor,
        scaled_teacher_full: torch.Tensor,
    ) -> torch.Tensor:
        student_norm = F.normalize(logits_student, p=2, dim=1)
        teacher_norm = F.normalize(scaled_teacher_full.detach(), p=2, dim=1)
        logits_st = student_norm @ teacher_norm.t() / max(self.temperature, 1e-6)
        labels = torch.arange(logits_student.size(0), device=logits_student.device)
        loss_st = F.cross_entropy(logits_st, labels)
        loss_ts = F.cross_entropy(logits_st.t(), labels)
        return 0.5 * (loss_st + loss_ts)

    def forward(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        loss_ce = self.ce(logits_student, targets)
        loss_tdl, tdl_breakdown, scaled_teacher_full = self._tdl_loss(
            logits_student, logits_teacher, targets
        )

        if self.mode == "full" and self.contrast_weight > 0:
            loss_contrast = self._contrastive_loss(logits_student, scaled_teacher_full)
        else:
            loss_contrast = logits_student.new_tensor(0.0)

        loss_topkd = self.tdl_weight * loss_tdl + self.contrast_weight * loss_contrast
        loss_total = self.ce_weight * loss_ce + loss_topkd

        breakdown = {
            "loss_ce": loss_ce.item(),
            "loss_tdl": loss_tdl.item(),
            "loss_contrast": loss_contrast.item(),
            "loss_topkd": loss_topkd.item(),
            "loss_kd": loss_topkd.item(),
            "loss_total": loss_total.item(),
            "topkd_k": float(min(self.topkd_k, logits_student.size(1))),
            **tdl_breakdown,
        }
        return loss_total, breakdown


# ─── 3. KD Loss with Auxiliary (untuk student yang masih pakai aux head) ─────

class KDLossWithAuxiliary(nn.Module):
    """
    Hinton KD dengan dukungan auxiliary head dari EvalNetwork.

    Saat student.training=True dan auxiliary=True, forward() mengembalikan
    (logits_main, logits_aux). Loss:
        L = HintonKD(logits_main, logits_teacher) + 0.4 * CE(logits_aux, targets)

    Gunakan ini HANYA jika student di-load dengan auxiliary=True.
    Pada KD disarankan auxiliary=False untuk simplisitas.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.3,
                 aux_weight: float = 0.4, label_smoothing: float = 0.1):
        super().__init__()
        self.kd  = HintonKDLoss(temperature, alpha, label_smoothing)
        self.ce  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux_weight = aux_weight

    def forward(self, student_output, logits_teacher, targets):
        if isinstance(student_output, tuple):
            logits_main, logits_aux = student_output
            loss_main, breakdown = self.kd(logits_main, logits_teacher, targets)
            loss_aux = self.ce(logits_aux, targets)
            loss_total = loss_main + self.aux_weight * loss_aux
            breakdown["loss_aux"]   = loss_aux.item()
            breakdown["loss_total"] = loss_total.item()
            return loss_total, breakdown, logits_main
        else:
            loss_total, breakdown = self.kd(student_output, logits_teacher, targets)
            return loss_total, breakdown, student_output


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_kd_loss(method: str = "hinton", **kwargs) -> nn.Module:
    """
    Factory untuk memilih loss function.

    Args:
        method: "hinton" (default) | "soft_ce"
        **kwargs: temperature, alpha, label_smoothing, dll.

    Returns:
        HintonKDLoss atau SoftCEKDLoss
    """
    if method == "hinton":
        return HintonKDLoss(
            temperature     = kwargs.get("temperature", 4.0),
            alpha           = kwargs.get("alpha", 0.3),
            label_smoothing = kwargs.get("label_smoothing", 0.1),
        )
    elif method == "soft_ce":
        return SoftCEKDLoss(
            alpha = kwargs.get("alpha", 0.3),
        )
    else:
        raise ValueError(f"Unknown KD method: {method}. Pilih 'hinton' atau 'soft_ce'.")
