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
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            loss_total : scalar loss untuk backward
            breakdown  : dict dengan komponen loss untuk logging
        """
        # Hard target loss (CE dengan label smoothing)
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
