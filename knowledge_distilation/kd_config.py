"""
Knowledge Distillation Configuration
=====================================
Semua hyperparameter KD dipusatkan di sini.
Edit nilai-nilai ini sebelum menjalankan kd_train.py.

Metode yang dipakai: Hinton KD (2015)
  Loss = alpha * CE(logits_student, hard_labels)
       + (1-alpha) * T^2 * KL(softmax(logits_student/T) || softmax(logits_teacher/T))

  - alpha   : bobot CE (hard target)  vs KD (soft target)
  - T       : temperature — makin besar, distribusi teacher makin "lembut"
              sehingga informasi inter-class lebih banyak ditransfer
"""

from dataclasses import dataclass, field
from pathlib import Path

# ─── Root paths ──────────────────────────────────────────────────────────────

_HERE        = Path(__file__).resolve().parent        # .../knowledge_distilation/
_STUDENT_DIR = _HERE.parent                           # .../Student/


@dataclass
class KDConfig:
    # ── Teacher ──────────────────────────────────────────────────────────────
    teacher_arch: str = "efficientnet_v2_m"
    # Path ke best_model.pth teacher (state_dict langsung)
    teacher_weights: str = str(_HERE / "best_model_teacher" / "best_model.pth")

    # ── Student ──────────────────────────────────────────────────────────────
    # Genotype diambil dari retrain/config.json (bukan dari file json terpisah)
    student_config_path: str = str(_STUDENT_DIR / "nas_results" / "retrain" / "config.json")
    # Pretrained student weights (hasil retrain terbaik)
    student_weights: str    = str(_STUDENT_DIR / "nas_results" / "retrain" / "best_model.pth")
    student_C_init: int     = 8
    student_num_cells: int  = 8
    num_classes: int        = 834
    student_dropout: float  = 0.3

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_dir: str     = str(_STUDENT_DIR.parent / "preprocessed_results")
    split_path: str   = str(_STUDENT_DIR / "split_info.json")
    input_size: int   = 224
    batch_size: int   = 64
    num_workers: int  = 4
    cutout_length: int = 16      # sama dengan retrain untuk konsistensi augmentasi

    # ── KD Hyperparameters ──────────────────────────────────────────────────
    # Temperature τ: mengontrol "kelembuatan" distribusi teacher
    #   τ=1  → distribusi asli (sharp)
    #   τ=4  → distribusi lebih rata → lebih banyak informasi inter-class
    #   Rekomendasi untuk dataset fine-grained (834 class): τ=4.0
    temperature: float = 4.0

    # Alpha: weight untuk CE loss (hard targets)
    #   alpha=0.0 → pure KD (hanya soft target)
    #   alpha=1.0 → pure CE (seperti retrain biasa)
    #   alpha=0.3 → 70% KD + 30% CE  ← recommended
    alpha: float = 0.3

    # ── Optimiser ────────────────────────────────────────────────────────────
    epochs: int          = 150
    lr: float            = 3e-4     # learning rate awalan (fine-tuning dari pretrained)
    lr_min: float        = 1e-6     # cosine annealing min LR
    weight_decay: float  = 0.02
    warmup_epochs: int   = 5        # warm-up linear LR sebelum cosine

    # ── Drop path selama KD ──────────────────────────────────────────────────
    drop_path_prob: float = 0.1     # lebih kecil dari retrain (0.2) karena sudah pretrained

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = str(_HERE / "kd_results")
    log_interval: int = 10          # print setiap N batch

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int          = 42
    device: str        = "auto"     # "auto" → pakai cuda jika tersedia
    amp: bool          = True       # Automatic Mixed Precision (lebih cepat di GPU)


# Instance default — langsung di-import
KD_CFG = KDConfig()


# ─── Helper: print config ─────────────────────────────────────────────────────

def print_config(cfg: KDConfig) -> None:
    print("\n" + "=" * 60)
    print("  Knowledge Distillation Config")
    print("=" * 60)
    print(f"  Teacher         : {cfg.teacher_arch}")
    print(f"  Teacher weights : {cfg.teacher_weights}")
    print(f"  Student C_init  : {cfg.student_C_init}  |  num_cells: {cfg.student_num_cells}")
    print(f"  Student weights : {cfg.student_weights}")
    print(f"  Num classes     : {cfg.num_classes}")
    print()
    print(f"  Temperature (τ) : {cfg.temperature}")
    print(f"  Alpha (CE weight): {cfg.alpha}  → KD weight: {1 - cfg.alpha:.1f}")
    print()
    print(f"  Epochs          : {cfg.epochs}")
    print(f"  Batch size      : {cfg.batch_size}")
    print(f"  LR              : {cfg.lr}  →  {cfg.lr_min} (cosine)")
    print(f"  Weight decay    : {cfg.weight_decay}")
    print(f"  Warmup epochs   : {cfg.warmup_epochs}")
    print(f"  Drop path prob  : {cfg.drop_path_prob}")
    print(f"  AMP             : {cfg.amp}")
    print(f"  Output dir      : {cfg.output_dir}")
    print("=" * 60 + "\n")
