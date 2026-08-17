"""
Knowledge Distillation Configuration
=====================================
Semua hyperparameter KD dipusatkan di sini.
Edit nilai-nilai ini sebelum menjalankan kd_train.py.

Konfigurasi ini mendukung beberapa metode KD. Hinton KD tetap menjadi default;
metode khusus seperti DKD, adaptive center-relation, dan ICD compactness harus
dipilih secara eksplisit melalui --kd_method beserta argumen terkait.
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
    teacher_weights: str = str(_HERE / "best_model.pth")
    teacher_config: str = ""
    teacher2_arch: str = "mobilenet_v3_small"
    teacher2_weights: str = str(_STUDENT_DIR / "Teacher" / "training_results" / "MobileNetV3Small" / "best_model.pth")

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
    augmentation_policy: str = "v1_legacy"
    train_sampler: str = "random"  # "random" or "pk"
    pk_p: int = 16                 # identities per PK batch
    pk_k: int = 4                  # samples per identity in PK batch
    teacher_center_cache: str = ""
    initial_student_weights: str = ""
    resume_training_state: str = ""
    continuation_source_epoch: int = 0
    continuation_type: str = "none"

    # ── MixUp / CutMix ───────────────────────────────────────────────────────
    # Batch-level augmentasi — meningkatkan generalisasi student
    # Set mixup_alpha=0 DAN cutmix_alpha=0 untuk disable sepenuhnya
    mixup_alpha: float      = 0.8   # Beta distribution param untuk MixUp (0=off)
    cutmix_alpha: float     = 1.0   # Beta distribution param untuk CutMix (0=off)
    mix_prob: float         = 1.0   # Probability menerapkan mix per batch (1.0=selalu)
    mix_switch_prob: float  = 0.5   # Prob memilih CutMix vs MixUp (0.5=50/50)

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

    # Method: "hinton" preserves the original logit KD path. "pairwise",
    # "embedding", and "hybrid" use biometric embedding/relation KD.
    kd_method: str = "hinton"
    icd_mode: str = "full"
    icd_bank_size: int = 5
    icd_valid_steps: int = 200
    icd_delta: float = 0.001
    icd_gamma: float = 50.0
    icd_sdc_start_epoch: int = 76
    icd_sdc_weight: float = 0.5
    icd_classification_weight: float = 0.1
    icd_logit_warmup_epochs: int = 20
    dkd_alpha: float = 1.0
    dkd_beta: float = 8.0
    dkd_warmup_epochs: int = 20
    skip_test_evaluation: bool = False
    adaface: bool = False
    adaface_m: float = 0.4
    adaface_h: float = 0.333
    adaface_s: float = 64.0
    adaface_t_alpha: float = 0.01
    ce_weight: float = 1.0
    relation_weight: float = 0.05
    embedding_weight: float = 0.0
    center_weight: float = 0.5
    feature_weight: float = 0.1
    center_scale: float = 64.0
    center_margin: float = 0.35
    relation_topk: int = 8
    relation_difference_threshold: float = 0.02
    adaptive_warmup_epochs: int = 20
    progressive_staging: bool = False
    progressive_center_start: int = 101
    progressive_relation_start: int = 201
    progressive_calibration_batches: int = 10
    progressive_center_grad_ratio: float = 0.10
    progressive_feature_grad_ratio: float = 0.05
    progressive_relation_grad_ratio: float = 0.05
    logit_kd_weight: float = 0.0
    topk_k: int = 5
    topk_weight: float = 0.05
    margin_weight: float = 0.10
    margin_m: float = 0.10
    hard_weight: float = 2.0
    hard_margin_threshold: float = 0.20
    teacher_conf_threshold: float = 0.50
    anchor_weights: str = ""          # default kosong -> pakai student_weights
    anchor_weight: float = 0.75       # KL ke checkpoint awal agar boundary tidak bergeser jauh
    anchor_temperature: float = 2.0
    teacher1_weight: float = 0.01
    teacher2_weight: float = 0.05
    teacher2_conf_threshold: float = 0.05
    teacher_agree_bonus: float = 1.5
    teacher_disagree_policy: str = "teacher2_only"
    topkd_mode: str = "lite"          # "lite" or "full"
    topkd_k: int = 20
    topkd_ce_weight: float = 1.0
    topkd_tdl_weight: float = 0.5
    topkd_contrast_weight: float = 0.05
    topkd_scale: float = 2.0
    topkd_temperature: float = 0.0    # 0.0 -> follow temperature
    topkd_include_gt: bool = True

    # ── Optimiser ────────────────────────────────────────────────────────────
    epochs: int          = 150
    lr: float            = 3e-4     # learning rate awalan (fine-tuning dari pretrained)
    lr_min: float        = 1e-6     # cosine annealing min LR
    weight_decay: float  = 0.02
    warmup_epochs: int   = 5        # warm-up linear LR sebelum cosine

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    # "cosine"  → standard cosine annealing (default)
    # "sgdr"    → cosine annealing with warm restarts (SGDR)
    scheduler: str       = "cosine"
    sgdr_T0: int         = 50       # epoch per cycle pertama (SGDR only)
    sgdr_T_mult: int     = 2        # multiplier panjang cycle berikutnya

    # ── Drop path selama KD ──────────────────────────────────────────────────
    drop_path_prob: float = 0.1     # lebih kecil dari retrain (0.2) karena sudah pretrained

    # ── Label smoothing pada CE component dari KD loss ───────────────────────
    # Catatan: KD soft labels sudah berfungsi sebagai regularizer, sehingga
    # label_smoothing tambahan bisa menyebabkan over-regularization (double smoothing).
    # Set 0.0 untuk menonaktifkan (recommended saat KD aktif).
    label_smoothing: float = 0.1

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = str(_HERE / "kd_results")
    log_interval: int = 10          # print setiap N batch

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int          = 42
    device: str        = "auto"     # "auto" → pakai cuda jika tersedia
    amp: bool          = True       # Automatic Mixed Precision (lebih cepat di GPU)
    no_pretrained_student: bool = False  # True → student inisialisasi random (from scratch)
    freeze_bn: bool = False              # True → BN pakai running stats pretrained saat KD

# Instance default — langsung di-import
KD_CFG = KDConfig()


# ─── Helper: print config ─────────────────────────────────────────────────────

def print_config(cfg: KDConfig) -> None:
    print("\n" + "=" * 60)
    print("  Knowledge Distillation Config")
    print("=" * 60)
    print(f"  Teacher         : {cfg.teacher_arch}")
    print(f"  Teacher weights : {cfg.teacher_weights}")
    if cfg.kd_method == "conservative_multiteacher":
        print(f"  Teacher 2       : {cfg.teacher2_arch}")
        print(f"  Teacher 2 weights: {cfg.teacher2_weights}")
    print(f"  Student C_init  : {cfg.student_C_init}  |  num_cells: {cfg.student_num_cells}")
    print(f"  Student weights : {cfg.student_weights}")
    print(f"  Num classes     : {cfg.num_classes}")
    print()
    print(f"  Temperature (τ) : {cfg.temperature}")
    print(f"  Alpha (CE weight): {cfg.alpha}  → KD weight: {1 - cfg.alpha:.1f}")
    print(f"  KD method       : {cfg.kd_method}")
    print(
        f"  BioKD weights   : CE={cfg.ce_weight}  relation={cfg.relation_weight}  "
        f"embedding={cfg.embedding_weight}  logit_kd={cfg.logit_kd_weight}"
    )
    if cfg.kd_method == "hard_topk":
        print(
            f"  HardTopK KD     : topk={cfg.topk_k}  topk_w={cfg.topk_weight}  "
            f"margin_w={cfg.margin_weight}  margin_m={cfg.margin_m}  "
            f"hard_w={cfg.hard_weight}"
        )
        print(
            f"                    hard_margin={cfg.hard_margin_threshold}  "
            f"teacher_conf={cfg.teacher_conf_threshold}"
        )
    if cfg.kd_method == "conservative":
        print(
            f"  Conservative KD : topk={cfg.topk_k}  topk_w={cfg.topk_weight}  "
            f"margin_w={cfg.margin_weight}  margin_m={cfg.margin_m}"
        )
        print(
            f"                    anchor_w={cfg.anchor_weight}  "
            f"anchor_T={cfg.anchor_temperature}"
        )
        print(f"                    anchor_weights={cfg.anchor_weights or cfg.student_weights}")
    if cfg.kd_method == "conservative_multiteacher":
        print(
            f"  ConsMT KD       : topk={cfg.topk_k}  t1_w={cfg.teacher1_weight}  "
            f"t2_w={cfg.teacher2_weight}"
        )
        print(
            f"                    t2_conf={cfg.teacher2_conf_threshold}  "
            f"agree_bonus={cfg.teacher_agree_bonus}  policy={cfg.teacher_disagree_policy}"
        )
        print(
            f"                    anchor_w={cfg.anchor_weight}  "
            f"anchor_T={cfg.anchor_temperature}"
        )
        print(f"                    anchor_weights={cfg.anchor_weights or cfg.student_weights}")
    if cfg.kd_method == "topkd":
        print(
            f"  Top-KD          : mode={cfg.topkd_mode}  K={cfg.topkd_k}  "
            f"CE={cfg.topkd_ce_weight}  TDL={cfg.topkd_tdl_weight}  "
            f"contrast={cfg.topkd_contrast_weight}"
        )
        print(
            f"                    scale={cfg.topkd_scale}  "
            f"T={cfg.topkd_temperature or cfg.temperature}  include_gt={cfg.topkd_include_gt}"
        )
    if cfg.kd_method == "adaptive_center_relation":
        print(
            f"  Adaptive CRD    : center={cfg.center_weight} feature={cfg.feature_weight} "
            f"relation={cfg.relation_weight} scale={cfg.center_scale} margin={cfg.center_margin}"
        )
        print(
            f"                    negative_topk={cfg.relation_topk} "
            f"difference_threshold={cfg.relation_difference_threshold} "
            f"warmup={cfg.adaptive_warmup_epochs}"
        )
    if cfg.kd_method == "icd_compactness":
        print(
            f"  ICD compactness : mode={cfg.icd_mode} bank={cfg.icd_bank_size} "
            f"valid_steps={cfg.icd_valid_steps} delta={cfg.icd_delta} gamma={cfg.icd_gamma}"
        )
        print(
            f"                    SDC start={cfg.icd_sdc_start_epoch} "
            f"SDC weight={cfg.icd_sdc_weight} ArcFace weight={cfg.icd_classification_weight}"
        )
        print(
            f"                    Logit KD weight={cfg.logit_kd_weight} "
            f"T={cfg.temperature} warmup={cfg.icd_logit_warmup_epochs}"
        )
    print()
    print(f"  Epochs          : {cfg.epochs}")
    print(f"  Batch size      : {cfg.batch_size}")
    print(f"  CutOut length   : {cfg.cutout_length}")
    print(f"  Aug policy      : {cfg.augmentation_policy}")
    print(f"  Train sampler   : {cfg.train_sampler}")
    if cfg.train_sampler == "pk":
        print(f"  PK sampler      : P={cfg.pk_p}  K={cfg.pk_k}  effective batch={cfg.pk_p * cfg.pk_k}")
    sched_desc = cfg.scheduler
    if cfg.scheduler == "sgdr":
        sched_desc += f"  T0={cfg.sgdr_T0}  T_mult={cfg.sgdr_T_mult}"
    print(f"  LR              : {cfg.lr}  →  {cfg.lr_min} ({sched_desc})")
    print(f"  Weight decay    : {cfg.weight_decay}")
    print(f"  Warmup epochs   : {cfg.warmup_epochs}")
    print(f"  Drop path prob  : {cfg.drop_path_prob}")
    print(f"  Freeze BN       : {cfg.freeze_bn}")
    print(f"  AMP             : {cfg.amp}")
    # MixUp / CutMix
    mix_enabled = cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0
    print(f"  MixUp alpha     : {cfg.mixup_alpha}  |  CutMix alpha: {cfg.cutmix_alpha}")
    print(f"  Mix prob        : {cfg.mix_prob}  |  Switch prob : {cfg.mix_switch_prob}")
    print(f"  Mix enabled     : {mix_enabled}")
    print(f"  Output dir      : {cfg.output_dir}")
    print("=" * 60 + "\n")
