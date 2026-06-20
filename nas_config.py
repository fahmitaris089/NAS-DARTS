"""
NAS Configuration — P-DARTS for Palm Vein Recognition
======================================================
All hyperparameters, search space definitions, and paths
centralised here for reproducibility.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "preprocessed_results"
TEACHER_DIR  = PROJECT_ROOT / "Teacher"
SPLIT_PATH   = Path(__file__).resolve().parent / "split_info.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "nas_results"
SEARCH_DIR   = RESULTS_DIR / "search"
RETRAIN_DIR  = RESULTS_DIR / "retrain"

# ─── Dataset ──────────────────────────────────────────────────────────────────

NUM_CLASSES   = 834
INPUT_SIZE    = 224          # retrain & eval use 224×224
SEARCH_INPUT_SIZE = 112     # search phase uses smaller resolution (standard DARTS practice)
IN_CHANNELS   = 3           # grayscale repeated to 3ch (ImageNet compat)
SEED          = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Search Space Primitives ─────────────────────────────────────────────────

# ── Experiment 4 (UNIFIED): all operator families compete head-to-head ──
# Rationale: previous experiments isolated operator families across separate
# searches (Exp1=sep+dil, Exp2=sep, Exp3=mbconv), so DARTS never selected
# mbconv vs sep_conv vs dil_conv in a single search. This unified 10-op space
# lets all three families compete directly under one protocol, so the claim
# "NAS prefers MBConv" can be tested fairly. Previous spaces preserved below.
PRIMITIVES = [
    'none',           # zero output (prune this edge)
    'skip_connect',   # identity / factorized reduce
    'sep_conv_3x3',   # separable conv 3×3
    'sep_conv_5x5',   # separable conv 5×5
    'dil_conv_3x3',   # dilated conv 3×3
    'dil_conv_5x5',   # dilated conv 5×5
    'mbconv3_3x3',    # inverted residual expand=3 (lightweight MBConv)
    'mbconv6_3x3',    # inverted residual expand=6 (richer MBConv)
    'rep_conv_3x3',   # re-parameterizable conv 3×3 (fuses to 1 conv at inference)
    'rep_conv_5x5',   # re-parameterizable conv 5×5 (fuses to 1 conv at inference)
    'avg_pool_3x3',   # average pooling 3×3
    'max_pool_3x3',   # max pooling 3×3
]  # 12 ops — UNIFIED head-to-head search space (Exp5: + re-parameterizable convs)

# ── Operator inference cost (Tier-1 proxy = number of conv kernels after fusion).
# Used by the op-count latency penalty in search when --oplat_lambda > 0 and no
# device LUT is supplied. Values are validated by Pi profiling: latency tracks
# the number of conv-call operators, NOT FLOPs/params. RepConv fuses to 1 conv;
# MBConv stays 3 (PW-DW-PW); SepConv/DilConv are depthwise-separable (~2).
OP_COST_PROXY = {
    'none':         0.0,
    'skip_connect': 0.0,
    'avg_pool_3x3': 1.0,
    'max_pool_3x3': 1.0,
    'rep_conv_3x3': 1.0,
    'rep_conv_5x5': 1.0,
    'dil_conv_3x3': 2.0,
    'dil_conv_5x5': 2.0,
    'sep_conv_3x3': 2.0,
    'sep_conv_5x5': 2.0,
    'mbconv3_3x3':  3.0,
    'mbconv6_3x3':  3.0,
}

# ── Exp 3 space: MBConv-only (produced search_mobile_v2) ──
PRIMITIVES_MBCONV = [
    'none', 'skip_connect',
    'mbconv3_3x3', 'mbconv6_3x3',
    'avg_pool_3x3',
]  # 5 ops

# ── Exp 1 space: original DARTS 8-op (produced search/, dil_conv won) ──
PRIMITIVES_FULL = [
    'none', 'skip_connect',
    'sep_conv_3x3', 'sep_conv_5x5',
    'dil_conv_3x3', 'dil_conv_5x5',
    'avg_pool_3x3', 'max_pool_3x3',
]

# ─── Cell Topology ───────────────────────────────────────────────────────────

NUM_NODES         = 4     # intermediate nodes per cell
NUM_INPUT_NODES   = 2     # c_{k-2} and c_{k-1}
TOP_K_EDGES       = 2     # edges kept per node when deriving genotype

# ─── P-DARTS Progressive Search Stages ───────────────────────────────────────
# Each stage: (num_cells, epochs, ops_to_keep)
# Stage 1: shallow + all ops → Stage 2: deeper + prune → Stage 3: deepest + prune more

# ── Experiment 4 (UNIFIED): 10-op space, progressive prune 10→6→3 ──
# 25 epochs/stage (matches Exp3 search_mobile_v2 protocol; alpha_warmup=10 →
# 15 effective alpha-update epochs/stage). Halves search time vs 50 ep/stage.
# NOTE: if Stage-1 alpha plots look unconverged, rerun with epochs=50.
PDARTS_STAGES = [
    {"cells": 5,  "epochs": 25, "num_ops": 12},  # all 12 ops compete
    {"cells": 8,  "epochs": 25, "num_ops": 7},   # prune to 7 (keeps skip + ≥2 conv)
    {"cells": 11, "epochs": 25, "num_ops": 4},   # prune to 4 → final genotype
]

# Full-length stages for Experiment 2+
PDARTS_STAGES_FULL = [
    {"cells": 5,  "epochs": 50, "num_ops": 8},
    {"cells": 8,  "epochs": 50, "num_ops": 5},
    {"cells": 11, "epochs": 50, "num_ops": 3},
]

# ─── Search Hyperparameters ──────────────────────────────────────────────────

SEARCH_CFG = {
    # Supernet channels (larger for 834-class problem)
    "C_search":       16,

    # Weight optimizer (SGD)
    "w_lr":           0.025,
    "w_momentum":     0.9,
    "w_weight_decay": 3e-4,
    "w_lr_min":       0.001,

    # Architecture optimizer (Adam for alpha)
    "a_lr":           6e-4,
    "a_betas":        (0.5, 0.999),
    "a_weight_decay": 1e-3,

    # Alpha warmup: train weights only for N epochs before updating alpha
    # IMPORTANT: warmup resets per stage — must be < epochs per stage!
    # Experiment 1: 30 epochs/stage → warmup=10 → 20 effective alpha updates per stage
    "alpha_warmup_epochs": 10,

    # Training
    "batch_size":     16,
    "num_workers":    4,
    "grad_clip":      5.0,

    # P-DARTS skip-connect dropout (linearly increases)
    "skip_dropout_start": 0.0,
    "skip_dropout_end":   0.5,
    "max_skip_connect":   2,    # max skip-connects per cell

    # Label smoothing (consistent with teacher)
    "label_smoothing": 0.1,

    # Search train/val split ratio (from training set)
    "search_train_ratio": 0.5,
}

# ─── Retrain Hyperparameters ─────────────────────────────────────────────────

RETRAIN_CFG = {
    # Architecture sizing (tuned to hit 250k-400k params)
    # Auto-tuned by find_optimal_C_init() at retrain time
    "C_init":         14,       # initial channels (adjustable, auto-tuned)
    "num_cells":      8,        # total cells in derived network
    "auxiliary":      True,     # auxiliary head at 2/3 depth
    "auxiliary_weight": 0.4,

    # Optimizer: AdamW (consistent with teacher)
    "optimizer":      "AdamW",
    "lr":             1e-3,
    "weight_decay":   0.05,

    # Scheduler
    "warmup_epochs":  10,
    "warmup_factor":  0.01,
    "epochs":         600,
    "lr_min":         1e-6,

    # Regularisation
    "label_smoothing": 0.2,
    "drop_path_prob":  0.2,     # DropPath (stochastic depth)
    "dropout":         0.3,     # classifier dropout
    "grad_clip":       1.0,
    "cutout_length":   16,      # CutOut augmentation

    # Data
    "batch_size":      64,
    "num_workers":     4,
    "use_augmentation": True,

    # Param budget
    "target_params_min": 250_000,
    "target_params_max": 400_000,
}
