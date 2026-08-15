"""
Knowledge Distillation Training — EfficientNet-V2-M → NAS Student
=================================================================
Teacher : EfficientNet-V2-M  (100% train acc, frozen)
Student : EvalNetwork (P-DARTS, C_init from student_config, params vary)
Method  : Hinton KD  — KL divergence (soft) + CE (hard)

Cara pakai:
    cd Student/
    python knowledge_distilation/kd_train.py

Override config via argparse:
    python knowledge_distilation/kd_train.py \
        --temperature 6.0 \
        --alpha 0.2 \
        --epochs 200 \
        --lr 1e-4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows (fixes UnicodeEncodeError for Greek chars like α, τ)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress spurious LR scheduler warning caused by scaler.step() interaction
warnings.filterwarnings(
    "ignore",
    message=r".*lr_scheduler\.step.*before.*optimizer\.step.*",
)

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve
from torch.amp import GradScaler, autocast

# ─── Pastikan root project ada di path untuk import model_eval, dll. ─────────
_HERE        = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from genotypes import dict_to_genotype
from kd_config import KD_CFG, KDConfig, print_config
from kd_loss import (
    HintonKDLoss,
    DecoupledKDLoss,
    HybridBiometricKDLoss,
    HardTopKMarginKDLoss,
    ConservativeAnchorKDLoss,
    ConservativeMultiTeacherKDLoss,
    TopKDLoss,
)
from model_eval import EvalNetwork
from adaface import replace_linear_with_adaface
from palm_vein_dataset import PalmVeinDataset, create_retrain_dataloaders, get_transforms
from torch.utils.data import DataLoader
from adaptive_center_relation import (
    AdaptiveCenterRelationLoss, load_center_cache, save_center_cache,
    sha256_file, stable_json_hash,
)


# ─── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("kd_train")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(output_dir / "kd_train.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Use a stream wrapper with UTF-8 to avoid cp1252 errors on Windows
    import io
    utf8_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sh = logging.StreamHandler(utf8_stream)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ─── Argparse (override KDConfig fields) ─────────────────────────────────────

def parse_args(cfg: KDConfig) -> KDConfig:
    parser = argparse.ArgumentParser(description="KD Training: EfficientNet-V2-M → NAS Student")

    parser.add_argument("--teacher_arch",      default=cfg.teacher_arch,
                        help="Teacher architecture. Pilihan: efficientnet_v2_m, "
                             "efficientnet_b4, densenet121, inception_v3, resnet50, "
                             "convnext_base, regnet_y_16gf, mobilenet_v3_large, vgg16")
    parser.add_argument("--teacher_weights",   default=cfg.teacher_weights)
    parser.add_argument("--teacher2_arch", default=cfg.teacher2_arch,
                        help="Second teacher architecture for conservative_multiteacher")
    parser.add_argument("--teacher2_weights", default=cfg.teacher2_weights,
                        help="Second teacher weights for conservative_multiteacher")
    parser.add_argument("--student_weights",   default=cfg.student_weights)
    parser.add_argument("--student_config",    default=cfg.student_config_path)
    parser.add_argument("--output_dir", default=None,
                        help="Folder hasil. Default: auto-generate dari parameter.")
    parser.add_argument("--data_dir", default=cfg.data_dir,
                        help="Dataset preprocessed directory")
    parser.add_argument("--split_path", default=cfg.split_path,
                        help="Split info JSON path")

    parser.add_argument("--temperature", type=float, default=cfg.temperature,
                        help="KD temperature τ (default: 4.0)")
    parser.add_argument("--alpha",       type=float, default=cfg.alpha,
                        help="CE weight α. KD weight = 1-α (default: 0.3)")
    parser.add_argument("--kd_method", default=cfg.kd_method,
                        choices=[
                            "hinton", "dkd", "pairwise", "embedding", "hybrid",
                            "hard_topk", "conservative", "conservative_multiteacher",
                            "topkd", "adaptive_center_relation",
                        ],
                        help="KD loss method. 'hinton' preserves the original logit KD path.")
    parser.add_argument("--dkd_alpha", type=float, default=cfg.dkd_alpha,
                        help="Target-class DKD weight")
    parser.add_argument("--dkd_beta", type=float, default=cfg.dkd_beta,
                        help="Non-target-class DKD weight")
    parser.add_argument("--dkd_warmup_epochs", type=int, default=cfg.dkd_warmup_epochs,
                        help="Linear DKD loss warm-up in epochs")
    parser.add_argument("--adaface", action="store_true", default=cfg.adaface,
                        help="Use AdaFace classification head from student config/checkpoint")
    parser.add_argument("--adaface_m", type=float, default=cfg.adaface_m)
    parser.add_argument("--adaface_h", type=float, default=cfg.adaface_h)
    parser.add_argument("--adaface_s", type=float, default=cfg.adaface_s)
    parser.add_argument("--adaface_t_alpha", type=float, default=cfg.adaface_t_alpha)
    parser.add_argument("--skip-test-evaluation", action="store_true",
                        help="Do not inspect the test split during screening")
    parser.add_argument("--ce_weight", type=float, default=cfg.ce_weight,
                        help="CE weight for pairwise/embedding/hybrid KD")
    parser.add_argument("--relation_weight", type=float, default=cfg.relation_weight,
                        help="Pairwise relation loss weight")
    parser.add_argument("--center_weight", type=float, default=cfg.center_weight)
    parser.add_argument("--feature_weight", type=float, default=cfg.feature_weight)
    parser.add_argument("--center_scale", type=float, default=cfg.center_scale)
    parser.add_argument("--center_margin", type=float, default=cfg.center_margin)
    parser.add_argument("--relation_topk", type=int, default=cfg.relation_topk)
    parser.add_argument("--relation_difference_threshold", type=float,
                        default=cfg.relation_difference_threshold)
    parser.add_argument("--adaptive_warmup_epochs", type=int,
                        default=cfg.adaptive_warmup_epochs)
    parser.add_argument("--progressive_staging", action="store_true",
                        help="Use fixed CE/center/relation stages with gradient-ratio calibration")
    parser.add_argument("--progressive_center_start", type=int,
                        default=cfg.progressive_center_start)
    parser.add_argument("--progressive_relation_start", type=int,
                        default=cfg.progressive_relation_start)
    parser.add_argument("--progressive_calibration_batches", type=int,
                        default=cfg.progressive_calibration_batches)
    parser.add_argument("--progressive_center_grad_ratio", type=float,
                        default=cfg.progressive_center_grad_ratio)
    parser.add_argument("--progressive_feature_grad_ratio", type=float,
                        default=cfg.progressive_feature_grad_ratio)
    parser.add_argument("--progressive_relation_grad_ratio", type=float,
                        default=cfg.progressive_relation_grad_ratio)
    parser.add_argument("--teacher_center_cache", default=cfg.teacher_center_cache)
    parser.add_argument("--initial_student_weights", default=cfg.initial_student_weights,
                        help="Optional common random initial state for controlled scratch runs")
    parser.add_argument("--resume_training_state", default=cfg.resume_training_state)
    parser.add_argument("--continuation_source_epoch", type=int,
                        default=cfg.continuation_source_epoch)
    parser.add_argument("--continuation_type", choices=["none", "weights_only"],
                        default=cfg.continuation_type)
    parser.add_argument("--embedding_weight", type=float, default=cfg.embedding_weight,
                        help="Projected embedding loss weight")
    parser.add_argument("--logit_kd_weight", type=float, default=cfg.logit_kd_weight,
                        help="Optional Hinton logit KD weight inside hybrid KD")
    parser.add_argument("--topk_k", type=int, default=cfg.topk_k,
                        help="Teacher top-k classes for hard_topk KD")
    parser.add_argument("--topk_weight", type=float, default=cfg.topk_weight,
                        help="Top-k KD loss weight for hard_topk")
    parser.add_argument("--margin_weight", type=float, default=cfg.margin_weight,
                        help="Margin-ranking loss weight for hard_topk")
    parser.add_argument("--margin_m", type=float, default=cfg.margin_m,
                        help="Required true-vs-best-wrong logit margin for hard_topk")
    parser.add_argument("--hard_weight", type=float, default=cfg.hard_weight,
                        help="Sample weight multiplier for online hard samples")
    parser.add_argument("--hard_margin_threshold", type=float, default=cfg.hard_margin_threshold,
                        help="Hard sample threshold for true-vs-best-wrong logit margin")
    parser.add_argument("--teacher_conf_threshold", type=float, default=cfg.teacher_conf_threshold,
                        help="Minimum teacher top-1 confidence for hard weighting")
    parser.add_argument("--anchor_weights", default=cfg.anchor_weights,
                        help="Frozen anchor student weights for conservative KD. Default: --student_weights")
    parser.add_argument("--anchor_weight", type=float, default=cfg.anchor_weight,
                        help="KL weight that keeps conservative KD close to the anchor student")
    parser.add_argument("--anchor_temperature", type=float, default=cfg.anchor_temperature,
                        help="Temperature for anchor KL in conservative KD")
    parser.add_argument("--teacher1_weight", type=float, default=cfg.teacher1_weight,
                        help="Top-k KD weight for teacher 1 in conservative_multiteacher")
    parser.add_argument("--teacher2_weight", type=float, default=cfg.teacher2_weight,
                        help="Selective top-k KD weight for teacher 2 in conservative_multiteacher")
    parser.add_argument("--teacher2_conf_threshold", type=float, default=cfg.teacher2_conf_threshold,
                        help="Minimum teacher2 confidence to activate teacher2 KD")
    parser.add_argument("--teacher_agree_bonus", type=float, default=cfg.teacher_agree_bonus,
                        help="Multiplier when both teachers predict the hard label correctly")
    parser.add_argument("--teacher_disagree_policy", default=cfg.teacher_disagree_policy,
                        choices=["conservative", "teacher2_only", "weighted"],
                        help="How to weight teacher2 when teacher1 and teacher2 disagree")
    parser.add_argument("--topkd_mode", default=cfg.topkd_mode,
                        choices=["lite", "full"],
                        help="Top-KD mode: lite=TSM+TDL, full=TSM+TDL+contrastive")
    parser.add_argument("--topkd_k", type=int, default=cfg.topkd_k,
                        help="Top-K logits used by Top-KD")
    parser.add_argument("--topkd_ce_weight", type=float, default=cfg.topkd_ce_weight,
                        help="CE weight for Top-KD")
    parser.add_argument("--topkd_tdl_weight", type=float, default=cfg.topkd_tdl_weight,
                        help="Top-K decoupled loss weight")
    parser.add_argument("--topkd_contrast_weight", type=float, default=cfg.topkd_contrast_weight,
                        help="Top-KD contrastive loss weight")
    parser.add_argument("--topkd_scale", type=float, default=cfg.topkd_scale,
                        help="Rank-dependent Top-K teacher logit scale")
    parser.add_argument("--topkd_temperature", type=float, default=cfg.topkd_temperature,
                        help="Top-KD temperature. 0 means follow --temperature")
    parser.add_argument("--no_topkd_include_gt", action="store_true",
                        help="Do not force hard-label class into teacher Top-K set")
    parser.add_argument("--epochs",      type=int,   default=cfg.epochs)
    parser.add_argument("--lr",          type=float, default=cfg.lr)
    parser.add_argument("--lr_min",      type=float, default=cfg.lr_min)
    parser.add_argument("--weight_decay",type=float, default=cfg.weight_decay)
    parser.add_argument("--batch_size",  type=int,   default=cfg.batch_size)
    parser.add_argument("--num_workers", type=int,   default=cfg.num_workers,
                        help="DataLoader workers. Use 0 on Windows/Python 3.14 if workers crash.")
    parser.add_argument("--cutout_length", type=int, default=cfg.cutout_length,
                        help="CutOut patch size for train augmentation (0=disable)")
    parser.add_argument("--augmentation_policy", default=cfg.augmentation_policy,
                        choices=[
                            "v1_legacy",
                            "v2_multi_distance",
                            "v3_no_flip_light",
                            "v4_robust_light",
                        ],
                        help=(
                            "Train augmentation policy: v1_legacy, v2_multi_distance, "
                            "v3_no_flip_light, or v4_robust_light"
                        ))
    parser.add_argument("--train_sampler", default=cfg.train_sampler,
                        choices=["random", "pk"],
                        help="Train sampler: random shuffle (default) or PK class-balanced batches")
    parser.add_argument("--pk_p", type=int, default=cfg.pk_p,
                        help="PK sampler identities per batch")
    parser.add_argument("--pk_k", type=int, default=cfg.pk_k,
                        help="PK sampler samples per identity")
    parser.add_argument("--warmup_epochs", type=int, default=cfg.warmup_epochs)
    parser.add_argument("--scheduler",     default=cfg.scheduler,
                        choices=["cosine", "sgdr"],
                        help="LR scheduler: 'cosine' (default) atau 'sgdr' (warm restarts)")
    parser.add_argument("--sgdr_T0",       type=int, default=cfg.sgdr_T0,
                        help="SGDR: epoch per cycle pertama (default: 50)")
    parser.add_argument("--sgdr_T_mult",   type=int, default=cfg.sgdr_T_mult,
                        help="SGDR: multiplier panjang cycle (default: 2)")
    parser.add_argument("--drop_path",   type=float, default=cfg.drop_path_prob)
    parser.add_argument("--label_smoothing", type=float, default=cfg.label_smoothing,
                        help="Label smoothing pada CE component (default 0.1). "
                             "Set 0.0 untuk menonaktifkan (recommended saat KD aktif).")
    parser.add_argument("--seed",        type=int,   default=cfg.seed)
    parser.add_argument("--no_amp",      action="store_true",
                        help="Disable Automatic Mixed Precision")
    parser.add_argument("--no_pretrained_student", action="store_true",
                        help="Train student from scratch (random init, ignore --student_weights)")
    parser.add_argument("--freeze_bn", action="store_true",
                        help="Keep student BatchNorm layers in eval mode during KD fine-tuning")

    # MixUp / CutMix
    parser.add_argument("--mixup_alpha",  type=float, default=cfg.mixup_alpha,
                        help="MixUp Beta distribution alpha (0=disable, default: 0.8)")
    parser.add_argument("--cutmix_alpha", type=float, default=cfg.cutmix_alpha,
                        help="CutMix Beta distribution alpha (0=disable, default: 1.0)")
    parser.add_argument("--mix_prob",     type=float, default=cfg.mix_prob,
                        help="Probability of applying MixUp/CutMix per batch (default: 1.0)")
    parser.add_argument("--mix_switch_prob", type=float, default=cfg.mix_switch_prob,
                        help="Prob of choosing CutMix over MixUp (default: 0.5)")
    parser.add_argument("--no_mix",       action="store_true",
                        help="Disable MixUp and CutMix entirely")
    parser.add_argument("--save_epoch_checkpoints", action="store_true",
                        help="Save periodic epoch checkpoints for SWA/checkpoint averaging")
    parser.add_argument("--checkpoint_start_epoch", type=int, default=80,
                        help="First epoch to save when --save_epoch_checkpoints is enabled")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Save every N epochs after --checkpoint_start_epoch")

    args = parser.parse_args()

    # Update cfg dengan nilai dari argparse
    cfg.teacher_arch        = args.teacher_arch
    cfg.teacher_weights     = args.teacher_weights
    cfg.teacher2_arch       = args.teacher2_arch
    cfg.teacher2_weights    = args.teacher2_weights
    cfg.student_weights     = args.student_weights
    cfg.student_config_path = args.student_config
    cfg.data_dir            = args.data_dir
    cfg.split_path          = args.split_path
    cfg.temperature         = args.temperature
    cfg.alpha               = args.alpha
    cfg.kd_method           = args.kd_method
    cfg.dkd_alpha           = args.dkd_alpha
    cfg.dkd_beta            = args.dkd_beta
    cfg.dkd_warmup_epochs   = args.dkd_warmup_epochs
    cfg.adaface             = args.adaface
    cfg.adaface_m           = args.adaface_m
    cfg.adaface_h           = args.adaface_h
    cfg.adaface_s           = args.adaface_s
    cfg.adaface_t_alpha     = args.adaface_t_alpha
    cfg.skip_test_evaluation = args.skip_test_evaluation
    cfg.ce_weight           = args.ce_weight
    cfg.relation_weight     = args.relation_weight
    cfg.center_weight       = args.center_weight
    cfg.feature_weight      = args.feature_weight
    cfg.center_scale        = args.center_scale
    cfg.center_margin       = args.center_margin
    cfg.relation_topk       = args.relation_topk
    cfg.relation_difference_threshold = args.relation_difference_threshold
    cfg.adaptive_warmup_epochs = args.adaptive_warmup_epochs
    cfg.progressive_staging = args.progressive_staging
    cfg.progressive_center_start = args.progressive_center_start
    cfg.progressive_relation_start = args.progressive_relation_start
    cfg.progressive_calibration_batches = args.progressive_calibration_batches
    cfg.progressive_center_grad_ratio = args.progressive_center_grad_ratio
    cfg.progressive_feature_grad_ratio = args.progressive_feature_grad_ratio
    cfg.progressive_relation_grad_ratio = args.progressive_relation_grad_ratio
    if cfg.progressive_staging:
        if cfg.kd_method != "adaptive_center_relation":
            parser.error("--progressive_staging requires --kd_method adaptive_center_relation")
        if not (1 <= cfg.progressive_center_start < cfg.progressive_relation_start <= args.epochs):
            parser.error("progressive stages must satisfy 1 <= center_start < relation_start <= epochs")
        if cfg.progressive_calibration_batches <= 0:
            parser.error("--progressive_calibration_batches must be positive")
    cfg.teacher_center_cache = args.teacher_center_cache
    cfg.initial_student_weights = args.initial_student_weights
    cfg.resume_training_state = args.resume_training_state
    cfg.continuation_source_epoch = args.continuation_source_epoch
    cfg.continuation_type = args.continuation_type
    if cfg.continuation_type == "weights_only" and cfg.continuation_source_epoch <= 0:
        parser.error("weights_only continuation requires --continuation_source_epoch > 0")
    cfg.embedding_weight    = args.embedding_weight
    cfg.logit_kd_weight     = args.logit_kd_weight
    cfg.topk_k              = args.topk_k
    cfg.topk_weight         = args.topk_weight
    cfg.margin_weight       = args.margin_weight
    cfg.margin_m            = args.margin_m
    cfg.hard_weight         = args.hard_weight
    cfg.hard_margin_threshold = args.hard_margin_threshold
    cfg.teacher_conf_threshold = args.teacher_conf_threshold
    cfg.anchor_weights      = args.anchor_weights or args.student_weights
    cfg.anchor_weight       = args.anchor_weight
    cfg.anchor_temperature  = args.anchor_temperature
    cfg.teacher1_weight     = args.teacher1_weight
    cfg.teacher2_weight     = args.teacher2_weight
    cfg.teacher2_conf_threshold = args.teacher2_conf_threshold
    cfg.teacher_agree_bonus = args.teacher_agree_bonus
    cfg.teacher_disagree_policy = args.teacher_disagree_policy
    cfg.topkd_mode          = args.topkd_mode
    cfg.topkd_k             = args.topkd_k
    cfg.topkd_ce_weight     = args.topkd_ce_weight
    cfg.topkd_tdl_weight    = args.topkd_tdl_weight
    cfg.topkd_contrast_weight = args.topkd_contrast_weight
    cfg.topkd_scale         = args.topkd_scale
    cfg.topkd_temperature   = args.topkd_temperature
    cfg.topkd_include_gt    = not args.no_topkd_include_gt
    cfg.epochs              = args.epochs
    cfg.lr                  = args.lr
    cfg.lr_min              = args.lr_min
    if cfg.lr_min > cfg.lr:
        corrected_lr_min = cfg.lr * 0.1
        print(
            f"WARNING: lr_min ({cfg.lr_min:g}) > lr ({cfg.lr:g}); "
            f"setting lr_min to {corrected_lr_min:g}. "
            "Pass --lr_min explicitly to override this safeguard."
        )
        cfg.lr_min = corrected_lr_min
    cfg.weight_decay        = args.weight_decay
    cfg.batch_size          = args.batch_size
    cfg.cutout_length       = args.cutout_length
    cfg.augmentation_policy = args.augmentation_policy
    cfg.train_sampler       = args.train_sampler
    cfg.pk_p                = args.pk_p
    cfg.pk_k                = args.pk_k
    if cfg.train_sampler == "pk" and cfg.pk_p * cfg.pk_k != cfg.batch_size:
        parser.error(
            f"--train_sampler pk requires --pk_p * --pk_k == --batch_size; "
            f"got {cfg.pk_p} * {cfg.pk_k} = {cfg.pk_p * cfg.pk_k}, "
            f"batch_size={cfg.batch_size}"
        )
    cfg.warmup_epochs       = args.warmup_epochs
    cfg.scheduler           = args.scheduler
    cfg.sgdr_T0             = args.sgdr_T0
    cfg.sgdr_T_mult         = args.sgdr_T_mult
    cfg.drop_path_prob      = args.drop_path
    cfg.label_smoothing     = args.label_smoothing
    cfg.seed                = args.seed
    if args.no_amp:
        cfg.amp = False
    cfg.no_pretrained_student = args.no_pretrained_student
    cfg.freeze_bn = args.freeze_bn
    cfg.save_epoch_checkpoints = args.save_epoch_checkpoints
    cfg.checkpoint_start_epoch = args.checkpoint_start_epoch
    cfg.checkpoint_interval = args.checkpoint_interval
    if cfg.save_epoch_checkpoints and cfg.checkpoint_interval <= 0:
        parser.error("--checkpoint_interval must be > 0 when --save_epoch_checkpoints is enabled")

    # MixUp / CutMix
    if args.no_mix:
        cfg.mixup_alpha  = 0.0
        cfg.cutmix_alpha = 0.0
    else:
        cfg.mixup_alpha     = args.mixup_alpha
        cfg.cutmix_alpha    = args.cutmix_alpha
        cfg.mix_prob        = args.mix_prob
        cfg.mix_switch_prob = args.mix_switch_prob

    if cfg.kd_method in {"hard_topk", "conservative", "conservative_multiteacher", "topkd", "adaptive_center_relation"} and (cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0):
        print(
            f"WARNING: {cfg.kd_method} KD requires unmixed identity targets; "
            "forcing MixUp/CutMix off. Pass --no_mix to silence this warning."
        )
        cfg.mixup_alpha = 0.0
        cfg.cutmix_alpha = 0.0

    # Auto-generate output folder dari parameter jika tidak di-set eksplisit
    # Format: kd_results/t{temp}_a{alpha}_e{epochs}
    # Contoh: kd_results/t4.0_a0.3_e150
    if args.output_dir is None:
        _base = Path(KD_CFG.output_dir).parent / "kd_results"
        t_str = str(cfg.temperature).rstrip("0").rstrip(".")  # 4.0 → "4", 4.5 → "4.5"
        a_str = str(cfg.alpha).rstrip("0").rstrip(".")
        folder_name = f"t{t_str}_a{a_str}_e{cfg.epochs}"
        cfg.output_dir = str(_base / folder_name)
    else:
        cfg.output_dir = args.output_dir

    return cfg


# ─── Load Teacher ─────────────────────────────────────────────────────────────

_SUPPORTED_TEACHER_ARCHS = [
    "efficientnet_v2_m", "efficientnet_b4", "densenet121",
    "inception_v3", "resnet50", "convnext_base",
    "regnet_y_16gf", "mobilenet_v3_large", "mobilenet_v3_small", "vgg16",
]


_TEACHER_ARCH_ALIASES = {
    "EfficientNetV2M": "efficientnet_v2_m",
    "EfficientNetB4": "efficientnet_b4",
    "DenseNet121": "densenet121",
    "InceptionV3": "inception_v3",
    "ResNet50": "resnet50",
    "ConvNeXtBase": "convnext_base",
    "RegNetY16GF": "regnet_y_16gf",
    "MobileNetV3Large": "mobilenet_v3_large",
    "MobileNetV3Small": "mobilenet_v3_small",
    "VGG16": "vgg16",
}


def _candidate_head_keys(expected_key: str) -> list[str]:
    if not (expected_key.endswith(".weight") or expected_key.endswith(".bias")):
        return []

    suffix = ".weight" if expected_key.endswith(".weight") else ".bias"
    stem = expected_key[: -len(suffix)]
    parts = stem.split(".")

    candidates: list[str] = []

    # fc.weight -> fc.1.weight, AuxLogits.fc.weight -> AuxLogits.fc.1.weight
    for idx in range(8):
        candidates.append(f"{stem}.{idx}{suffix}")

    # classifier.weight -> classifier.1.weight
    if stem == "classifier":
        for idx in range(8):
            candidates.append(f"classifier.{idx}{suffix}")

    # classifier.2.weight -> classifier.3.weight, classifier.6.weight, etc.
    if len(parts) >= 2 and parts[-1].isdigit():
        parent = ".".join(parts[:-1])
        for idx in range(8):
            candidates.append(f"{parent}.{idx}{suffix}")

    return candidates


def _remap_sequential_classifier_keys(state_dict: dict, model: nn.Module | None = None) -> dict:
    """
    Beberapa teacher lama disimpan dengan head berbentuk Sequential(Dropout, Linear),
    sehingga key Linear menjadi *.1.weight/*.1.bias. Loader KD memakai head Linear
    langsung atau memakai indeks classifier berbeda, jadi key perlu dipetakan.

    Jika model diberikan, remap dilakukan dengan mencocokkan expected key dan shape.
    Ini membuat loader robust untuk EfficientNet, ResNet, RegNet, DenseNet, ConvNeXt,
    Inception, MobileNet, dan VGG selama backbone checkpoint-nya sama.
    """
    remapped = dict(state_dict)

    if model is None:
        replacements = {
            "fc.1.weight": "fc.weight",
            "fc.1.bias": "fc.bias",
            "AuxLogits.fc.1.weight": "AuxLogits.fc.weight",
            "AuxLogits.fc.1.bias": "AuxLogits.fc.bias",
            "classifier.1.weight": "classifier.weight",
            "classifier.1.bias": "classifier.bias",
            "classifier.3.weight": "classifier.2.weight",
            "classifier.3.bias": "classifier.2.bias",
        }
        return {replacements.get(k, k): v for k, v in remapped.items()}

    expected = model.state_dict()
    used_source_keys: set[str] = set()

    for expected_key, expected_value in expected.items():
        if expected_key in remapped:
            continue

        for source_key in _candidate_head_keys(expected_key):
            if source_key in state_dict and source_key not in used_source_keys:
                if tuple(state_dict[source_key].shape) == tuple(expected_value.shape):
                    remapped[expected_key] = state_dict[source_key]
                    used_source_keys.add(source_key)
                    break

    for source_key in used_source_keys:
        remapped.pop(source_key, None)

    return remapped


def load_teacher(cfg: KDConfig, device: torch.device, logger: logging.Logger) -> nn.Module:
    """
    Load teacher model. Mendukung 9 arsitektur berbeda.
    Model di-freeze total (eval mode selamanya).

    Catatan InceptionV3:
      - Harus dibangun dengan aux_logits=True agar state_dict cocok
        (teacher di-train dengan aux head aktif).
      - Saat eval mode, PyTorch InceptionV3 otomatis hanya return main logits
        (bukan InceptionOutputs namedtuple) — training loop tidak perlu diubah.
    """
    arch = _TEACHER_ARCH_ALIASES.get(cfg.teacher_arch, cfg.teacher_arch)
    logger.info(f"  Loading teacher: {arch}  weights={cfg.teacher_weights}")

    if arch == "efficientnet_v2_m":
        teacher = tv_models.efficientnet_v2_m(weights=None)
        in_features = teacher.classifier[1].in_features
        teacher.classifier[1] = nn.Linear(in_features, cfg.num_classes)

    elif arch == "efficientnet_b4":
        teacher = tv_models.efficientnet_b4(weights=None)
        in_features = teacher.classifier[1].in_features
        teacher.classifier[1] = nn.Linear(in_features, cfg.num_classes)

    elif arch == "densenet121":
        teacher = tv_models.densenet121(weights=None)
        in_features = teacher.classifier.in_features
        teacher.classifier = nn.Linear(in_features, cfg.num_classes)

    elif arch == "inception_v3":
        # aux_logits=True supaya state_dict match dengan checkpoint yang di-train pakai aux head
        teacher = tv_models.inception_v3(weights=None, aux_logits=True)
        in_features = teacher.fc.in_features
        teacher.fc = nn.Linear(in_features, cfg.num_classes)
        # Aux classifier head juga harus diganti ke num_classes yang benar
        in_features_aux = teacher.AuxLogits.fc.in_features
        teacher.AuxLogits.fc = nn.Linear(in_features_aux, cfg.num_classes)
        _raw_sd = torch.load(cfg.teacher_weights, map_location="cpu")
        _remapped = _remap_sequential_classifier_keys(_raw_sd, teacher)
        teacher.load_state_dict(_remapped, strict=True)
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        n_params = sum(p.numel() for p in teacher.parameters()) / 1e6
        logger.info(f"  Teacher loaded: {n_params:.1f}M params  |  FROZEN")
        return teacher

    elif arch == "resnet50":
        teacher = tv_models.resnet50(weights=None)
        in_features = teacher.fc.in_features
        teacher.fc = nn.Linear(in_features, cfg.num_classes)

    elif arch == "convnext_base":
        teacher = tv_models.convnext_base(weights=None)
        in_features = teacher.classifier[2].in_features
        teacher.classifier[2] = nn.Linear(in_features, cfg.num_classes)

    elif arch == "regnet_y_16gf":
        teacher = tv_models.regnet_y_16gf(weights=None)
        in_features = teacher.fc.in_features
        teacher.fc = nn.Linear(in_features, cfg.num_classes)

    elif arch == "mobilenet_v3_large":
        teacher = tv_models.mobilenet_v3_large(weights=None)
        in_features = teacher.classifier[3].in_features
        teacher.classifier[3] = nn.Linear(in_features, cfg.num_classes)

    elif arch == "mobilenet_v3_small":
        teacher = tv_models.mobilenet_v3_small(weights=None)
        in_features = teacher.classifier[3].in_features
        teacher.classifier[3] = nn.Linear(in_features, cfg.num_classes)

    elif arch == "vgg16":
        teacher = tv_models.vgg16(weights=None)
        in_features = teacher.classifier[6].in_features
        teacher.classifier[6] = nn.Linear(in_features, cfg.num_classes)

    else:
        raise ValueError(
            f"Teacher arch tidak dikenal: '{arch}'. "
            f"Pilihan yang tersedia: {_SUPPORTED_TEACHER_ARCHS}"
        )

    state_dict = torch.load(cfg.teacher_weights, map_location="cpu")
    state_dict = _remap_sequential_classifier_keys(state_dict, teacher)
    teacher.load_state_dict(state_dict, strict=True)

    teacher.to(device)
    teacher.eval()

    # Freeze semua parameter teacher
    for p in teacher.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    logger.info(f"  Teacher loaded: {n_params:.1f}M params  |  FROZEN")
    return teacher


def load_teacher2(cfg: KDConfig, device: torch.device, logger: logging.Logger) -> nn.Module:
    """Load the second frozen teacher for conservative multi-teacher KD."""
    teacher2_cfg = deepcopy(cfg)
    teacher2_cfg.teacher_arch = cfg.teacher2_arch
    teacher2_cfg.teacher_weights = cfg.teacher2_weights
    logger.info("  Loading teacher 2 for conservative multi-teacher KD...")
    return load_teacher(teacher2_cfg, device, logger)


def _find_final_linear(module: nn.Module) -> nn.Linear:
    final_linear = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            final_linear = child
    if final_linear is None:
        raise ValueError("Tidak menemukan nn.Linear final pada teacher untuk ekstraksi embedding.")
    return final_linear


def get_teacher_embedding_dim(teacher: nn.Module) -> int:
    return int(_find_final_linear(teacher).in_features)


def get_student_embedding_dim(student: nn.Module) -> int:
    return int(student.classifier.in_features)


def teacher_forward_with_embeddings(
    teacher: nn.Module,
    arch: str,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return logits and penultimate embeddings for torchvision teacher models.
    Captures the input to the final Linear layer, so it works across supported
    EfficientNet/ResNet/ConvNeXt/DenseNet-style classifiers.
    """
    del arch  # Arch is kept in the interface for logging/debug extension points.
    final_linear = _find_final_linear(teacher)
    captured: dict[str, torch.Tensor] = {}

    def _capture(_module, inputs, _output):
        embedding = inputs[0]
        captured["embedding"] = embedding.flatten(1)

    handle = final_linear.register_forward_hook(_capture)
    try:
        logits = teacher(images)
    finally:
        handle.remove()

    if isinstance(logits, tuple):
        logits = logits[0]
    if "embedding" not in captured:
        raise RuntimeError("Gagal mengambil teacher embedding dari final Linear hook.")
    return logits, captured["embedding"]


@torch.no_grad()
def prepare_teacher_center_bank(teacher, train_samples, cfg, device, logger,
                                label_map) -> tuple[torch.Tensor, dict]:
    """Build/reuse training-only teacher class centers with strict provenance."""
    teacher_hash = sha256_file(cfg.teacher_weights)
    split_hash = sha256_file(cfg.split_path)
    teacher_dim = get_teacher_embedding_dim(teacher)
    metadata = {
        "teacher_sha256": teacher_hash,
        "split_sha256": split_hash,
        "label_map_sha256": stable_json_hash(label_map),
        "preprocessing_sha256": stable_json_hash({
            "input_size": cfg.input_size,
            "transform": "deterministic_validation_imagenet_normalization",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "grayscale_to_rgb": "repeat_channels",
        }),
        "num_classes": cfg.num_classes,
        "embedding_dim": teacher_dim,
        "source_partition": "train",
        "sample_count": len(train_samples),
    }
    cache_path = Path(cfg.teacher_center_cache) if cfg.teacher_center_cache else (
        _HERE / "cache" / f"teacher_centers_{teacher_hash[:12]}_{split_hash[:12]}.pth"
    )
    cfg.teacher_center_cache = str(cache_path)
    if cache_path.exists():
        logger.info(f"  Loading verified teacher center cache: {cache_path}")
        return load_center_cache(cache_path, metadata).to(device), metadata

    dataset = PalmVeinDataset(train_samples, get_transforms("val", cfg.input_size))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    sums = torch.zeros(cfg.num_classes, teacher_dim, dtype=torch.float64, device=device)
    counts = torch.zeros(cfg.num_classes, dtype=torch.long, device=device)
    teacher.eval()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, embeddings = teacher_forward_with_embeddings(teacher, cfg.teacher_arch, images)
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
        sums.index_add_(0, labels, embeddings.double())
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))
    missing = torch.where(counts == 0)[0].tolist()
    if missing:
        raise ValueError(f"Cannot initialize center bank; training classes missing: {missing[:20]}")
    centers = torch.nn.functional.normalize(
        (sums / counts[:, None]).float(), dim=1
    )
    save_center_cache(cache_path, centers, metadata)
    logger.info(f"  Created training-only teacher center cache: {cache_path}")
    return centers, metadata


# ─── Load Student ─────────────────────────────────────────────────────────────

def load_student(cfg: KDConfig, device: torch.device, logger: logging.Logger) -> nn.Module:
    """
    Load NAS student dari genotype (dalam config.json) lalu load pretrained weights.
    auxiliary=False untuk KD — hanya satu output logit.
    """
    logger.info(f"  Loading student config: {cfg.student_config_path}")

    with open(cfg.student_config_path, "r") as f:
        retrain_cfg = json.load(f)

    genotype = dict_to_genotype(retrain_cfg["genotype"])

    # Baca C_init dan num_cells langsung dari config.json (lebih akurat dari default cfg)
    c_init    = int(retrain_cfg.get("C_init",    cfg.student_C_init))
    num_cells = int(retrain_cfg.get("num_cells", cfg.student_num_cells))
    stem_downsample = int(retrain_cfg.get("stem_downsample", 2))
    
    # Parse reduction_indices: handle both list [4, 9] and string "4, 9"
    reduction_indices_raw = retrain_cfg.get("reduction_indices", None)
    if reduction_indices_raw is None:
        reduction_indices = None
    elif isinstance(reduction_indices_raw, list):
        reduction_indices = [int(x) for x in reduction_indices_raw]
    elif isinstance(reduction_indices_raw, str):
        # Parse comma-separated string "4, 9" → [4, 9]
        reduction_indices = [int(x.strip()) for x in reduction_indices_raw.split(",") if x.strip()]
    else:
        logger.warning(f"  Unexpected reduction_indices type: {type(reduction_indices_raw)}, using None")
        reduction_indices = None
    
    logger.info(
        f"  Student arch: C_init={c_init}, num_cells={num_cells}, "
        f"stem_downsample={stem_downsample}, reduction_indices={reduction_indices}"
    )

    student = EvalNetwork(
        genotype          = genotype,
        C_init            = c_init,
        num_cells         = num_cells,
        num_classes       = cfg.num_classes,
        auxiliary         = False,   # KD: hanya pakai main head
        dropout           = cfg.student_dropout,
        stem_downsample   = stem_downsample,
        reduction_indices = reduction_indices,
    )
    use_adaface = bool(cfg.adaface or retrain_cfg.get("loss_mode") == "adaface")
    if use_adaface:
        replace_linear_with_adaface(
            student, num_classes=cfg.num_classes,
            m=float(retrain_cfg.get("adaface_m", cfg.adaface_m)),
            h=float(retrain_cfg.get("adaface_h", cfg.adaface_h)),
            s=float(retrain_cfg.get("adaface_s", cfg.adaface_s)),
            t_alpha=float(retrain_cfg.get("adaface_t_alpha", cfg.adaface_t_alpha)),
        )
        cfg.adaface = True
        logger.info("  Student head: AdaFace (margin training; cosine inference logits)")

    if cfg.no_pretrained_student:
        if cfg.initial_student_weights:
            state_dict = torch.load(cfg.initial_student_weights, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "student" in state_dict:
                state_dict = state_dict["student"]
            student.load_state_dict(state_dict, strict=True)
            logger.info(f"  Student: common initial state {cfg.initial_student_weights}")
        else:
            logger.info("  Student: random initialization (from scratch, --no_pretrained_student)")
    else:
        logger.info(f"  Loading student weights: {cfg.student_weights}")
        state_dict = torch.load(cfg.student_weights, map_location="cpu")

        # strict=False karena checkpoint mungkin punya kunci _auxiliary_head.*
        # yang tidak ada di student dengan auxiliary=False
        missing, unexpected = student.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            aux_keys = [k for k in unexpected if "_auxiliary_head" in k]
            other_unexpect = [k for k in unexpected if "_auxiliary_head" not in k]
            if aux_keys:
                logger.info(f"  Auxiliary head keys skipped (expected, auxiliary=False): {len(aux_keys)} keys")
            if other_unexpect:
                logger.warning(f"  Other unexpected keys: {other_unexpect}")

    student.to(device)

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e3
    logger.info(f"  Student loaded: {n_params:.1f}K trainable params")

    return student


def load_anchor_student(cfg: KDConfig, device: torch.device, logger: logging.Logger) -> nn.Module:
    """
    Load frozen copy of the original student for conservative KD.

    The anchor keeps the fine-tuned student close to the known-good retrain
    checkpoint so KD cannot freely move decision boundaries that were already
    correct.
    """
    anchor_cfg = deepcopy(cfg)
    anchor_cfg.no_pretrained_student = False
    anchor_cfg.student_weights = cfg.anchor_weights or cfg.student_weights

    logger.info("  Loading conservative anchor student...")
    logger.info(f"  Anchor weights: {anchor_cfg.student_weights}")
    anchor = load_student(anchor_cfg, device, logger)
    anchor.eval()
    for param in anchor.parameters():
        param.requires_grad_(False)

    n_params = sum(p.numel() for p in anchor.parameters()) / 1e3
    logger.info(f"  Anchor student frozen: {n_params:.1f}K params")
    return anchor


# ─── LR Scheduler: Linear Warmup + Cosine Annealing (or SGDR) ────────────────

def build_scheduler(optimizer, cfg: KDConfig, steps_per_epoch: int):
    """
    Warmup linear selama warmup_epochs, kemudian:
      - scheduler="cosine"  → standard cosine annealing ke lr_min
      - scheduler="sgdr"    → cosine annealing with warm restarts (SGDR)
                               LR di-reset ke lr awal setiap akhir cycle.
                               Panjang cycle: T0, T0*T_mult, T0*T_mult^2, ...
    """
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps  = cfg.epochs * steps_per_epoch
    lr_ratio     = cfg.lr_min / cfg.lr   # rasio lr_min / lr_max

    if cfg.scheduler == "sgdr":
        T0_steps = cfg.sgdr_T0 * steps_per_epoch
        T_mult   = cfg.sgdr_T_mult

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            # Tentukan posisi dalam cycle SGDR saat ini
            s = step - warmup_steps
            if T_mult == 1:
                # Semua cycle panjangnya sama
                cycle_pos = s % T0_steps
                cycle_len = T0_steps
            else:
                # Geometric cycles: T0, T0*Tm, T0*Tm^2, ...
                cumul = 0
                cycle_len = T0_steps
                while cumul + cycle_len <= s:
                    cumul += cycle_len
                    cycle_len = int(cycle_len * T_mult)
                cycle_pos = s - cumul
            progress = cycle_pos / max(cycle_len, 1)
            cos_val  = 0.5 * (1.0 + np.cos(np.pi * progress))
            return lr_ratio + (1.0 - lr_ratio) * cos_val
    else:
        # Standard cosine annealing (default)
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cos_val  = 0.5 * (1.0 + np.cos(np.pi * progress))
            return lr_ratio + (1.0 - lr_ratio) * cos_val

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── MixUp / CutMix augmentation (batch-level) ──────────────────────────────

def rand_bbox(size, lam):
    """Random bounding box untuk CutMix. Returns (y1, y2, x1, x2)."""
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    return y1, y2, x1, x2


def apply_mixup_cutmix(images, targets, cfg):
    """
    Terapkan MixUp atau CutMix pada batch saat training.

    Logika:
    1. Jika kedua alpha == 0 → skip (return original)
    2. Random uniform < mix_prob → apply, else skip
    3. Random uniform < mix_switch_prob → CutMix, else MixUp

    Returns:
        mixed_images : [B, C, H, W] — gambar hasil mix
        targets_a    : [B] — labels asli
        targets_b    : [B] — labels dari shuffled index
        lam          : float — mixing coefficient (adjusted for CutMix)
        is_mixed     : bool — apakah mixing benar-benar diterapkan
    """
    mixup_alpha  = cfg.mixup_alpha
    cutmix_alpha = cfg.cutmix_alpha

    # Cek apakah mix diaktifkan
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, targets, targets, 1.0, False

    # Probability gate
    if np.random.rand() > cfg.mix_prob:
        return images, targets, targets, 1.0, False

    # Pilih MixUp atau CutMix
    use_cutmix = False
    if cutmix_alpha > 0 and mixup_alpha > 0:
        use_cutmix = np.random.rand() < cfg.mix_switch_prob
    elif cutmix_alpha > 0:
        use_cutmix = True

    # Sample lambda dari Beta distribution
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)

    # Shuffle index
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    targets_a = targets
    targets_b = targets[index]

    if use_cutmix:
        # CutMix: potong region dan tempelkan dari gambar lain
        y1, y2, x1, x2 = rand_bbox(images.size(), lam)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        # Adjust lambda ke rasio area aktual
        lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (images.size(-2) * images.size(-1))
    else:
        # MixUp: interpolasi linear
        mixed_images = lam * images + (1.0 - lam) * images[index]

    return mixed_images, targets_a, targets_b, lam, True


# ─── BatchNorm freezing ──────────────────────────────────────────────────────

def freeze_batchnorm(model: nn.Module, freeze_affine: bool = True) -> int:
    """
    Paksa semua BatchNorm memakai running mean/var dari checkpoint pretrained.
    Conv/Linear tetap trainable; hanya statistik BN yang tidak ikut berubah.
    """
    n_bn = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            n_bn += 1
            if freeze_affine:
                if module.weight is not None:
                    module.weight.requires_grad_(False)
                if module.bias is not None:
                    module.bias.requires_grad_(False)
    return n_bn


# ─── Train one epoch ─────────────────────────────────────────────────────────

def train_one_epoch(
    student,
    teacher,
    loader,
    optimizer,
    scheduler,
    criterion,
    scaler,
    device,
    epoch,
    cfg,
    logger,
    anchor=None,
    teacher2=None,
):
    student.train()
    if cfg.freeze_bn:
        freeze_batchnorm(student)
    if anchor is not None:
        anchor.eval()
    if teacher2 is not None:
        teacher2.eval()
    total_loss = total_ce = total_kd = total_center = total_rel = total_emb = total_logit_kd = 0.0
    total_topk = total_margin = total_hard_ratio = total_true_rank = 0.0
    total_anchor = 0.0
    total_teacher1_kd = total_teacher2_kd = total_teacher2_active = 0.0
    total_tdl = total_contrast = total_topkd = 0.0
    total_tckd = total_nckd = 0.0
    diagnostic_keys = (
        "weighted_center", "weighted_feature", "weighted_relation",
        "center_weight_effective", "feature_weight_effective", "relation_weight_effective",
        "grad_norm_ce", "grad_norm_center", "grad_norm_feature", "grad_norm_relation",
        "adaptive_stage",
    )
    progressive_totals = {key: 0.0 for key in diagnostic_keys}
    correct = n_samples = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ── MixUp / CutMix (batch-level augmentation) ───────────────
        mixed_images, targets_a, targets_b, lam, is_mixed = \
            apply_mixup_cutmix(images, targets, cfg)

        # Teacher forward — tidak perlu gradien
        # Teacher melihat gambar yang SAMA (mixed) dengan student
        with torch.no_grad():
            logits_anchor = None
            logits_teacher2 = None
            if cfg.kd_method in {"hinton", "dkd", "hard_topk", "conservative", "conservative_multiteacher", "topkd"}:
                logits_teacher = teacher(mixed_images)
                if cfg.kd_method in {"conservative", "conservative_multiteacher"}:
                    if anchor is None:
                        raise RuntimeError(f"{cfg.kd_method} KD requires an anchor student")
                    logits_anchor = anchor(mixed_images)
                if cfg.kd_method == "conservative_multiteacher":
                    if teacher2 is None:
                        raise RuntimeError("conservative_multiteacher KD requires teacher2")
                    logits_teacher2 = teacher2(mixed_images)
                teacher_embeddings = None
            elif cfg.kd_method == "adaptive_center_relation":
                if is_mixed:
                    raise RuntimeError("Adaptive center-relation distillation requires unmixed labels")
                if cfg.progressive_staging and epoch < cfg.progressive_center_start:
                    teacher_embeddings = None
                else:
                    _, teacher_embeddings = teacher_forward_with_embeddings(
                        teacher, cfg.teacher_arch, mixed_images
                    )
                logits_teacher = None
            elif cfg.kd_method == "adaptive_center_relation":
                logits_student, student_embeddings = student.forward_with_embeddings(mixed_images)
                loss, breakdown = criterion(
                    logits_student=logits_student,
                    student_embeddings=student_embeddings,
                    teacher_embeddings=teacher_embeddings,
                    targets=targets,
                    epoch=epoch,
                    batch_index=batch_idx,
                )
            else:
                logits_teacher, teacher_embeddings = teacher_forward_with_embeddings(
                    teacher, cfg.teacher_arch, mixed_images
                )

        optimizer.zero_grad(set_to_none=True)

        # AMP forward
        with autocast("cuda", enabled=cfg.amp):
            if cfg.kd_method == "hinton":
                logits_student = student(mixed_images)
                # Student dengan auxiliary=False selalu return tensor tunggal
                if is_mixed:
                    mix_targets = (targets_a, targets_b, lam)
                    loss, breakdown = criterion(logits_student, logits_teacher,
                                                targets_a, mix_targets=mix_targets)
                else:
                    loss, breakdown = criterion(logits_student, logits_teacher, targets)
            elif cfg.kd_method == "dkd":
                if cfg.adaface:
                    logits_student, classification_logits, _ = student.forward_adaface(mixed_images, targets)
                else:
                    logits_student = student(mixed_images)
                    classification_logits = None
                loss, breakdown = criterion(
                    logits_student, logits_teacher, targets,
                    epoch=epoch, classification_logits=classification_logits,
                )
            elif cfg.kd_method == "hard_topk":
                logits_student = student(mixed_images)
                loss, breakdown = criterion(
                    logits_student=logits_student,
                    logits_teacher=logits_teacher,
                    targets=targets,
                )
            elif cfg.kd_method == "conservative":
                logits_student = student(mixed_images)
                loss, breakdown = criterion(
                    logits_student=logits_student,
                    logits_teacher=logits_teacher,
                    logits_anchor=logits_anchor,
                    targets=targets,
                )
            elif cfg.kd_method == "conservative_multiteacher":
                logits_student = student(mixed_images)
                loss, breakdown = criterion(
                    logits_student=logits_student,
                    logits_teacher1=logits_teacher,
                    logits_teacher2=logits_teacher2,
                    logits_anchor=logits_anchor,
                    targets=targets,
                )
            elif cfg.kd_method == "topkd":
                logits_student = student(mixed_images)
                loss, breakdown = criterion(
                    logits_student=logits_student,
                    logits_teacher=logits_teacher,
                    targets=targets,
                )
            else:
                logits_student, student_embeddings = student.forward_with_embeddings(mixed_images)
                mix_targets = (targets_a, targets_b, lam) if is_mixed else None
                loss, breakdown = criterion(
                    logits_student=logits_student,
                    logits_teacher=logits_teacher,
                    student_embeddings=student_embeddings,
                    teacher_embeddings=teacher_embeddings,
                    targets=targets,
                    mix_targets=mix_targets,
                )

        scaler.scale(loss).backward()
        # Gradient clipping
        scaler.unscale_(optimizer)
        parameters = [
            parameter for group in optimizer.param_groups
            for parameter in group["params"] if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Statistik — untuk mixed inputs, accuracy diweightkan
        with torch.no_grad():
            pred = logits_student.argmax(dim=1)
            if is_mixed:
                correct += (lam * (pred == targets_a).float().sum().item() +
                            (1.0 - lam) * (pred == targets_b).float().sum().item())
            else:
                correct += (pred == targets).sum().item()
            n_samples += targets.size(0)

        total_loss += breakdown["loss_total"]
        total_ce   += breakdown["loss_ce"]
        total_kd   += breakdown.get("loss_kd", 0.0)
        total_center += breakdown.get("loss_center", 0.0)
        total_rel  += breakdown.get("loss_relation", 0.0)
        total_emb  += breakdown.get("loss_embedding", 0.0)
        total_logit_kd += breakdown.get("loss_logit_kd", 0.0)
        total_topk += breakdown.get("loss_topk", 0.0)
        total_teacher1_kd += breakdown.get("loss_teacher1_kd", 0.0)
        total_teacher2_kd += breakdown.get("loss_teacher2_kd", 0.0)
        total_teacher2_active += breakdown.get("teacher2_active", 0.0)
        total_margin += breakdown.get("loss_margin", 0.0)
        total_anchor += breakdown.get("loss_anchor", 0.0)
        total_hard_ratio += breakdown.get("hard_ratio", 0.0)
        total_true_rank += breakdown.get("avg_true_rank", 0.0)
        total_tdl += breakdown.get("loss_tdl", 0.0)
        total_contrast += breakdown.get("loss_contrast", 0.0)
        total_topkd += breakdown.get("loss_topkd", 0.0)
        total_tckd += breakdown.get("loss_tckd", 0.0)
        total_nckd += breakdown.get("loss_nckd", 0.0)
        for key in diagnostic_keys:
            progressive_totals[key] += float(breakdown.get(key, 0.0))

        if (batch_idx + 1) % cfg.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.debug(
                f"  E{epoch:3d} [{batch_idx+1:4d}/{len(loader)}] "
                f"loss={breakdown['loss_total']:.4f} "
                f"ce={breakdown['loss_ce']:.4f} "
                f"kd={breakdown.get('loss_kd', 0.0):.4f} "
                f"tckd={breakdown.get('loss_tckd', 0.0):.4f} "
                f"nckd={breakdown.get('loss_nckd', 0.0):.4f} "
                f"center={breakdown.get('loss_center', 0.0):.4f} "
                f"rel={breakdown.get('loss_relation', 0.0):.4f} "
                f"emb={breakdown.get('loss_embedding', 0.0):.4f} "
                f"topk={breakdown.get('loss_topk', 0.0):.4f} "
                f"tdl={breakdown.get('loss_tdl', 0.0):.4f} "
                f"contrast={breakdown.get('loss_contrast', 0.0):.4f} "
                f"topkd={breakdown.get('loss_topkd', 0.0):.4f} "
                f"t1={breakdown.get('loss_teacher1_kd', 0.0):.4f} "
                f"t2={breakdown.get('loss_teacher2_kd', 0.0):.4f} "
                f"margin={breakdown.get('loss_margin', 0.0):.4f} "
                f"anchor={breakdown.get('loss_anchor', 0.0):.4f} "
                f"hard={breakdown.get('hard_ratio', 0.0):.2f} "
                f"t2_active={breakdown.get('teacher2_active', 0.0):.2f} "
                f"lr={current_lr:.2e}"
            )

    n_batches  = len(loader)
    train_loss = total_loss / n_batches
    train_ce   = total_ce   / n_batches
    train_kd   = total_kd   / n_batches
    train_center = total_center / n_batches
    train_rel  = total_rel  / n_batches
    train_emb  = total_emb  / n_batches
    train_logit_kd = total_logit_kd / n_batches
    train_topk = total_topk / n_batches
    train_margin = total_margin / n_batches
    train_anchor = total_anchor / n_batches
    train_teacher1_kd = total_teacher1_kd / n_batches
    train_teacher2_kd = total_teacher2_kd / n_batches
    train_teacher2_active = total_teacher2_active / n_batches
    train_hard_ratio = total_hard_ratio / n_batches
    train_true_rank = total_true_rank / n_batches
    train_tdl = total_tdl / n_batches
    train_contrast = total_contrast / n_batches
    train_topkd = total_topkd / n_batches
    train_tckd = total_tckd / n_batches
    train_nckd = total_nckd / n_batches
    progressive_metrics = {
        key: value / n_batches for key, value in progressive_totals.items()
    }
    train_acc  = correct    / n_samples

    return (
        train_loss, train_ce, train_kd, train_center, train_acc,
        train_rel, train_emb, train_logit_kd,
        train_topk, train_margin, train_anchor,
        train_teacher1_kd, train_teacher2_kd, train_teacher2_active,
        train_hard_ratio, train_true_rank,
        train_tdl, train_contrast, train_topkd,
        train_tckd, train_nckd,
        progressive_metrics,
    )

# ─── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(student, loader, device, compute_auc: bool = False):
    """
    Evaluasi student di validation atau test set.
    Returns dict: acc, loss, auc (opsional)
    """
    student.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = correct = n_samples = 0
    total_true_margin = 0.0
    all_probs  = [] if compute_auc else None
    all_labels = [] if compute_auc else None

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = student(images)
        loss   = criterion(logits, targets)

        pred     = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total_loss += loss.item()
        n_samples  += targets.size(0)
        true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, targets.unsqueeze(1), float("-inf"))
        total_true_margin += (true_logits - masked.max(dim=1).values).sum().item()

        if compute_auc:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(targets.cpu().numpy().tolist())

    results = {
        "acc" : correct / n_samples,
        "loss": total_loss / len(loader),
        "true_class_margin": total_true_margin / n_samples,
        "correct": int(correct),
        "samples": int(n_samples),
        "errors": int(n_samples - correct),
    }

    if compute_auc and all_probs:
        all_probs_np  = np.concatenate(all_probs, axis=0)
        all_labels_np = np.array(all_labels)
        try:
            if all_probs_np.shape[1] > 2:
                auc = roc_auc_score(all_labels_np, all_probs_np, multi_class="ovr", average="macro")
            else:
                auc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
            results["auc"] = float(auc)
        except Exception:
            results["auc"] = None

    return results


# ─── Save checkpoint ─────────────────────────────────────────────────────────

def save_checkpoint(student, epoch: int, val_acc: float,
                    is_best_loss: bool, is_best_acc: bool,
                    output_dir: Path, cfg=None) -> None:
    """Save inference-compatible state dictionaries for both validation criteria."""
    ckpt_path = output_dir / "last_model.pth"
    torch.save(student.state_dict(), ckpt_path)

    if is_best_loss:
        torch.save(student.state_dict(), output_dir / "best_by_val_loss.pth")
        # Backward-compatible alias used by existing exporters and reports.
        torch.save(student.state_dict(), output_dir / "best_model.pth")

    if is_best_acc:
        torch.save(student.state_dict(), output_dir / "best_by_val_acc.pth")

    if cfg is not None and getattr(cfg, "save_epoch_checkpoints", False):
        start_epoch = int(getattr(cfg, "checkpoint_start_epoch", 80))
        interval = int(getattr(cfg, "checkpoint_interval", 10))
        if epoch >= start_epoch and (epoch - start_epoch) % interval == 0:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
            torch.save(student.state_dict(), epoch_path)


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _sampler_state(loader):
    sampler = getattr(loader, "batch_sampler", None)
    return sampler.state_dict() if hasattr(sampler, "state_dict") else None


def save_training_state(path, *, student, criterion, optimizer, scheduler,
                        scaler, train_loader, epoch, best_metrics, provenance):
    torch.save({
        "student": student.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "sampler": _sampler_state(train_loader),
        "epoch": int(epoch),
        "best_metrics": dict(best_metrics),
        "rng": _capture_rng_state(),
        "provenance": dict(provenance),
    }, path)


# ─── Plot training curves ─────────────────────────────────────────────────────

def plot_curves(history: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs    = [r["epoch"]     for r in history]
        train_acc = [r["train_acc"] for r in history]
        val_acc   = [r["val_acc"]   for r in history]
        train_loss= [r["train_loss"]for r in history]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        axes[0].plot(epochs, [a * 100 for a in train_acc], label="Train Acc")
        axes[0].plot(epochs, [a * 100 for a in val_acc],   label="Val Acc", linestyle="--")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("KD Student Accuracy")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Loss
        axes[1].plot(epochs, train_loss, label="Total Loss")
        if "loss_ce" in history[0]:
            axes[1].plot(epochs, [r["loss_ce"] for r in history], label="CE Loss",  linestyle=":")
            axes[1].plot(epochs, [r["loss_kd"] for r in history], label="KD Loss",  linestyle="-.")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("KD Training Losses")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close()
    except Exception as e:
        pass  # plot gagal tidak menghentikan training


# ─── EER (Equal Error Rate) ──────────────────────────────────────────────────

@torch.no_grad()
def compute_eer(student, loader, device) -> float:
    """
    Hitung Equal Error Rate per-class biometric verification scenario.
    Sama dengan retrain.py: untuk setiap identitas, hitung EER genuine vs impostor
    menggunakan probabilitas kelas sebagai skor, lalu rata-ratakan.
    """
    student.eval()

    all_probs  = []
    all_labels = []

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = student(images)
        probs  = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(targets.cpu().numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)   # (N, num_classes)
    all_labels = np.concatenate(all_labels, axis=0)   # (N,)

    eers = []
    for cls in np.unique(all_labels):
        y_bin  = (all_labels == cls).astype(int)      # genuine=1, impostor=0
        scores = all_probs[:, cls]                    # probabilitas kelas cls
        fpr, tpr, _ = roc_curve(y_bin, scores)
        fnr = 1.0 - tpr
        if len(fpr) > 1:
            try:
                eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
                eers.append(eer)
            except Exception:
                pass

    return float(np.mean(eers)) if eers else float("nan")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = parse_args(deepcopy(KD_CFG))

    # ── Setup ──
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # Pre-read student config so print_config shows the real C_init/num_cells
    if cfg.student_config_path and Path(cfg.student_config_path).exists():
        with open(cfg.student_config_path, "r") as _f:
            _sc = json.load(_f)
        cfg.student_C_init    = int(_sc.get("C_init",    cfg.student_C_init))
        cfg.student_num_cells = int(_sc.get("num_cells", cfg.student_num_cells))

    print_config(cfg)
    logger.info(f"Device: {device}")
    if str(device) == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Simpan config ──
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # ── Dataset ──
    logger.info("  Loading datasets...")
    train_loader, val_loader, test_loader, _ds_info = create_retrain_dataloaders(
        data_dir         = cfg.data_dir,
        split_path       = cfg.split_path,
        batch_size       = cfg.batch_size,
        num_workers      = cfg.num_workers,
        input_size       = cfg.input_size,
        use_augmentation = True,
        cutout_length    = cfg.cutout_length,
        augmentation_policy = cfg.augmentation_policy,
        sampler_type     = cfg.train_sampler,
        pk_p             = cfg.pk_p,
        pk_k             = cfg.pk_k,
        seed             = cfg.seed,
    )
    logger.info(
        f"  Train batches: {len(train_loader)}  |  "
        f"Val batches: {len(val_loader)}  |  "
        f"Test batches: {len(test_loader)}"
    )
    if cfg.train_sampler == "pk":
        logger.info(
            f"  Train sampler: PK  P={cfg.pk_p}  K={cfg.pk_k}  "
            f"effective_batch={cfg.pk_p * cfg.pk_k}"
        )

    # ── Models ──
    logger.info("\n  Memuat model...")
    if cfg.epochs > 0:
        teacher = load_teacher(cfg, device, logger)
    else:
        teacher = None
        logger.info("  Teacher: skipped (epochs=0, evaluation only)")
    student = load_student(cfg, device, logger)
    if cfg.kd_method == "adaptive_center_relation":
        if int(student.classifier.out_features) != int(_ds_info["num_classes"]):
            raise ValueError("Student classifier and dataset label-map class count differ")
        if teacher is not None and int(_find_final_linear(teacher).out_features) != int(_ds_info["num_classes"]):
            raise ValueError("Teacher classifier and dataset label-map class count differ")
        logger.info(
            "  Label-map compatibility: class dimensions match configured split. "
            "Teacher checkpoint has no embedded label-map hash; ordering is bound by the supplied split provenance."
        )
    anchor_student = None
    teacher2 = None
    if cfg.kd_method in {"conservative", "conservative_multiteacher"}:
        anchor_student = load_anchor_student(cfg, device, logger)
    if cfg.kd_method == "conservative_multiteacher" and cfg.epochs > 0:
        teacher2 = load_teacher2(cfg, device, logger)
    if cfg.freeze_bn:
        n_bn = freeze_batchnorm(student)
        n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e3
        logger.info(
            f"  Freeze BN enabled: {n_bn} BatchNorm layers fixed "
            f"| trainable params now {n_trainable:.1f}K"
        )

    # ── Initial evaluation before any KD update ──
    logger.info("  Evaluasi initial student sebelum KD...")
    initial_val_results = evaluate(student, val_loader, device)
    initial_test_results = None if cfg.skip_test_evaluation else evaluate(student, test_loader, device)
    logger.info(
        f"  Initial VAL  : acc={initial_val_results['acc']*100:.2f}% "
        f"loss={initial_val_results['loss']:.4f}"
    )
    if initial_test_results is not None:
        logger.info(
            f"  Initial TEST : acc={initial_test_results['acc']*100:.2f}% "
            f"loss={initial_test_results['loss']:.4f}"
        )
    else:
        logger.info("  Initial TEST : skipped by screening protocol")

    # Epoch 0 is a real candidate: the CE/AdaFace student before any KD update.
    # This prevents fine-tuning from silently replacing a stronger initializer.
    initial_state = student.state_dict()
    if not cfg.resume_training_state:
        torch.save(initial_state, output_dir / "initial_student.pth")
        torch.save(initial_state, output_dir / "best_by_val_loss.pth")
        torch.save(initial_state, output_dir / "best_by_val_acc.pth")
        torch.save(initial_state, output_dir / "best_screening.pth")
        torch.save(initial_state, output_dir / "best_model.pth")
        torch.save(initial_state, output_dir / "last_model.pth")
    logger.info(
        "  Epoch 0 checkpoint registered as initial candidate: "
        "initial_student.pth, best_by_val_loss.pth, best_by_val_acc.pth"
    )

    # ── Loss ──
    if cfg.kd_method == "hinton":
        criterion = HintonKDLoss(
            temperature     = cfg.temperature,
            alpha           = cfg.alpha,
            label_smoothing = cfg.label_smoothing,
        )
        logger.info(
            f"  Loss: HintonKD  T={cfg.temperature}  "
            f"alpha={cfg.alpha} (CE={cfg.alpha*100:.0f}%, KD={(1-cfg.alpha)*100:.0f}%)"
        )
    elif cfg.kd_method == "dkd":
        criterion = DecoupledKDLoss(
            temperature=cfg.temperature,
            alpha=cfg.dkd_alpha,
            beta=cfg.dkd_beta,
            warmup_epochs=cfg.dkd_warmup_epochs,
            label_smoothing=cfg.label_smoothing,
        ).to(device)
        logger.info(
            f"  Loss: DKD T={cfg.temperature} TCKD={cfg.dkd_alpha} "
            f"NCKD={cfg.dkd_beta} warmup={cfg.dkd_warmup_epochs} "
            f"classification={'AdaFace' if cfg.adaface else 'CE'}"
        )
    elif cfg.kd_method == "hard_topk":
        criterion = HardTopKMarginKDLoss(
            ce_weight=cfg.ce_weight,
            topk_k=cfg.topk_k,
            topk_weight=cfg.topk_weight,
            margin_weight=cfg.margin_weight,
            margin_m=cfg.margin_m,
            hard_weight=cfg.hard_weight,
            hard_margin_threshold=cfg.hard_margin_threshold,
            teacher_conf_threshold=cfg.teacher_conf_threshold,
            temperature=cfg.temperature,
            label_smoothing=cfg.label_smoothing,
        ).to(device)
        logger.info(
            f"  Loss: HardTopKMarginKD CE={cfg.ce_weight} "
            f"topk_k={cfg.topk_k} topk_w={cfg.topk_weight} "
            f"margin_w={cfg.margin_weight} margin_m={cfg.margin_m} "
            f"hard_w={cfg.hard_weight} hard_margin={cfg.hard_margin_threshold} "
            f"teacher_conf={cfg.teacher_conf_threshold}"
        )
    elif cfg.kd_method == "conservative":
        criterion = ConservativeAnchorKDLoss(
            ce_weight=cfg.ce_weight,
            topk_k=cfg.topk_k,
            topk_weight=cfg.topk_weight,
            margin_weight=cfg.margin_weight,
            margin_m=cfg.margin_m,
            anchor_weight=cfg.anchor_weight,
            temperature=cfg.temperature,
            anchor_temperature=cfg.anchor_temperature,
            label_smoothing=cfg.label_smoothing,
        ).to(device)
        logger.info(
            f"  Loss: ConservativeAnchorKD CE={cfg.ce_weight} "
            f"topk_k={cfg.topk_k} topk_w={cfg.topk_weight} "
            f"margin_w={cfg.margin_weight} margin_m={cfg.margin_m} "
            f"anchor_w={cfg.anchor_weight} anchor_T={cfg.anchor_temperature}"
        )
    elif cfg.kd_method == "conservative_multiteacher":
        criterion = ConservativeMultiTeacherKDLoss(
            ce_weight=cfg.ce_weight,
            topk_k=cfg.topk_k,
            teacher1_weight=cfg.teacher1_weight,
            teacher2_weight=cfg.teacher2_weight,
            teacher2_conf_threshold=cfg.teacher2_conf_threshold,
            teacher_agree_bonus=cfg.teacher_agree_bonus,
            teacher_disagree_policy=cfg.teacher_disagree_policy,
            anchor_weight=cfg.anchor_weight,
            temperature=cfg.temperature,
            anchor_temperature=cfg.anchor_temperature,
            label_smoothing=cfg.label_smoothing,
        ).to(device)
        logger.info(
            f"  Loss: ConservativeMultiTeacherKD CE={cfg.ce_weight} "
            f"topk_k={cfg.topk_k} teacher1_w={cfg.teacher1_weight} "
            f"teacher2_w={cfg.teacher2_weight} teacher2_conf={cfg.teacher2_conf_threshold} "
            f"agree_bonus={cfg.teacher_agree_bonus} policy={cfg.teacher_disagree_policy} "
            f"anchor_w={cfg.anchor_weight} anchor_T={cfg.anchor_temperature}"
        )
    elif cfg.kd_method == "topkd":
        topkd_temperature = cfg.topkd_temperature or cfg.temperature
        criterion = TopKDLoss(
            mode=cfg.topkd_mode,
            topkd_k=cfg.topkd_k,
            ce_weight=cfg.topkd_ce_weight,
            tdl_weight=cfg.topkd_tdl_weight,
            contrast_weight=cfg.topkd_contrast_weight,
            scale=cfg.topkd_scale,
            temperature=topkd_temperature,
            include_gt=cfg.topkd_include_gt,
            label_smoothing=cfg.label_smoothing,
        ).to(device)
        logger.info(
            f"  Loss: TopKD mode={cfg.topkd_mode} K={cfg.topkd_k} "
            f"CE={cfg.topkd_ce_weight} TDL={cfg.topkd_tdl_weight} "
            f"contrast={cfg.topkd_contrast_weight} scale={cfg.topkd_scale} "
            f"T={topkd_temperature} include_gt={cfg.topkd_include_gt}"
        )
    elif cfg.kd_method == "adaptive_center_relation":
        centers, center_metadata = prepare_teacher_center_bank(
            teacher, train_loader.dataset.samples, cfg, device, logger,
            _ds_info["label_map"],
        )
        criterion = AdaptiveCenterRelationLoss(
            student_dim=get_student_embedding_dim(student),
            teacher_dim=get_teacher_embedding_dim(teacher),
            num_classes=cfg.num_classes,
            initial_centers=centers,
            center_weight=cfg.center_weight,
            feature_weight=cfg.feature_weight,
            relation_weight=cfg.relation_weight,
            scale=cfg.center_scale,
            margin=cfg.center_margin,
            topk_negatives=cfg.relation_topk,
            difference_threshold=cfg.relation_difference_threshold,
            warmup_epochs=cfg.adaptive_warmup_epochs,
            label_smoothing=cfg.label_smoothing,
            progressive_staging=cfg.progressive_staging,
            center_start_epoch=cfg.progressive_center_start,
            relation_start_epoch=cfg.progressive_relation_start,
            calibration_batches=cfg.progressive_calibration_batches,
            center_grad_ratio=cfg.progressive_center_grad_ratio,
            feature_grad_ratio=cfg.progressive_feature_grad_ratio,
            relation_grad_ratio=cfg.progressive_relation_grad_ratio,
        ).to(device)
        logger.info(
            f"  Loss: {criterion.method_label}; progressive={cfg.progressive_staging}; "
            f"center_start={cfg.progressive_center_start}; "
            f"relation_start={cfg.progressive_relation_start}; ramp={cfg.adaptive_warmup_epochs}"
        )
    else:
        use_relation = cfg.kd_method in {"pairwise", "hybrid"}
        use_embedding = cfg.kd_method in {"embedding", "hybrid"} and cfg.embedding_weight > 0
        relation_weight = cfg.relation_weight if use_relation else 0.0
        embedding_weight = cfg.embedding_weight if use_embedding else 0.0
        logit_kd_weight = cfg.logit_kd_weight if cfg.kd_method == "hybrid" else 0.0

        criterion = HybridBiometricKDLoss(
            ce_weight=cfg.ce_weight,
            relation_weight=relation_weight,
            embedding_weight=embedding_weight,
            logit_kd_weight=logit_kd_weight,
            temperature=cfg.temperature,
            label_smoothing=cfg.label_smoothing,
            student_dim=get_student_embedding_dim(student),
            teacher_dim=get_teacher_embedding_dim(teacher),
        ).to(device)
        logger.info(
            f"  Loss: BiometricKD method={cfg.kd_method} "
            f"CE={cfg.ce_weight} relation={relation_weight} "
            f"embedding={embedding_weight} logit_kd={logit_kd_weight}"
        )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + [
            p for p in criterion.parameters() if p.requires_grad
        ],
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = GradScaler("cuda", enabled=cfg.amp)

    provenance = {
        "teacher_sha256": sha256_file(cfg.teacher_weights),
        "student_config_sha256": sha256_file(cfg.student_config_path),
        "split_sha256": sha256_file(cfg.split_path),
        "initial_student_sha256": (
            sha256_file(cfg.initial_student_weights) if cfg.initial_student_weights else
            sha256_file(cfg.student_weights) if not cfg.no_pretrained_student else None
        ),
        "method": cfg.kd_method,
        "method_label": getattr(criterion, "method_label", cfg.kd_method),
        "label_map_compatibility": (
            "class_dimension_plus_supplied_split; teacher checkpoint lacks embedded label-map hash"
            if cfg.kd_method == "adaptive_center_relation" else "not_audited_by_this_method"
        ),
        "continuation_type": cfg.continuation_type,
        "continuation_source_epoch": cfg.continuation_source_epoch,
        "teacher_center_cache": cfg.teacher_center_cache or None,
        "teacher_center_cache_sha256": (
            sha256_file(cfg.teacher_center_cache) if cfg.teacher_center_cache else None
        ),
    }
    # The cache path may be auto-resolved after the initial config dump.
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # ── Training loop ──
    best_val_acc  = initial_val_results["acc"]
    best_val_loss = initial_val_results["loss"]
    best_loss_epoch = 0
    best_acc_epoch = 0
    best_acc_tiebreak_loss = initial_val_results["loss"]
    best_screening = (
        initial_val_results["errors"], initial_val_results["loss"],
        -initial_val_results["true_class_margin"],
    )
    best_screening_epoch = 0
    start_epoch = 1
    history       = []

    if cfg.resume_training_state:
        state = torch.load(cfg.resume_training_state, map_location=device, weights_only=False)
        if state.get("provenance") != provenance:
            raise ValueError("Resume provenance does not match current experiment")
        student.load_state_dict(state["student"], strict=True)
        criterion.load_state_dict(state["criterion"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        sampler = getattr(train_loader, "batch_sampler", None)
        if state.get("sampler") is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(state["sampler"])
        _restore_rng_state(state["rng"])
        metrics = state["best_metrics"]
        best_val_acc = metrics["best_val_acc"]
        best_val_loss = metrics["best_val_loss"]
        best_loss_epoch = metrics["best_loss_epoch"]
        best_acc_epoch = metrics["best_acc_epoch"]
        best_acc_tiebreak_loss = metrics["best_acc_tiebreak_loss"]
        best_screening = tuple(metrics["best_screening"])
        best_screening_epoch = metrics["best_screening_epoch"]
        start_epoch = int(state["epoch"]) + 1
        logger.info(f"  Exact resume restored at epoch {state['epoch']}; next={start_epoch}")
    else:
        # Teacher/cache/model construction consumes RNG. Reset before epoch 1
        # so scratch ablations share the same sampling/augmentation stream.
        set_seed(cfg.seed)
        initial_best_metrics = {
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_loss_epoch": best_loss_epoch,
            "best_acc_epoch": best_acc_epoch,
            "best_acc_tiebreak_loss": best_acc_tiebreak_loss,
            "best_screening": list(best_screening),
            "best_screening_epoch": best_screening_epoch,
        }
        save_training_state(
            output_dir / "training_state_best.pth", student=student,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, train_loader=train_loader, epoch=0,
            best_metrics=initial_best_metrics, provenance=provenance,
        )
        save_training_state(
            output_dir / "training_state_last.pth", student=student,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, train_loader=train_loader, epoch=0,
            best_metrics=initial_best_metrics, provenance=provenance,
        )

    csv_path = output_dir / "training_log.csv"
    csv_mode = "a" if cfg.resume_training_state and csv_path.exists() else "w"
    training_log_fields = [
        "epoch", "train_loss", "train_ce", "train_kd", "train_center",
        "train_relation", "train_embedding", "train_logit_kd",
        "train_topk", "train_margin", "train_anchor",
        "train_teacher1_kd", "train_teacher2_kd", "teacher2_active",
        "train_tdl", "train_contrast", "train_topkd", "train_tckd", "train_nckd",
        "hard_ratio", "avg_true_rank", "weighted_center", "weighted_feature",
        "weighted_relation", "center_weight_effective", "feature_weight_effective",
        "relation_weight_effective", "grad_norm_ce", "grad_norm_center",
        "grad_norm_feature", "grad_norm_relation", "adaptive_stage",
        "train_acc", "val_loss", "val_acc", "lr", "time_s", "val_true_class_margin",
    ]
    with open(csv_path, csv_mode, newline="") as f:
        writer = csv.writer(f)
        if csv_mode == "w":
            writer.writerow(training_log_fields)
            initial_row = {key: "" for key in training_log_fields}
            initial_row.update({
                "epoch": 0,
                "val_loss": round(initial_val_results["loss"], 6),
                "val_acc": round(initial_val_results["acc"], 4),
                "val_true_class_margin": round(initial_val_results["true_class_margin"], 6),
            })
            writer.writerow([initial_row[key] for key in training_log_fields])

    logger.info("\n" + "=" * 70)
    logger.info(f"  Mulai KD Training  |  {cfg.epochs} epochs  |  device={device}")
    logger.info("=" * 70)

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        # Drop path schedule: linear 0 → drop_path_prob
        dp_prob = cfg.drop_path_prob * epoch / cfg.epochs
        student.set_drop_path_prob(dp_prob)
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch - 1)

        # Train
        (
            train_loss, train_ce, train_kd, train_center, train_acc,
            train_rel, train_emb, train_logit_kd,
            train_topk, train_margin, train_anchor,
            train_teacher1_kd, train_teacher2_kd, train_teacher2_active,
            train_hard_ratio, train_true_rank,
            train_tdl, train_contrast, train_topkd,
            train_tckd, train_nckd,
            progressive_metrics,
        ) = train_one_epoch(
            student, teacher, train_loader, optimizer, scheduler,
            criterion, scaler, device, epoch, cfg, logger,
            anchor=anchor_student,
            teacher2=teacher2,
        )

        # Validate
        val_results = evaluate(student, val_loader, device)
        val_acc  = val_results["acc"]
        val_loss = val_results["loss"]

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        is_best_loss = val_loss < best_val_loss
        if is_best_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch

        # Accuracy is primary for this file; validation loss breaks accuracy ties.
        is_best_acc = (
            val_acc > best_val_acc
            or (val_acc == best_val_acc and val_loss < best_acc_tiebreak_loss)
        )
        if is_best_acc:
            best_val_acc = val_acc
            best_acc_epoch = epoch
            best_acc_tiebreak_loss = val_loss

        screening_tuple = (
            val_results["errors"], val_loss, -val_results["true_class_margin"]
        )
        is_best_screening = screening_tuple < best_screening
        if is_best_screening:
            best_screening = screening_tuple
            best_screening_epoch = epoch
            torch.save(student.state_dict(), output_dir / "best_screening.pth")

        save_checkpoint(
            student, epoch, val_acc, is_best_loss, is_best_acc, output_dir, cfg
        )
        best_metrics = {
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_loss_epoch": best_loss_epoch,
            "best_acc_epoch": best_acc_epoch,
            "best_acc_tiebreak_loss": best_acc_tiebreak_loss,
            "best_screening": list(best_screening),
            "best_screening_epoch": best_screening_epoch,
        }
        save_training_state(
            output_dir / "training_state_last.pth", student=student,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, train_loader=train_loader, epoch=epoch,
            best_metrics=best_metrics, provenance=provenance,
        )
        if is_best_screening:
            save_training_state(
                output_dir / "training_state_best.pth", student=student,
                criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, train_loader=train_loader, epoch=epoch,
                best_metrics=best_metrics, provenance=provenance,
            )

        # Log
        logger.info(
            f"  E {epoch:3d}/{cfg.epochs} | "
            f"loss={train_loss:.4f} ce={train_ce:.4f} kd={train_kd:.4f} "
            f"center={train_center:.4f} "
            f"rel={train_rel:.4f} emb={train_emb:.4f} logit_kd={train_logit_kd:.4f} "
            f"topk={train_topk:.4f} margin={train_margin:.4f} anchor={train_anchor:.4f} "
            f"tdl={train_tdl:.4f} contrast={train_contrast:.4f} topkd={train_topkd:.4f} "
            f"tckd={train_tckd:.4f} nckd={train_nckd:.4f} "
            f"t1={train_teacher1_kd:.4f} t2={train_teacher2_kd:.4f} "
            f"t2_active={train_teacher2_active:.2f} "
            f"hard={train_hard_ratio:.2f} rank={train_true_rank:.2f} "
            f"stage={progressive_metrics['adaptive_stage']:.1f} "
            f"w(c/f/r)={progressive_metrics['center_weight_effective']:.4g}/"
            f"{progressive_metrics['feature_weight_effective']:.4g}/"
            f"{progressive_metrics['relation_weight_effective']:.4g} "
            f"g(ce/c/f/r)={progressive_metrics['grad_norm_ce']:.3g}/"
            f"{progressive_metrics['grad_norm_center']:.3g}/"
            f"{progressive_metrics['grad_norm_feature']:.3g}/"
            f"{progressive_metrics['grad_norm_relation']:.3g} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"{'** BEST_LOSS' if is_best_loss else ''}"
            f"{' ** BEST_ACC' if is_best_acc else ''} | "
            f"lr={current_lr:.2e}  {elapsed:.1f}s"
        )

        row = {
            "epoch"     : epoch,
            "train_loss": round(train_loss, 6),
            "train_ce"  : round(train_ce, 6),
            "train_kd"  : round(train_kd, 6),
            "train_center": round(train_center, 6),
            "train_relation": round(train_rel, 6),
            "train_embedding": round(train_emb, 6),
            "train_logit_kd": round(train_logit_kd, 6),
            "train_topk": round(train_topk, 6),
            "train_margin": round(train_margin, 6),
            "train_anchor": round(train_anchor, 6),
            "train_teacher1_kd": round(train_teacher1_kd, 6),
            "train_teacher2_kd": round(train_teacher2_kd, 6),
            "teacher2_active": round(train_teacher2_active, 6),
            "train_tdl": round(train_tdl, 6),
            "train_contrast": round(train_contrast, 6),
            "train_topkd": round(train_topkd, 6),
            "train_tckd": round(train_tckd, 6),
            "train_nckd": round(train_nckd, 6),
            "hard_ratio": round(train_hard_ratio, 6),
            "avg_true_rank": round(train_true_rank, 6),
            **{key: round(value, 8) for key, value in progressive_metrics.items()},
            "train_acc" : round(train_acc, 4),
            "val_loss"  : round(val_loss, 6),
            "val_acc"   : round(val_acc, 4),
            "lr"        : round(current_lr, 8),
            "time_s"    : round(elapsed, 1),
            "val_true_class_margin": round(val_results["true_class_margin"], 6),
        }
        history.append(row)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(key, "") for key in training_log_fields])

    # ── Final test evaluation ──
    logger.info("\n" + "=" * 70)
    logger.info(
        f"  Training selesai. Best val_loss epoch={best_loss_epoch} "
        f"best_val_loss={best_val_loss:.6f} | best val_acc epoch={best_acc_epoch} "
        f"best_val_acc={best_val_acc:.4f}"
    )

    best_model_path = output_dir / "best_by_val_loss.pth"
    if best_model_path.exists():
        logger.info("  Memuat best_by_val_loss.pth untuk evaluasi test...")
        best_state = torch.load(best_model_path, map_location=device)
        student.load_state_dict(best_state)
    else:
        logger.info("  best_by_val_loss.pth tidak ada — evaluasi menggunakan weights yang sudah di-load.")

    test_results = None
    if not cfg.skip_test_evaluation:
        logger.info("  Evaluasi TEST set...")
        test_results = evaluate(student, test_loader, device, compute_auc=True)

    # EER
    try:
        if test_results is None:
            raise RuntimeError("test evaluation disabled")
        eer = compute_eer(student, test_loader, device)
        test_results["eer_pct"] = round(eer * 100, 4)
    except Exception as e:
        if test_results is not None:
            test_results["eer_pct"] = None
            logger.warning(f"  EER gagal dihitung: {e}")

    if test_results is not None:
        logger.info("=" * 70)
        logger.info(f"  TEST ACCURACY  : {test_results['acc']*100:.2f}%")
        logger.info(f"  TEST LOSS      : {test_results['loss']:.4f}")
        logger.info(f"  TEST AUC       : {test_results.get('auc', 'N/A')}")
        logger.info(f"  TEST EER       : {test_results.get('eer_pct', 'N/A')}%")
        logger.info("=" * 70)
    else:
        logger.info("  TEST evaluation skipped; screening results contain validation metrics only.")

    # ── Simpan hasil ──
    final_results = {
        "timestamp"    : datetime.now().isoformat(),
        "best_epoch"   : best_loss_epoch,
        "best_val_loss_epoch": best_loss_epoch,
        "best_val_acc_epoch": best_acc_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_acc" : round(best_val_acc, 4),
        "best_screening_epoch": best_screening_epoch,
        "best_screening_key": {
            "validation_errors": int(best_screening[0]),
            "validation_ce_loss": float(best_screening[1]),
            "negative_true_class_margin": float(best_screening[2]),
        },
        "initial_val_acc": round(initial_val_results["acc"], 4),
        "initial_val_loss": round(initial_val_results["loss"], 4),
        "initial_test_acc": None if initial_test_results is None else round(initial_test_results["acc"], 4),
        "initial_test_loss": None if initial_test_results is None else round(initial_test_results["loss"], 4),
        "initial_student_checkpoint": str(output_dir / "initial_student.pth"),
        "best_by_val_loss_checkpoint": str(output_dir / "best_by_val_loss.pth"),
        "best_by_val_acc_checkpoint": str(output_dir / "best_by_val_acc.pth"),
        "best_screening_checkpoint": str(output_dir / "best_screening.pth"),
        "last_model_checkpoint": str(output_dir / "last_model.pth"),
        "test_acc"     : None if test_results is None else round(test_results["acc"], 4),
        "test_loss"    : None if test_results is None else round(test_results["loss"], 4),
        "test_auc"     : None if test_results is None else test_results.get("auc"),
        "test_eer_pct" : None if test_results is None else test_results.get("eer_pct"),
        "kd_config"    : {
            "teacher"    : cfg.teacher_arch,
            "teacher_weights": cfg.teacher_weights,
            "teacher2"   : cfg.teacher2_arch,
            "teacher2_weights": cfg.teacher2_weights,
            "kd_method"  : cfg.kd_method,
            "temperature": cfg.temperature,
            "alpha"      : cfg.alpha,
            "dkd_alpha"  : cfg.dkd_alpha,
            "dkd_beta"   : cfg.dkd_beta,
            "dkd_warmup_epochs": cfg.dkd_warmup_epochs,
            "adaface"    : cfg.adaface,
            "skip_test_evaluation": cfg.skip_test_evaluation,
            "kd_weight"  : round(1 - cfg.alpha, 2),
            "ce_weight"  : cfg.ce_weight,
            "relation_weight": cfg.relation_weight,
            "center_weight": cfg.center_weight,
            "feature_weight": cfg.feature_weight,
            "center_scale": cfg.center_scale,
            "center_margin": cfg.center_margin,
            "relation_topk": cfg.relation_topk,
            "relation_difference_threshold": cfg.relation_difference_threshold,
            "adaptive_warmup_epochs": cfg.adaptive_warmup_epochs,
            "teacher_center_cache": cfg.teacher_center_cache,
            "initial_student_weights": cfg.initial_student_weights,
            "embedding_weight": cfg.embedding_weight,
            "logit_kd_weight": cfg.logit_kd_weight,
            "topk_k"     : cfg.topk_k,
            "topk_weight": cfg.topk_weight,
            "margin_weight": cfg.margin_weight,
            "margin_m"   : cfg.margin_m,
            "hard_weight": cfg.hard_weight,
            "hard_margin_threshold": cfg.hard_margin_threshold,
            "teacher_conf_threshold": cfg.teacher_conf_threshold,
            "anchor_weights": cfg.anchor_weights,
            "anchor_weight": cfg.anchor_weight,
            "anchor_temperature": cfg.anchor_temperature,
            "teacher1_weight": cfg.teacher1_weight,
            "teacher2_weight": cfg.teacher2_weight,
            "teacher2_conf_threshold": cfg.teacher2_conf_threshold,
            "teacher_agree_bonus": cfg.teacher_agree_bonus,
            "teacher_disagree_policy": cfg.teacher_disagree_policy,
            "topkd_mode": cfg.topkd_mode,
            "topkd_k": cfg.topkd_k,
            "topkd_ce_weight": cfg.topkd_ce_weight,
            "topkd_tdl_weight": cfg.topkd_tdl_weight,
            "topkd_contrast_weight": cfg.topkd_contrast_weight,
            "topkd_scale": cfg.topkd_scale,
            "topkd_temperature": cfg.topkd_temperature or cfg.temperature,
            "topkd_include_gt": cfg.topkd_include_gt,
            "freeze_bn"  : cfg.freeze_bn,
            "cutout_length": cfg.cutout_length,
            "augmentation_policy": cfg.augmentation_policy,
            "train_sampler": cfg.train_sampler,
            "num_workers"  : cfg.num_workers,
            "pk_p"       : cfg.pk_p,
            "pk_k"       : cfg.pk_k,
            "save_epoch_checkpoints": getattr(cfg, "save_epoch_checkpoints", False),
            "checkpoint_start_epoch": getattr(cfg, "checkpoint_start_epoch", 80),
            "checkpoint_interval": getattr(cfg, "checkpoint_interval", 10),
            "epochs"     : cfg.epochs,
            "lr"         : cfg.lr,
        },
        "provenance": provenance,
    }
    result_name = "screening_results.json" if cfg.skip_test_evaluation else "test_results.json"
    with open(output_dir / result_name, "w") as f:
        json.dump(final_results, f, indent=2)

    # Plot
    plot_curves(history, output_dir)
    logger.info(f"\n  Output disimpan di: {output_dir}")
    logger.info("  Done.")

    return final_results


if __name__ == "__main__":
    main()
