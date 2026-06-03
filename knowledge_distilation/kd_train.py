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
from kd_loss import HintonKDLoss
from model_eval import EvalNetwork
from palm_vein_dataset import create_retrain_dataloaders


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
    parser.add_argument("--student_weights",   default=cfg.student_weights)
    parser.add_argument("--student_config",    default=cfg.student_config_path)
    parser.add_argument("--output_dir", default=None,
                        help="Folder hasil. Default: auto-generate dari parameter.")

    parser.add_argument("--temperature", type=float, default=cfg.temperature,
                        help="KD temperature τ (default: 4.0)")
    parser.add_argument("--alpha",       type=float, default=cfg.alpha,
                        help="CE weight α. KD weight = 1-α (default: 0.3)")
    parser.add_argument("--epochs",      type=int,   default=cfg.epochs)
    parser.add_argument("--lr",          type=float, default=cfg.lr)
    parser.add_argument("--lr_min",      type=float, default=cfg.lr_min)
    parser.add_argument("--weight_decay",type=float, default=cfg.weight_decay)
    parser.add_argument("--batch_size",  type=int,   default=cfg.batch_size)
    parser.add_argument("--warmup_epochs", type=int, default=cfg.warmup_epochs)
    parser.add_argument("--scheduler",     default=cfg.scheduler,
                        choices=["cosine", "sgdr"],
                        help="LR scheduler: 'cosine' (default) atau 'sgdr' (warm restarts)")
    parser.add_argument("--sgdr_T0",       type=int, default=cfg.sgdr_T0,
                        help="SGDR: epoch per cycle pertama (default: 50)")
    parser.add_argument("--sgdr_T_mult",   type=int, default=cfg.sgdr_T_mult,
                        help="SGDR: multiplier panjang cycle (default: 2)")
    parser.add_argument("--drop_path",   type=float, default=cfg.drop_path_prob)
    parser.add_argument("--seed",        type=int,   default=cfg.seed)
    parser.add_argument("--no_amp",      action="store_true",
                        help="Disable Automatic Mixed Precision")
    parser.add_argument("--no_pretrained_student", action="store_true",
                        help="Train student from scratch (random init, ignore --student_weights)")

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

    args = parser.parse_args()

    # Update cfg dengan nilai dari argparse
    cfg.teacher_arch        = args.teacher_arch
    cfg.teacher_weights     = args.teacher_weights
    cfg.student_weights     = args.student_weights
    cfg.student_config_path = args.student_config
    cfg.temperature         = args.temperature
    cfg.alpha               = args.alpha
    cfg.epochs              = args.epochs
    cfg.lr                  = args.lr
    cfg.lr_min              = args.lr_min
    cfg.weight_decay        = args.weight_decay
    cfg.batch_size          = args.batch_size
    cfg.warmup_epochs       = args.warmup_epochs
    cfg.scheduler           = args.scheduler
    cfg.sgdr_T0             = args.sgdr_T0
    cfg.sgdr_T_mult         = args.sgdr_T_mult
    cfg.drop_path_prob      = args.drop_path
    cfg.seed                = args.seed
    if args.no_amp:
        cfg.amp = False
    cfg.no_pretrained_student = args.no_pretrained_student

    # MixUp / CutMix
    if args.no_mix:
        cfg.mixup_alpha  = 0.0
        cfg.cutmix_alpha = 0.0
    else:
        cfg.mixup_alpha     = args.mixup_alpha
        cfg.cutmix_alpha    = args.cutmix_alpha
        cfg.mix_prob        = args.mix_prob
        cfg.mix_switch_prob = args.mix_switch_prob

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
    "regnet_y_16gf", "mobilenet_v3_large", "vgg16",
]


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
    arch = cfg.teacher_arch
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
        # Remap keys jika teacher di-train dengan fc sebagai nn.Sequential([Dropout, Linear])
        # (key "fc.1.*" dan "AuxLogits.fc.1.*") → plain nn.Linear ("fc.*", "AuxLogits.fc.*")
        _raw_sd = torch.load(cfg.teacher_weights, map_location="cpu")
        _remapped = {}
        for k, v in _raw_sd.items():
            nk = k.replace("fc.1.weight", "fc.weight") \
                  .replace("fc.1.bias",   "fc.bias") \
                  .replace("AuxLogits.fc.1.weight", "AuxLogits.fc.weight") \
                  .replace("AuxLogits.fc.1.bias",   "AuxLogits.fc.bias")
            _remapped[nk] = v
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
    teacher.load_state_dict(state_dict, strict=True)

    teacher.to(device)
    teacher.eval()

    # Freeze semua parameter teacher
    for p in teacher.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    logger.info(f"  Teacher loaded: {n_params:.1f}M params  |  FROZEN")
    return teacher


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
    logger.info(f"  Student arch: C_init={c_init}, num_cells={num_cells}")

    student = EvalNetwork(
        genotype    = genotype,
        C_init      = c_init,
        num_cells   = num_cells,
        num_classes = cfg.num_classes,
        auxiliary   = False,   # KD: hanya pakai main head
        dropout     = cfg.student_dropout,
    )

    if cfg.no_pretrained_student:
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
):
    student.train()
    total_loss = total_ce = total_kd = correct = n_samples = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ── MixUp / CutMix (batch-level augmentation) ───────────────
        mixed_images, targets_a, targets_b, lam, is_mixed = \
            apply_mixup_cutmix(images, targets, cfg)

        # Teacher forward — tidak perlu gradien
        # Teacher melihat gambar yang SAMA (mixed) dengan student
        with torch.no_grad():
            logits_teacher = teacher(mixed_images)

        optimizer.zero_grad(set_to_none=True)

        # AMP forward
        with autocast("cuda", enabled=cfg.amp):
            logits_student = student(mixed_images)
            # Student dengan auxiliary=False selalu return tensor tunggal
            if is_mixed:
                mix_targets = (targets_a, targets_b, lam)
                loss, breakdown = criterion(logits_student, logits_teacher,
                                            targets_a, mix_targets=mix_targets)
            else:
                loss, breakdown = criterion(logits_student, logits_teacher, targets)

        scaler.scale(loss).backward()
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
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
        total_kd   += breakdown["loss_kd"]

        if (batch_idx + 1) % cfg.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.debug(
                f"  E{epoch:3d} [{batch_idx+1:4d}/{len(loader)}] "
                f"loss={breakdown['loss_total']:.4f} "
                f"ce={breakdown['loss_ce']:.4f} "
                f"kd={breakdown['loss_kd']:.4f} "
                f"lr={current_lr:.2e}"
            )

    n_batches  = len(loader)
    train_loss = total_loss / n_batches
    train_ce   = total_ce   / n_batches
    train_kd   = total_kd   / n_batches
    train_acc  = correct    / n_samples

    return train_loss, train_ce, train_kd, train_acc


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

        if compute_auc:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(targets.cpu().numpy().tolist())

    results = {
        "acc" : correct / n_samples,
        "loss": total_loss / len(loader),
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
                    is_best: bool, output_dir: Path) -> None:
    """Simpan state_dict saja (tidak perlu optimizer untuk inference)."""
    ckpt_path = output_dir / "last_model.pth"
    torch.save(student.state_dict(), ckpt_path)

    if is_best:
        best_path = output_dir / "best_model.pth"
        torch.save(student.state_dict(), best_path)


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
    )
    logger.info(
        f"  Train batches: {len(train_loader)}  |  "
        f"Val batches: {len(val_loader)}  |  "
        f"Test batches: {len(test_loader)}"
    )

    # ── Models ──
    logger.info("\n  Memuat model...")
    if cfg.epochs > 0:
        teacher = load_teacher(cfg, device, logger)
    else:
        teacher = None
        logger.info("  Teacher: skipped (epochs=0, evaluation only)")
    student = load_student(cfg, device, logger)

    # ── Loss ──
    criterion = HintonKDLoss(
        temperature     = cfg.temperature,
        alpha           = cfg.alpha,
        label_smoothing = 0.1,
    )
    logger.info(
        f"  Loss: HintonKD  T={cfg.temperature}  "
        f"alpha={cfg.alpha} (CE={cfg.alpha*100:.0f}%, KD={(1-cfg.alpha)*100:.0f}%)"
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = GradScaler("cuda", enabled=cfg.amp)

    # ── Training loop ──
    best_val_acc  = 0.0
    best_epoch    = 0
    history       = []

    csv_path = output_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_ce", "train_kd",
            "train_acc", "val_loss", "val_acc", "lr", "time_s",
        ])

    logger.info("\n" + "=" * 70)
    logger.info(f"  Mulai KD Training  |  {cfg.epochs} epochs  |  device={device}")
    logger.info("=" * 70)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Drop path schedule: linear 0 → drop_path_prob
        dp_prob = cfg.drop_path_prob * epoch / cfg.epochs
        student.set_drop_path_prob(dp_prob)

        # Train
        train_loss, train_ce, train_kd, train_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, scheduler,
            criterion, scaler, device, epoch, cfg, logger,
        )

        # Validate
        val_results = evaluate(student, val_loader, device)
        val_acc  = val_results["acc"]
        val_loss = val_results["loss"]

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch   = epoch

        save_checkpoint(student, epoch, val_acc, is_best, output_dir)

        # Log
        logger.info(
            f"  E {epoch:3d}/{cfg.epochs} | "
            f"loss={train_loss:.4f} ce={train_ce:.4f} kd={train_kd:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"{'** BEST' if is_best else ''} | "
            f"lr={current_lr:.2e}  {elapsed:.1f}s"
        )

        row = {
            "epoch"     : epoch,
            "train_loss": round(train_loss, 6),
            "train_ce"  : round(train_ce, 6),
            "train_kd"  : round(train_kd, 6),
            "train_acc" : round(train_acc, 4),
            "val_loss"  : round(val_loss, 6),
            "val_acc"   : round(val_acc, 4),
            "lr"        : round(current_lr, 8),
            "time_s"    : round(elapsed, 1),
        }
        history.append(row)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(list(row.values()))

    # ── Final test evaluation ──
    logger.info("\n" + "=" * 70)
    logger.info(f"  Training selesai. Best epoch={best_epoch}  best_val_acc={best_val_acc:.4f}")

    best_model_path = output_dir / "best_model.pth"
    if best_model_path.exists():
        logger.info("  Memuat best_model.pth untuk evaluasi test...")
        best_state = torch.load(best_model_path, map_location=device)
        student.load_state_dict(best_state)
    else:
        logger.info("  best_model.pth tidak ada (epochs=0) — evaluasi menggunakan weights yang sudah di-load.")

    logger.info("  Evaluasi TEST set...")
    test_results = evaluate(student, test_loader, device, compute_auc=True)

    # EER
    try:
        eer = compute_eer(student, test_loader, device)
        test_results["eer_pct"] = round(eer * 100, 4)
    except Exception as e:
        test_results["eer_pct"] = None
        logger.warning(f"  EER gagal dihitung: {e}")

    logger.info("=" * 70)
    logger.info(f"  TEST ACCURACY  : {test_results['acc']*100:.2f}%")
    logger.info(f"  TEST LOSS      : {test_results['loss']:.4f}")
    logger.info(f"  TEST AUC       : {test_results.get('auc', 'N/A')}")
    logger.info(f"  TEST EER       : {test_results.get('eer_pct', 'N/A')}%")
    logger.info("=" * 70)

    # ── Simpan hasil ──
    final_results = {
        "timestamp"    : datetime.now().isoformat(),
        "best_epoch"   : best_epoch,
        "best_val_acc" : round(best_val_acc, 4),
        "test_acc"     : round(test_results["acc"], 4),
        "test_loss"    : round(test_results["loss"], 4),
        "test_auc"     : test_results.get("auc"),
        "test_eer_pct" : test_results.get("eer_pct"),
        "kd_config"    : {
            "teacher"    : cfg.teacher_arch,
            "temperature": cfg.temperature,
            "alpha"      : cfg.alpha,
            "kd_weight"  : round(1 - cfg.alpha, 2),
            "epochs"     : cfg.epochs,
            "lr"         : cfg.lr,
        },
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Plot
    plot_curves(history, output_dir)
    logger.info(f"\n  Output disimpan di: {output_dir}")
    logger.info("  Done.")

    return final_results


if __name__ == "__main__":
    main()
