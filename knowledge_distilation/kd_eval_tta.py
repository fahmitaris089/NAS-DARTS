"""
Test-Time Augmentation (TTA) Evaluation for KD Student Model
=============================================================
Evaluate any trained KD student model with TTA — multiple augmented
views per image, averaged softmax probabilities, then argmax.

Zero training cost — just re-evaluates existing best_model.pth.

Usage:
    cd Student/
    python knowledge_distilation/kd_eval_tta.py \
        --model_dir knowledge_distilation/kd_results/run5_efficientNetV2M_t10_a0.5_e350

    # Custom TTA views:
    python knowledge_distilation/kd_eval_tta.py \
        --model_dir ... --tta_views 7 --split both

    # No TTA (sanity check, should match original val_acc):
    python knowledge_distilation/kd_eval_tta.py \
        --model_dir ... --no_tta
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve

# ─── Pastikan root project ada di path ────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

from genotypes import dict_to_genotype
from kd_config import KDConfig
from model_eval import EvalNetwork
from palm_vein_dataset import (
    PalmVeinDataset,
    build_image_list,
    build_label_map,
    load_split,
)

# ─── Constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


# ─── TTA Transform Functions ─────────────────────────────────────────────────

def _base_transform(img: Image.Image) -> torch.Tensor:
    """Standard val transform: Resize → ToTensor → Gray→RGB → Normalize."""
    img = TF.resize(img, [INPUT_SIZE, INPUT_SIZE])
    t = TF.to_tensor(img)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t


def _hflip_transform(img: Image.Image) -> torch.Tensor:
    """Horizontal flip."""
    img = TF.hflip(img)
    return _base_transform(img)


def _rotate_transform(img: Image.Image, angle: float) -> torch.Tensor:
    """Rotation by given degrees."""
    img = TF.rotate(img, angle)
    return _base_transform(img)


def _brightness_transform(img: Image.Image, factor: float) -> torch.Tensor:
    """Brightness adjustment."""
    img = TF.adjust_brightness(img, factor)
    return _base_transform(img)


def _translate_transform(img: Image.Image, dx: float, dy: float) -> torch.Tensor:
    """Small translation (fraction of image size)."""
    img = TF.resize(img, [INPUT_SIZE, INPUT_SIZE])
    px_x = int(dx * INPUT_SIZE)
    px_y = int(dy * INPUT_SIZE)
    img = TF.affine(img, angle=0, translate=[px_x, px_y], scale=1.0, shear=0)
    t = TF.to_tensor(img)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t


def get_tta_transforms(n_views: int = 5):
    """
    Return list of (name, transform_fn) pairs for TTA.

    Available views (ordered by expected usefulness):
      1. original          — identity
      2. hflip             — horizontal flip
      3. rotate +5°        — slight clockwise
      4. rotate −5°        — slight counter-clockwise
      5. brightness 1.1    — slightly brighter
      6. brightness 0.9    — slightly darker
      7. translate (+2%,0) — shift right
      8. translate (0,+2%) — shift down
      9. rotate +3°
     10. rotate −3°

    Returns first n_views from this list.
    """
    all_transforms = [
        ("original",       lambda img: _base_transform(img)),
        ("hflip",          lambda img: _hflip_transform(img)),
        ("rotate_+5",     lambda img: _rotate_transform(img, 5.0)),
        ("rotate_-5",     lambda img: _rotate_transform(img, -5.0)),
        ("bright_1.1",    lambda img: _brightness_transform(img, 1.1)),
        ("bright_0.9",    lambda img: _brightness_transform(img, 0.9)),
        ("translate_r",   lambda img: _translate_transform(img, 0.02, 0.0)),
        ("translate_d",   lambda img: _translate_transform(img, 0.0, 0.02)),
        ("rotate_+3",     lambda img: _rotate_transform(img, 3.0)),
        ("rotate_-3",     lambda img: _rotate_transform(img, -3.0)),
    ]
    n_views = min(n_views, len(all_transforms))
    return all_transforms[:n_views]


# ─── Load Student Model ──────────────────────────────────────────────────────

def load_student_from_dir(model_dir: Path, device: torch.device) -> nn.Module:
    """Load student model from a KD results directory."""
    config_path = model_dir / "config.json"
    best_model_path = model_dir / "best_model.pth"

    assert config_path.exists(), f"config.json not found in {model_dir}"
    assert best_model_path.exists(), f"best_model.pth not found in {model_dir}"

    with open(config_path, "r") as f:
        saved_cfg = json.load(f)

    # Load student architecture from student_config
    student_config_path = saved_cfg.get("student_config_path")
    if student_config_path is None:
        raise ValueError("student_config_path not found in config.json")

    with open(student_config_path, "r") as f:
        retrain_cfg = json.load(f)

    genotype = dict_to_genotype(retrain_cfg["genotype"])
    c_init = int(retrain_cfg.get("C_init", 8))
    num_cells = int(retrain_cfg.get("num_cells", 8))
    num_classes = int(saved_cfg.get("num_classes", 834))
    dropout = float(saved_cfg.get("student_dropout", 0.3))

    student = EvalNetwork(
        genotype=genotype,
        C_init=c_init,
        num_cells=num_cells,
        num_classes=num_classes,
        auxiliary=False,
        dropout=dropout,
    )

    state_dict = torch.load(best_model_path, map_location="cpu")
    student.load_state_dict(state_dict, strict=False)
    student.to(device)
    student.eval()

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e3
    print(f"  Student loaded: {n_params:.1f}K params, C_init={c_init}, num_cells={num_cells}")
    return student, num_classes


# ─── TTA Evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_tta(
    student: nn.Module,
    samples: list,
    device: torch.device,
    tta_transforms: list,
    batch_size: int = 64,
    compute_auc: bool = True,
) -> dict:
    """
    TTA evaluation: for each image, apply multiple augmented views,
    average softmax probabilities across views, then predict.

    Args:
        student:         Model in eval mode
        samples:         List of (image_path, label) tuples
        device:          torch device
        tta_transforms:  List of (name, transform_fn) pairs
        batch_size:      Batch images for GPU efficiency
        compute_auc:     Whether to compute AUC and EER

    Returns:
        dict with acc, loss, auc, eer_pct, per_view_acc, n_samples
    """
    student.eval()
    n_views = len(tta_transforms)
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total_loss = 0.0
    n_samples = len(samples)

    all_avg_probs = []  # (N, num_classes) — averaged probabilities
    all_labels = []
    per_view_correct = [0] * n_views  # track each view's individual accuracy

    print(f"  TTA evaluation: {n_samples} samples × {n_views} views = {n_samples * n_views} forward passes")

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_samples = samples[start_idx:end_idx]
        batch_size_actual = len(batch_samples)

        # Load images once (PIL)
        pil_images = []
        labels = []
        for img_path, label in batch_samples:
            pil_images.append(Image.open(img_path).convert("L"))
            labels.append(label)

        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

        # For each TTA view, transform all images in batch → forward → collect softmax
        view_probs = []  # list of (B, num_classes) tensors
        for v_idx, (v_name, v_fn) in enumerate(tta_transforms):
            # Transform images
            tensors = [v_fn(img) for img in pil_images]
            batch_tensor = torch.stack(tensors).to(device, non_blocking=True)

            # Forward pass
            logits = student(batch_tensor)
            probs = torch.softmax(logits, dim=1)  # (B, num_classes)
            view_probs.append(probs)

            # Per-view accuracy
            pred_v = logits.argmax(dim=1)
            per_view_correct[v_idx] += (pred_v == labels_tensor).sum().item()

        # Average softmax across views
        stacked = torch.stack(view_probs, dim=0)  # (n_views, B, num_classes)
        avg_probs = stacked.mean(dim=0)  # (B, num_classes)

        # Predictions from averaged probabilities
        pred = avg_probs.argmax(dim=1)
        correct += (pred == labels_tensor).sum().item()

        # Loss on averaged log-probs
        loss = criterion(torch.log(avg_probs + 1e-10), labels_tensor)
        total_loss += loss.item() * batch_size_actual

        if compute_auc:
            all_avg_probs.append(avg_probs.cpu().numpy())
            all_labels.extend(labels)

        # Progress
        done = min(end_idx, n_samples)
        if done % (batch_size * 4) == 0 or done == n_samples:
            print(f"    [{done}/{n_samples}] running acc={correct/done*100:.2f}%")

    # ─── Aggregate results ──
    results = {
        "acc": correct / n_samples,
        "loss": total_loss / n_samples,
        "n_samples": n_samples,
        "n_views": n_views,
    }

    # Per-view accuracy
    view_accs = {}
    for v_idx, (v_name, _) in enumerate(tta_transforms):
        view_accs[v_name] = round(per_view_correct[v_idx] / n_samples, 4)
    results["per_view_acc"] = view_accs

    # AUC
    if compute_auc and all_avg_probs:
        all_probs_np = np.concatenate(all_avg_probs, axis=0)
        all_labels_np = np.array(all_labels)
        try:
            auc = roc_auc_score(all_labels_np, all_probs_np, multi_class="ovr", average="macro")
            results["auc"] = float(auc)
        except Exception:
            results["auc"] = None

        # EER
        try:
            eers = []
            for cls in np.unique(all_labels_np):
                y_bin = (all_labels_np == cls).astype(int)
                scores = all_probs_np[:, cls]
                fpr, tpr, _ = roc_curve(y_bin, scores)
                fnr = 1.0 - tpr
                if len(fpr) > 1:
                    try:
                        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
                        eers.append(eer)
                    except Exception:
                        pass
            results["eer_pct"] = round(float(np.mean(eers)) * 100, 4) if eers else None
        except Exception:
            results["eer_pct"] = None

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TTA Evaluation for KD Student")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to KD results folder (containing best_model.pth, config.json)")
    parser.add_argument("--split", type=str, default="both", choices=["val", "test", "both"],
                        help="Which split to evaluate: val, test, or both (default: both)")
    parser.add_argument("--tta_views", type=int, default=5,
                        help="Number of TTA views per image (default: 5, max: 10)")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable TTA — single forward pass only (sanity check)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    assert model_dir.exists(), f"Model directory not found: {model_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  TTA Evaluation")
    print(f"{'='*60}")
    print(f"  Model dir   : {model_dir}")
    print(f"  Device      : {device}")
    print(f"  Split       : {args.split}")

    if args.no_tta:
        n_views = 1
        print(f"  TTA         : OFF (single view, sanity check)")
    else:
        n_views = min(args.tta_views, 10)
        print(f"  TTA views   : {n_views}")

    # ── Load config ──
    with open(model_dir / "config.json", "r") as f:
        saved_cfg = json.load(f)

    data_dir = saved_cfg.get("data_dir")
    split_path = saved_cfg.get("split_path")
    assert data_dir and Path(data_dir).exists(), f"data_dir not found: {data_dir}"
    assert split_path and Path(split_path).exists(), f"split_path not found: {split_path}"

    # ── Load student ──
    student, num_classes = load_student_from_dir(model_dir, device)

    # ── Load data ──
    split = load_split(split_path)
    label_map = build_label_map(split["subjects"])

    val_samples = build_image_list(data_dir, split["val"], label_map)
    test_samples = build_image_list(data_dir, split["test"], label_map)
    print(f"  Val samples : {len(val_samples)}")
    print(f"  Test samples: {len(test_samples)}")

    # ── TTA transforms ──
    tta_transforms = get_tta_transforms(n_views)
    view_names = [name for name, _ in tta_transforms]
    print(f"  Views       : {view_names}")
    print(f"{'='*60}\n")

    results = {}

    # ── Evaluate ──
    if args.split in ("val", "both"):
        print("─── Evaluating VAL set ───")
        t0 = time.time()
        val_results = evaluate_tta(
            student, val_samples, device, tta_transforms,
            batch_size=args.batch_size, compute_auc=True,
        )
        val_time = time.time() - t0
        val_results["time_s"] = round(val_time, 1)
        results["val"] = val_results

        print(f"\n  VAL ACCURACY (TTA={n_views} views): {val_results['acc']*100:.2f}%")
        print(f"  VAL AUC     : {val_results.get('auc', 'N/A')}")
        print(f"  VAL EER     : {val_results.get('eer_pct', 'N/A')}%")
        print(f"  Per-view acc: {val_results['per_view_acc']}")
        print(f"  Time        : {val_time:.1f}s\n")

    if args.split in ("test", "both"):
        print("─── Evaluating TEST set ───")
        t0 = time.time()
        test_results = evaluate_tta(
            student, test_samples, device, tta_transforms,
            batch_size=args.batch_size, compute_auc=True,
        )
        test_time = time.time() - t0
        test_results["time_s"] = round(test_time, 1)
        results["test"] = test_results

        print(f"\n  TEST ACCURACY (TTA={n_views} views): {test_results['acc']*100:.2f}%")
        print(f"  TEST AUC     : {test_results.get('auc', 'N/A')}")
        print(f"  TEST EER     : {test_results.get('eer_pct', 'N/A')}%")
        print(f"  Per-view acc: {test_results['per_view_acc']}")
        print(f"  Time         : {test_time:.1f}s\n")

    # ── Save results ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "model_dir": str(model_dir),
        "tta_views": n_views,
        "tta_view_names": view_names,
        "no_tta": args.no_tta,
        "results": {},
    }
    for split_name, split_results in results.items():
        output["results"][split_name] = {
            "acc": round(split_results["acc"], 4),
            "loss": round(split_results["loss"], 4),
            "auc": split_results.get("auc"),
            "eer_pct": split_results.get("eer_pct"),
            "per_view_acc": split_results["per_view_acc"],
            "n_samples": split_results["n_samples"],
            "time_s": split_results["time_s"],
        }

    suffix = "no_tta" if args.no_tta else f"tta{n_views}"
    out_path = model_dir / f"tta_results_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved: {out_path}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
