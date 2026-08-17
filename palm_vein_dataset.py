"""
Palm Vein Dataset for NAS — Reuses Teacher's Split
====================================================
Loads the SAME split_info.json used by the 9 teacher models.

For search phase: further splits training set into train_search (50%)
and val_search (50%) for bilevel optimisation.

For retrain phase: uses full training set + original val/test.
"""

import json
import math
import random
import warnings
from pathlib import Path
from typing import Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from palm_input_preprocessing import (
    ApplyInputProfile,
    LEGACY_INPUT_PROFILE,
    validate_input_profile,
)

from nas_config import (
    DATA_DIR, SPLIT_PATH, SEED,
    IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE,
    SEARCH_CFG, RETRAIN_CFG,
)


# ─── Grayscale → 3-channel ──────────────────────────────────────────────────

class GrayscaleToRGB:
    """Repeat single-channel tensor to 3 channels (for ImageNet pretrained compat)."""
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


# ─── CutOut Augmentation ────────────────────────────────────────────────────

class Cutout:
    """Randomly mask out a square patch after normalisation."""
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones(h, w, dtype=img.dtype)
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
        x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
        mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


class RandomGammaContrast:
    """Domain-safe photometric perturbation for the strong consistency view."""

    def __init__(self, gamma=(0.65, 1.45), contrast=(0.65, 1.35)):
        self.gamma = tuple(float(value) for value in gamma)
        self.contrast = tuple(float(value) for value in contrast)

    def __call__(self, image):
        gamma = random.uniform(*self.gamma)
        contrast = random.uniform(*self.contrast)
        return TF.adjust_contrast(TF.adjust_gamma(image, gamma), contrast)


class TwoViewTransform:
    """Return independently augmented reference and robust views."""

    def __init__(self, reference_transform, robust_transform):
        self.reference_transform = reference_transform
        self.robust_transform = robust_transform

    def __call__(self, image):
        return self.reference_transform(image.copy()), self.robust_transform(image.copy())


# ─── Transforms ─────────────────────────────────────────────────────────────

def get_transforms(split="train", input_size=INPUT_SIZE,
                   use_augmentation=True, cutout_length=0,
                   augmentation_policy="v1_legacy",
                   input_profile=LEGACY_INPUT_PROFILE,
                   consistency_mode="none"):
    """
    Get transforms consistent with teacher pipeline.

    Args:
        split:          "train", "val", or "test"
        input_size:     resize target (224)
        use_augmentation: enable augmentation for train
        cutout_length:  CutOut patch size (0 = disabled)
        augmentation_policy: "v1_legacy" (with horizontal flip), "v2_multi_distance"
                             (no flip, more aggressive), "v3_no_flip_light"
                             (no flip, mild fine-tuning/KD policy), or
                             "v4_robust_light" (no flip, mild geometry with
                             stronger brightness/contrast robustness)
    """
    validate_input_profile(input_profile)
    if consistency_mode not in {"none", "js_two_view"}:
        raise ValueError("consistency_mode must be 'none' or 'js_two_view'")
    if consistency_mode != "none" and split != "train":
        raise ValueError("Two-view consistency is valid only for the training split")

    common_tail = [
        ApplyInputProfile(input_profile),
        transforms.ToTensor(),
        GrayscaleToRGB(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if split == "train" and use_augmentation:
        if augmentation_policy == "v2_multi_distance":
            # Augmentation v2: NO horizontal flip (fixes cross-hand confusion)
            # More aggressive rotation, affine, and color jitter for robustness
            aug_list = [
                transforms.Resize((input_size, input_size)),
                # NO RandomHorizontalFlip — left hand ≠ right hand!
                transforms.RandomRotation(degrees=15),  # Increased from 10
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),  # Increased from 0.05
                    scale=(0.78, 1.28),      # Wider range (was 0.95-1.05) to simulate distance variation
                ),
                transforms.ColorJitter(brightness=0.20, contrast=0.15),  # Increased from 0.15/0.1
                *common_tail,
            ]
        elif augmentation_policy == "v3_no_flip_light":
            # Augmentation v3: light fine-tuning/KD policy.
            # No horizontal flip, mild geometry only, and mild photometric jitter.
            aug_list = [
                transforms.Resize((input_size, input_size)),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.03, 0.03),
                    scale=(0.97, 1.08),
                ),
                transforms.ColorJitter(brightness=0.08, contrast=0.05),
                *common_tail,
            ]
        elif augmentation_policy == "v4_robust_light":
            # Augmentation v4: robustness-aware KD policy for outlier crops and
            # weak/over-bright vein patterns. No horizontal flip.
            aug_list = [
                transforms.Resize((input_size, input_size)),
                transforms.RandomRotation(degrees=4),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.04, 0.04),
                    scale=(0.94, 1.10),
                ),
                transforms.ColorJitter(brightness=0.18, contrast=0.12),
                *common_tail,
            ]
        else:
            # Augmentation v1 (legacy): with horizontal flip
            aug_list = [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.1),
                *common_tail,
            ]
        
        if cutout_length > 0:
            aug_list.append(Cutout(cutout_length))
        reference_transform = transforms.Compose(aug_list)
        if consistency_mode == "none":
            return reference_transform

        robust_aug = [
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.90, 1.12),
            ),
            RandomGammaContrast(gamma=(0.65, 1.45), contrast=(0.65, 1.35)),
            *common_tail,
        ]
        if cutout_length > 0:
            robust_aug.append(Cutout(cutout_length))
        return TwoViewTransform(reference_transform, transforms.Compose(robust_aug))
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            *common_tail,
        ])


# ─── Dataset ────────────────────────────────────────────────────────────────

class PalmVeinDataset(Dataset):
    """Palm Vein dataset — reads grayscale BMP, returns 3-channel tensor."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


class PKBatchSampler(Sampler):
    """
    Class-balanced sampler for metric/relation KD.

    Each batch contains P identities and K samples per identity. This creates
    genuine positive pairs inside the batch while retaining many negative pairs.
    """

    def __init__(self, samples, p_classes=16, k_samples=4, seed=SEED, drop_last=True):
        if p_classes <= 0 or k_samples <= 0:
            raise ValueError("p_classes and k_samples must be positive integers")

        self.samples = samples
        self.p_classes = int(p_classes)
        self.k_samples = int(k_samples)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.batch_size = self.p_classes * self.k_samples

        label_to_indices = {}
        for idx, (_, label) in enumerate(samples):
            label_to_indices.setdefault(int(label), []).append(idx)
        if len(label_to_indices) < self.p_classes:
            raise ValueError(
                f"PK sampler needs at least {self.p_classes} classes, "
                f"but dataset has {len(label_to_indices)}"
            )
        self.label_to_indices = label_to_indices
        self.labels = sorted(label_to_indices.keys())
        # One class must contribute enough K-sized groups to cover all of its
        # images.  For SCUT-PV (8 train images/class, K=4), this is two class
        # appearances per epoch.  Padding class slots are rotated by epoch.
        self.groups_per_label = {
            label: max(1, math.ceil(len(indices) / self.k_samples))
            for label, indices in self.label_to_indices.items()
        }
        total_slots = sum(self.groups_per_label.values())
        self.num_batches = math.ceil(total_slots / self.p_classes)
        self.epoch = 0
        self.replacement_labels = sorted(
            label for label, indices in self.label_to_indices.items()
            if len(indices) < self.k_samples
        )
        if self.replacement_labels:
            warnings.warn(
                f"PK sampler uses replacement for {len(self.replacement_labels)} classes "
                f"with fewer than K={self.k_samples} samples.", RuntimeWarning,
            )
        self.last_epoch_class_counts = {}

    def set_epoch(self, epoch):
        """Select the deterministic schedule used by the next iteration."""
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "p_classes": self.p_classes,
            "k_samples": self.k_samples,
            "num_batches": self.num_batches,
        }

    def load_state_dict(self, state):
        for key in ("seed", "p_classes", "k_samples", "num_batches"):
            if int(state[key]) != int(getattr(self, key)):
                raise ValueError(f"PK sampler state mismatch for {key}")
        self.set_epoch(int(state["epoch"]))

    def _class_schedule(self, rng):
        remaining = dict(self.groups_per_label)
        total_slots = self.num_batches * self.p_classes
        extras = total_slots - sum(remaining.values())
        # Rotate padding slots rather than permanently oversampling the same IDs.
        offset = (self.seed + self.epoch * max(1, extras)) % len(self.labels)
        for i in range(extras):
            remaining[self.labels[(offset + i) % len(self.labels)]] += 1

        tie_order = self.labels[:]
        rng.shuffle(tie_order)
        priority = {label: rank for rank, label in enumerate(tie_order)}
        batches = []
        for _ in range(self.num_batches):
            available = [label for label, count in remaining.items() if count > 0]
            if len(available) < self.p_classes:
                raise RuntimeError("Unable to construct a unique-class PK batch")
            available.sort(key=lambda label: (-remaining[label], priority[label]))
            chosen = available[:self.p_classes]
            rng.shuffle(chosen)
            for label in chosen:
                remaining[label] -= 1
            batches.append(chosen)
        return batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        class_batches = self._class_schedule(rng)
        class_counts = {label: 0 for label in self.labels}
        pools = {}
        cursors = {}
        for label, indices in self.label_to_indices.items():
            pools[label] = indices[:]
            rng.shuffle(pools[label])
            cursors[label] = 0

        for chosen_labels in class_batches:
            batch = []
            for label in chosen_labels:
                indices = self.label_to_indices[label]
                class_counts[label] += 1
                if len(indices) >= self.k_samples:
                    cursor = cursors[label]
                    if cursor + self.k_samples > len(pools[label]):
                        pools[label] = indices[:]
                        rng.shuffle(pools[label])
                        cursor = 0
                    selected = pools[label][cursor:cursor + self.k_samples]
                    cursors[label] = cursor + self.k_samples
                    batch.extend(selected)
                else:
                    batch.extend(rng.choices(indices, k=self.k_samples))
            rng.shuffle(batch)
            yield batch
        self.last_epoch_class_counts = class_counts

    def __len__(self):
        return self.num_batches


# ─── Build helpers ───────────────────────────────────────────────────────────

def build_label_map(all_subjects):
    """Subject ID → class index (sorted numerically)."""
    sorted_subjects = sorted(all_subjects, key=lambda x: int(x))
    return {subj_id: idx for idx, subj_id in enumerate(sorted_subjects)}


def build_image_list(data_dir, items, label_map):
    """Build list of (image_path, label) from split items."""
    data_dir = Path(data_dir)
    samples = []
    for subj_id, filename in items:
        img_path = data_dir / subj_id / filename
        if img_path.exists():
            samples.append((img_path, label_map[subj_id]))
    return samples


def load_split(split_path=None):
    """Load the teacher's split_info.json."""
    sp = Path(split_path or SPLIT_PATH)
    assert sp.exists(), f"Split file not found: {sp}\nRun Teacher training first."
    with open(sp, "r") as f:
        split = json.load(f)
    return split


# ─── Search DataLoaders ─────────────────────────────────────────────────────

def create_search_dataloaders(
    data_dir=None,
    split_path=None,
    batch_size=None,
    input_size=INPUT_SIZE,
    num_workers=None,
    search_train_ratio=None,
    seed=SEED,
):
    """
    Create DataLoaders for P-DARTS search phase.

    The training set is split 50/50 into:
      - train_search: for weight (w) updates
      - val_search:   for architecture (α) updates

    Original val/test sets are kept for monitoring only.

    Returns: (train_search_loader, val_search_loader, val_loader, test_loader, info)
    """
    data_dir = Path(data_dir or DATA_DIR)
    batch_size = batch_size if batch_size is not None else SEARCH_CFG["batch_size"]
    num_workers = num_workers if num_workers is not None else SEARCH_CFG["num_workers"]
    search_train_ratio = search_train_ratio if search_train_ratio is not None else SEARCH_CFG["search_train_ratio"]

    # Load teacher's split
    split = load_split(split_path)
    label_map = build_label_map(split["subjects"])
    num_classes = len(label_map)

    # Build sample lists
    train_samples = build_image_list(data_dir, split["train"], label_map)
    val_samples = build_image_list(data_dir, split["val"], label_map)
    test_samples = build_image_list(data_dir, split["test"], label_map)

    # Split training into search_train + search_val (50/50)
    rng = random.Random(seed)
    train_indices = list(range(len(train_samples)))
    rng.shuffle(train_indices)

    n_search_train = int(len(train_indices) * search_train_ratio)
    search_train_idx = train_indices[:n_search_train]
    search_val_idx = train_indices[n_search_train:]

    search_train_samples = [train_samples[i] for i in search_train_idx]
    search_val_samples = [train_samples[i] for i in search_val_idx]

    print(f"\nSearch Dataset Split (seed={seed}):")
    print(f"  Search train : {len(search_train_samples)} images (weight updates)")
    print(f"  Search val   : {len(search_val_samples)} images (alpha updates)")
    print(f"  Val (monitor): {len(val_samples)} images")
    print(f"  Test         : {len(test_samples)} images")
    print(f"  Classes      : {num_classes}")

    # Transforms — light augmentation for search (faster)
    train_tf = get_transforms("train", input_size, use_augmentation=True)
    eval_tf = get_transforms("val", input_size)

    # Datasets
    search_train_ds = PalmVeinDataset(search_train_samples, train_tf)
    search_val_ds = PalmVeinDataset(search_val_samples, eval_tf)
    val_ds = PalmVeinDataset(val_samples, eval_tf)
    test_ds = PalmVeinDataset(test_samples, eval_tf)

    # DataLoaders
    search_train_loader = DataLoader(
        search_train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    search_val_loader = DataLoader(
        search_val_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    info = {
        "num_classes": num_classes,
        "label_map": label_map,
        "search_train_size": len(search_train_samples),
        "search_val_size": len(search_val_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
    }

    return search_train_loader, search_val_loader, val_loader, test_loader, info


# ─── Retrain DataLoaders ────────────────────────────────────────────────────

def create_retrain_dataloaders(
    data_dir=None,
    split_path=None,
    batch_size=None,
    input_size=INPUT_SIZE,
    num_workers=None,
    use_augmentation=True,
    cutout_length=0,
    augmentation_policy="v1_legacy",
    sampler_type="random",
    pk_p=16,
    pk_k=4,
    seed=SEED,
    include_test=True,
    input_profile=LEGACY_INPUT_PROFILE,
    consistency_mode="none",
):
    """
    Create DataLoaders for retrain phase.

    Uses FULL training set (not split for search).
    Same val/test as teacher for fair comparison.

    Args:
        augmentation_policy: "v1_legacy" (with horizontal flip), "v2_multi_distance" (no flip),
                             "v3_no_flip_light" (no flip, mild fine-tuning/KD policy),
                             or "v4_robust_light" (no flip, robust light/crop policy)
        sampler_type: "random" for standard shuffled batches, or "pk" for P*K identity-balanced batches
        pk_p: number of identities per PK batch
        pk_k: number of samples per identity in PK batch

    Returns: (train_loader, val_loader, test_loader, info)
    """
    data_dir = Path(data_dir or DATA_DIR)
    batch_size = batch_size if batch_size is not None else RETRAIN_CFG["batch_size"]
    num_workers = num_workers if num_workers is not None else RETRAIN_CFG["num_workers"]

    split = load_split(split_path)
    label_map = build_label_map(split["subjects"])
    num_classes = len(label_map)

    train_samples = build_image_list(data_dir, split["train"], label_map)
    val_samples = build_image_list(data_dir, split["val"], label_map)
    # Screening must not touch test image files. The split metadata remains
    # available for provenance, but paths are resolved only for final testing.
    test_samples = (
        build_image_list(data_dir, split["test"], label_map) if include_test else []
    )

    print(f"\nRetrain Dataset (same split as Teacher):")
    print(f"  Train : {len(train_samples)} images")
    print(f"  Val   : {len(val_samples)} images")
    print(
        f"  Test  : {len(test_samples)} images"
        if include_test else
        f"  Test  : not-created ({len(split['test'])} entries declared in split metadata)"
    )
    print(f"  Classes: {num_classes}")
    print(f"  Augment: {'ON' if use_augmentation else 'OFF'}")
    print(f"  Aug Policy: {augmentation_policy}")
    print(f"  Train sampler: {sampler_type}")
    if sampler_type == "pk":
        print(f"  PK sampler: P={pk_p}, K={pk_k}, effective batch={pk_p * pk_k}")
    if cutout_length > 0:
        print(f"  CutOut : {cutout_length}px")

    train_tf = get_transforms(
        "train", input_size, use_augmentation, cutout_length,
        augmentation_policy, input_profile, consistency_mode,
    )
    eval_tf = get_transforms(
        "val", input_size, input_profile=input_profile,
    )

    train_ds = PalmVeinDataset(train_samples, train_tf)
    val_ds = PalmVeinDataset(val_samples, eval_tf)
    test_ds = PalmVeinDataset(test_samples, eval_tf) if include_test else None

    if sampler_type == "random":
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    elif sampler_type == "pk":
        if pk_p * pk_k != batch_size:
            raise ValueError(
                f"PK sampler requires pk_p * pk_k == batch_size; "
                f"got {pk_p} * {pk_k} = {pk_p * pk_k}, batch_size={batch_size}"
            )
        pk_sampler = PKBatchSampler(
            train_samples,
            p_classes=pk_p,
            k_samples=pk_k,
            seed=seed,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=pk_sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}. Use 'random' or 'pk'.")
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    info = {
        "num_classes": num_classes,
        "label_map": label_map,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples) if include_test else None,
        "test_declared_size": len(split["test"]),
        "test_loader_created": bool(include_test),
        "input_profile": input_profile,
        "consistency_mode": consistency_mode,
        "sampler_type": sampler_type,
        "pk_p": pk_p if sampler_type == "pk" else None,
        "pk_k": pk_k if sampler_type == "pk" else None,
        "pk_num_batches": len(train_loader) if sampler_type == "pk" else None,
        "pk_replacement_labels": pk_sampler.replacement_labels if sampler_type == "pk" else [],
    }

    return train_loader, val_loader, test_loader, info


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing search dataloaders...")
    loaders = create_search_dataloaders(num_workers=0, batch_size=4)
    train_s, val_s, val_l, test_l, info = loaders

    batch_x, batch_y = next(iter(train_s))
    print(f"  Search train batch: {batch_x.shape}, labels: {batch_y.tolist()}")
    print(f"  Pixel range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
    print(f"  Num classes: {info['num_classes']}")

    print("\nTesting retrain dataloaders...")
    train_r, val_r, test_r, info_r = create_retrain_dataloaders(num_workers=0, batch_size=4)
    batch_x, batch_y = next(iter(train_r))
    print(f"  Retrain train batch: {batch_x.shape}, labels: {batch_y.tolist()}")
    print("Done.")
