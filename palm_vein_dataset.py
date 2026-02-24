"""
Palm Vein Dataset for NAS — Reuses Teacher's Split
====================================================
Loads the SAME split_info.json used by the 9 teacher models.

For search phase: further splits training set into train_search (50%)
and val_search (50%) for bilevel optimisation.

For retrain phase: uses full training set + original val/test.
"""

import json
import random
from pathlib import Path
from typing import Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

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


# ─── Transforms ─────────────────────────────────────────────────────────────

def get_transforms(split="train", input_size=INPUT_SIZE,
                   use_augmentation=True, cutout_length=0):
    """
    Get transforms consistent with teacher pipeline.

    Args:
        split:          "train", "val", or "test"
        input_size:     resize target (224)
        use_augmentation: enable augmentation for train
        cutout_length:  CutOut patch size (0 = disabled)
    """
    common_tail = [
        transforms.ToTensor(),
        GrayscaleToRGB(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if split == "train" and use_augmentation:
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
        return transforms.Compose(aug_list)
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
):
    """
    Create DataLoaders for retrain phase.

    Uses FULL training set (not split for search).
    Same val/test as teacher for fair comparison.

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
    test_samples = build_image_list(data_dir, split["test"], label_map)

    print(f"\nRetrain Dataset (same split as Teacher):")
    print(f"  Train : {len(train_samples)} images")
    print(f"  Val   : {len(val_samples)} images")
    print(f"  Test  : {len(test_samples)} images")
    print(f"  Classes: {num_classes}")
    print(f"  Augment: {'ON' if use_augmentation else 'OFF'}")
    if cutout_length > 0:
        print(f"  CutOut : {cutout_length}px")

    train_tf = get_transforms("train", input_size, use_augmentation, cutout_length)
    eval_tf = get_transforms("val", input_size)

    train_ds = PalmVeinDataset(train_samples, train_tf)
    val_ds = PalmVeinDataset(val_samples, eval_tf)
    test_ds = PalmVeinDataset(test_samples, eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
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
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
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
