"""
Palm Vein Dataset & Data Split Module
=====================================
- Subject-level stratified split (no data leakage)
- Grayscale → 3-channel for pretrained CNN
- ImageNet normalization
- Configurable augmentation (on/off)
"""

import os
import json
import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path("C:\\Users\\Nanik Suciati\\Downloads\\Palm Vein Tesis\\preprocessed_results")
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
SEED             = 42


# ─── Image-Level Split ────────────────────────────────────────────────────────

def create_image_split(
    data_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=SEED,
    save_path=None,
):
    """
    Split IMAGES within each subject into train/val/test.
    Every subject appears in all splits → classification is possible.

    For 10 images per subject with 80/10/10:
      train=8, val=1, test=1 per subject.

    Returns: {"train": [(subj_id, filename), ...],
              "val":   [...],
              "test":  [...],
              "subjects": [all_subj_ids]}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    data_dir = Path(data_dir)
    # Gather all numeric subject folders
    subjects = sorted(
        [d.name for d in data_dir.iterdir()
         if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x),
    )
    assert len(subjects) > 0, f"No subject folders found in {data_dir}"

    rng = random.Random(seed)

    train_items = []
    val_items   = []
    test_items  = []

    for subj_id in subjects:
        images = sorted(f.name for f in (data_dir / subj_id).glob("*.bmp"))
        n = len(images)
        if n == 0:
            continue

        # Shuffle images within subject
        shuffled = images.copy()
        rng.shuffle(shuffled)

        # Split counts (ensure at least 1 for val and test if possible)
        n_test  = max(1, int(n * test_ratio))
        n_val   = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
            # Very few images: prioritize train
            n_train, n_val, n_test = n, 0, 0

        train_items.extend((subj_id, img) for img in shuffled[:n_train])
        val_items.extend((subj_id, img)   for img in shuffled[n_train:n_train+n_val])
        test_items.extend((subj_id, img)  for img in shuffled[n_train+n_val:])

    split = {
        "train":    train_items,
        "val":      val_items,
        "test":     test_items,
        "subjects": subjects,
    }

    n_subj = len(subjects)
    print(f"Image-level split (seed={seed}):")
    print(f"  Subjects : {n_subj}")
    print(f"  Train    : {len(train_items)} images")
    print(f"  Val      : {len(val_items)} images")
    print(f"  Test     : {len(test_items)} images")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"  Saved -> {save_path}")

    return split


def load_or_create_split(data_dir, split_path, **kwargs):
    """Load existing split or create a new one."""
    split_path = Path(split_path)
    if split_path.exists():
        with open(split_path, "r") as f:
            split = json.load(f)
        # Check if this is a new image-level split (has 'subjects' key)
        if "subjects" in split:
            n_img = len(split["train"]) + len(split["val"]) + len(split["test"])
            print(f"Loaded existing image-level split from {split_path} ({n_img} images)")
            return split
        else:
            # Old subject-level split detected, recreate as image-level
            print(f"Old subject-level split detected, recreating as image-level...")
            split_path.unlink()
    return create_image_split(data_dir, save_path=split_path, **kwargs)


# ─── Build (image_path, label) List ──────────────────────────────────────────

def build_image_list_from_items(data_dir, items, label_map):
    """Build list of (image_path, class_label) from split items.

    Args:
        data_dir: root data directory
        items: list of [subj_id, filename] pairs
        label_map: subject_id → class index
    """
    data_dir = Path(data_dir)
    samples = []
    for subj_id, filename in items:
        img_path = data_dir / subj_id / filename
        if img_path.exists():
            samples.append((img_path, label_map[subj_id]))
    return samples


def build_label_map(all_subjects):
    """
    Create consistent label mapping: subject_id -> class index.
    Covers ALL subjects so labels are consistent across splits.
    """
    sorted_subjects = sorted(all_subjects, key=lambda x: int(x))
    return {subj_id: idx for idx, subj_id in enumerate(sorted_subjects)}


# ─── Transforms ──────────────────────────────────────────────────────────────

class GrayscaleToRGB:
    """Repeat a single-channel tensor to 3 channels (picklable on Windows)."""
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


def get_transforms(split="train", input_size=224, use_augmentation=True):
    """
    Get transforms for given split.

    Args:
        split: "train", "val", or "test"
        input_size: resize target (224 for most models, 299 for InceptionV3)
        use_augmentation: if False, no augmentation even for train split

    Grayscale image → 3-channel by repeating → ImageNet normalization
    """

    # Common tail: grayscale→3ch, normalize
    common_tail = [
        transforms.ToTensor(),  # [0,255] → [0,1], shape (1, H, W)
        GrayscaleToRGB(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if split == "train" and use_augmentation:
        # Light/conservative augmentation for biometric:
        # only realistic variations (hand placement variance during scan)
        return transforms.Compose([
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
        ])
    else:
        # No augmentation for val/test, or when augmentation is disabled
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            *common_tail,
        ])


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class PalmVeinDataset(Dataset):
    """Palm Vein dataset — reads grayscale BMP, returns 3-channel tensor."""

    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


# ─── DataLoader Factory ─────────────────────────────────────────────────────

def create_dataloaders(
    data_dir="preprocessed_results",
    split_path="split_info.json",
    batch_size=32,
    input_size=224,
    num_workers=4,
    use_augmentation=True,
):
    """
    Create train/val/test DataLoaders.

    Args:
        data_dir: path to preprocessed dataset
        split_path: path to save/load split JSON
        batch_size: batch size
        input_size: image resize (224 or 299 for InceptionV3)
        num_workers: dataloader workers
        use_augmentation: enable/disable train augmentation

    Returns: (train_loader, val_loader, test_loader, info_dict)
    """
    data_dir   = Path(data_dir)
    split_path = Path(split_path)

    # 1. Create or load split
    split = load_or_create_split(data_dir, split_path)

    all_subjects = split["subjects"]
    label_map    = build_label_map(all_subjects)
    num_classes  = len(label_map)

    # 2. Build image lists
    train_samples = build_image_list_from_items(data_dir, split["train"], label_map)
    val_samples   = build_image_list_from_items(data_dir, split["val"],   label_map)
    test_samples  = build_image_list_from_items(data_dir, split["test"],  label_map)

    print(f"\nImages : train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")
    print(f"Classes: {num_classes}")
    print(f"Augment: {'ON (light)' if use_augmentation else 'OFF'}")

    # 3. Create datasets
    train_dataset = PalmVeinDataset(
        train_samples, get_transforms("train", input_size, use_augmentation))
    val_dataset = PalmVeinDataset(
        val_samples, get_transforms("val", input_size))
    test_dataset = PalmVeinDataset(
        test_samples, get_transforms("test", input_size))

    # 4. Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    info = {
        "num_classes":     num_classes,
        "label_map":       label_map,
        "split":           split,
        "train_samples":   len(train_samples),
        "val_samples":     len(val_samples),
        "test_samples":    len(test_samples),
        "use_augmentation": use_augmentation,
    }

    return train_loader, val_loader, test_loader, info


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "preprocessed_results"

    train_loader, val_loader, test_loader, info = create_dataloaders(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,
    )

    print(f"\n--- Sanity check ---")
    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch shape : {batch_x.shape}")
    print(f"Labels      : {batch_y.tolist()}")
    print(f"Pixel range : [{batch_x.min():.3f}, {batch_x.max():.3f}]")
    print(f"Num classes : {info['num_classes']}")
    print("Done.")
