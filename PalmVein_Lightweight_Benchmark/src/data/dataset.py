from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.common import PROJECT_ROOT, resolve_project_path, sha256_file

import sys

REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from palm_input_preprocessing import ApplyInputProfile  # noqa: E402


class GrayscaleToRGB:
    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1) if tensor.shape[0] == 1 else tensor


class PalmVeinDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("L")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def load_dataset_config(path: str | Path = "configs/dataset.json") -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["config_path"] = str(config_path)
    config["data_dir"] = str(resolve_project_path(config["data_dir"]))
    config["split_path"] = str(resolve_project_path(config["split_path"]))
    config["calibration_manifest"] = str(resolve_project_path(config["calibration_manifest"]))
    return config


def load_split(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    required = {"train", "val", "test"}
    if not required.issubset(split):
        raise ValueError(f"Split must contain {sorted(required)}; got {sorted(split)}")
    return split


def label_map_from_split(split: dict[str, Any]) -> dict[str, int]:
    subjects = split.get("subjects") or sorted(
        {str(subject) for name in ("train", "val", "test") for subject, _ in split[name]},
        key=lambda value: int(value),
    )
    subjects = [str(subject) for subject in subjects]
    return {subject: index for index, subject in enumerate(sorted(subjects, key=lambda value: int(value)))}


def build_samples(data_dir: str | Path, items: list[list[str]], label_map: dict[str, int], *, require_files: bool = True):
    root = Path(data_dir)
    samples: list[tuple[Path, int]] = []
    missing: list[str] = []
    for subject, filename in items:
        path = root / str(subject) / filename
        if require_files and not path.is_file():
            missing.append(str(path))
        else:
            samples.append((path, label_map[str(subject)]))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} split images. First entries:\n{preview}")
    return samples


def build_transforms(config: dict[str, Any], protocol: dict[str, Any], training: bool):
    size = int(config["input_size"])
    input_profile = str(protocol.get("input_profile", config.get("input_profile", "legacy")))
    tail = [
        ApplyInputProfile(input_profile),
        transforms.ToTensor(),
        GrayscaleToRGB(),
        transforms.Normalize(config["imagenet_mean"], config["imagenet_std"]),
    ]
    if not training:
        return transforms.Compose([transforms.Resize((size, size)), *tail])
    aug = protocol["augmentation"]
    if aug.get("horizontal_flip", False):
        raise ValueError("Horizontal flip is prohibited: left and right hands are different identities.")
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomRotation(float(aug["rotation_degrees"])),
            transforms.RandomAffine(
                degrees=0,
                translate=(float(aug["translate"]), float(aug["translate"])),
                scale=tuple(float(value) for value in aug["scale"]),
            ),
            transforms.ColorJitter(brightness=float(aug["brightness"]), contrast=float(aug["contrast"])),
            *tail,
        ]
    )


def worker_seed(worker_id: int) -> None:
    import numpy as np
    import random
    import torch

    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(dataset_config: dict[str, Any], protocol: dict[str, Any], seed: int, *, batch_size: int | None = None, num_workers: int | None = None, include_test: bool = True):
    import torch

    split = load_split(dataset_config["split_path"])
    label_map = label_map_from_split(split)
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": int(batch_size or protocol["batch_size"]),
        "num_workers": int(protocol["num_workers"] if num_workers is None else num_workers),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_seed,
        "generator": generator,
    }
    loaders = {}
    names = ("train", "val", "test") if include_test else ("train", "val")
    for name in names:
        samples = build_samples(dataset_config["data_dir"], split[name], label_map)
        dataset = PalmVeinDataset(samples, build_transforms(dataset_config, protocol, name == "train"))
        loaders[name] = DataLoader(dataset, shuffle=name == "train", drop_last=False, **common)
    return loaders, label_map


def validate_dataset(config: dict[str, Any], *, verify_images: bool = True) -> dict[str, Any]:
    split_path = Path(config["split_path"])
    data_dir = Path(config["data_dir"])
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    split = load_split(split_path)
    expected_counts = config["expected_counts"]
    observed_counts = {name: len(split[name]) for name in ("train", "val", "test")}
    if observed_counts != expected_counts:
        raise ValueError(f"Split counts differ: expected {expected_counts}, observed {observed_counts}")
    sets = {name: {(str(subject), filename) for subject, filename in split[name]} for name in observed_counts}
    overlap = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"Split overlap detected: {overlap}")
    all_items = sets["train"] | sets["val"] | sets["test"]
    subjects = {subject for subject, _ in all_items}
    if len(all_items) != int(config["expected_total"]) or len(subjects) != int(config["expected_classes"]):
        raise ValueError(f"Expected {config['expected_total']} images/{config['expected_classes']} classes; found {len(all_items)}/{len(subjects)}")
    invalid_names = [(subject, name) for subject, name in all_items if Path(name).stem.split("_")[0] != subject]
    if invalid_names:
        raise ValueError(f"Filename/subject mismatch, first entries: {invalid_names[:10]}")
    missing = []
    if verify_images:
        for subject, filename in sorted(all_items):
            path = data_dir / subject / filename
            if not path.is_file():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} images; first entries: {missing[:10]}")
    per_split_classes = {name: len(Counter(str(subject) for subject, _ in split[name])) for name in observed_counts}
    return {
        "valid": True,
        "dataset_root": str(data_dir),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "counts": observed_counts,
        "class_counts": per_split_classes,
        "total": len(all_items),
        "classes": len(subjects),
        "overlap": overlap,
        "files_verified": verify_images,
    }
