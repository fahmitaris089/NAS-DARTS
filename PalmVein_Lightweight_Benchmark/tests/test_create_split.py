from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "create_split.py"
SPEC = importlib.util.spec_from_file_location("create_split", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_dataset(root: Path, subjects: int = 3, images: int = 10) -> Path:
    for subject_index in range(1, subjects + 1):
        subject = root / str(subject_index)
        subject.mkdir(parents=True)
        for image_index in range(1, images + 1):
            (subject / f"{subject_index}_{image_index}.bmp").write_bytes(b"test")
    return root


def test_60_20_20_is_deterministic_and_stratified(tmp_path):
    data_dir = make_dataset(tmp_path / "dataset")
    kwargs = dict(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    first = MODULE.create_split(data_dir, **kwargs)
    second = MODULE.create_split(data_dir, **kwargs)
    assert first == second
    assert [len(first[name]) for name in ("train", "val", "test")] == [18, 6, 6]
    for name, expected in (("train", 6), ("val", 2), ("test", 2)):
        assert set(Counter(subject for subject, _ in first[name]).values()) == {expected}
    sets = {name: {tuple(item) for item in first[name]} for name in ("train", "val", "test")}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_invalid_ratios_are_rejected(tmp_path):
    data_dir = make_dataset(tmp_path / "dataset", subjects=1)
    with pytest.raises(ValueError, match="sum to 1.0"):
        MODULE.create_split(data_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.2, seed=42)
