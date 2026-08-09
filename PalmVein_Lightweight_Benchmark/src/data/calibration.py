from __future__ import annotations

from pathlib import Path

from src.common import save_json, sha256_file
from src.data.dataset import load_split


def create_calibration_manifest(dataset_config: dict, output_path: str | Path | None = None) -> dict:
    split = load_split(dataset_config["split_path"])
    selected = {}
    for subject, filename in sorted(split["train"], key=lambda item: (int(item[0]), item[1])):
        selected.setdefault(str(subject), filename)
    expected = int(dataset_config["expected_classes"])
    if len(selected) != expected:
        raise ValueError(f"Calibration requires one training image for every class; found {len(selected)}/{expected}")
    root = Path(dataset_config["data_dir"])
    entries = []
    for subject in sorted(selected, key=int):
        filename = selected[subject]
        image_path = root / subject / filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        entries.append(
            {
                "subject": subject,
                "filename": filename,
                "relative_path": f"{subject}/{filename}",
                "sha256": sha256_file(image_path),
                "source_split": "train",
            }
        )
    manifest = {
        "selection_rule": "lexicographically first filename per numeric subject from training split only",
        "split_sha256": sha256_file(dataset_config["split_path"]),
        "count": len(entries),
        "entries": entries,
    }
    save_json(manifest, output_path or dataset_config["calibration_manifest"])
    return manifest


def validate_calibration_manifest(dataset_config: dict, manifest: dict) -> dict:
    split = load_split(dataset_config["split_path"])
    train = {(str(subject), filename) for subject, filename in split["train"]}
    val_test = {(str(subject), filename) for name in ("val", "test") for subject, filename in split[name]}
    entries = manifest.get("entries", [])
    keys = {(str(entry["subject"]), entry["filename"]) for entry in entries}
    if len(entries) != int(dataset_config["expected_classes"]) or len(keys) != len(entries):
        raise ValueError("Calibration manifest must contain one unique image per class")
    if not keys.issubset(train):
        raise ValueError("Calibration manifest contains entries outside training split")
    if keys & val_test:
        raise ValueError("Calibration manifest overlaps validation/test split")
    if manifest.get("split_sha256") != sha256_file(dataset_config["split_path"]):
        raise ValueError("Calibration manifest split hash is stale")
    return {"valid": True, "count": len(entries), "train_only": True}
