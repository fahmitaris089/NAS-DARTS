#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model


OLD_ID = "mnasnet_a1"
NEW_ID = "mnasnet_b1_torchvision"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_tensor_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def replace_identifier(value):
    if isinstance(value, dict):
        return {key: replace_identifier(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_identifier(item) for item in value]
    if isinstance(value, str):
        return value.replace(OLD_ID, NEW_ID)
    return value


def migrate_json(path: Path) -> dict:
    before_sha256 = file_sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = replace_identifier(payload)
    if path.name == "test_results.json":
        payload["best_checkpoint"] = str(
            Path("artifacts/checkpoints/pretrained")
            / NEW_ID
            / f"seed_{payload['seed']}"
            / "best.pth"
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "before_file_sha256": before_sha256,
        "after_file_sha256": file_sha256(path),
    }


def migrate_checkpoint(path: Path) -> dict:
    before_file_sha256 = file_sha256(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    before_tensor_sha256 = state_tensor_sha256(state["model_state"])
    metadata = dict(state.get("metadata") or {})
    if metadata.get("model") == NEW_ID:
        model = build_model(NEW_ID, num_classes=int(metadata["num_classes"]))
        model.load_state_dict(state["model_state"], strict=True)
        return {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "before_file_sha256": before_file_sha256,
            "after_file_sha256": before_file_sha256,
            "model_state_sha256": before_tensor_sha256,
            "strict_load_verified": True,
            "already_migrated": True,
        }
    if metadata.get("model") != OLD_ID:
        raise RuntimeError(f"Unexpected model metadata in {path}: {metadata.get('model')!r}")
    metadata["model"] = NEW_ID
    state["metadata"] = metadata
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)

    migrated = torch.load(path, map_location="cpu", weights_only=False)
    after_tensor_sha256 = state_tensor_sha256(migrated["model_state"])
    if before_tensor_sha256 != after_tensor_sha256:
        raise RuntimeError(f"Model tensor hash changed while migrating {path}")
    model = build_model(NEW_ID, num_classes=int(metadata["num_classes"]))
    model.load_state_dict(migrated["model_state"], strict=True)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "before_file_sha256": before_file_sha256,
        "after_file_sha256": file_sha256(path),
        "model_state_sha256": after_tensor_sha256,
        "strict_load_verified": True,
    }


def main() -> None:
    old_results = PROJECT_ROOT / "results/pretrained" / OLD_ID
    new_results = PROJECT_ROOT / "results/pretrained" / NEW_ID
    old_checkpoints = PROJECT_ROOT / "artifacts/checkpoints/pretrained" / OLD_ID
    new_checkpoints = PROJECT_ROOT / "artifacts/checkpoints/pretrained" / NEW_ID
    manifest_path = PROJECT_ROOT / "results/migrations/mnasnet_a1_to_b1_torchvision.json"

    if not old_results.exists() and not old_checkpoints.exists():
        if new_results.exists() and new_checkpoints.exists() and manifest_path.exists():
            print(f"Migration already completed: {manifest_path}")
            return
        if not (new_results.exists() and new_checkpoints.exists()):
            raise FileNotFoundError("Legacy MnasNet result/checkpoint directories were not found")
    elif new_results.exists() or new_checkpoints.exists():
        raise FileExistsError("Migration destination already exists; refusing to merge directories")

    training_logs = {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in sorted((old_results if old_results.exists() else new_results).glob("seed_*/training_log.csv"))
    }
    if old_results.exists():
        shutil.move(str(old_results), str(new_results))
    if old_checkpoints.exists():
        shutil.move(str(old_checkpoints), str(new_checkpoints))

    json_records = [migrate_json(path) for path in sorted(new_results.glob("seed_*/*.json"))]
    checkpoint_records = [
        migrate_checkpoint(path) for path in sorted(new_checkpoints.glob("seed_*/*.pth"))
    ]
    for relative_path, expected_hash in training_logs.items():
        migrated_path = PROJECT_ROOT / relative_path.replace(OLD_ID, NEW_ID)
        if file_sha256(migrated_path) != expected_hash:
            raise RuntimeError(f"Training log changed during migration: {migrated_path}")

    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(new_results.glob("seed_*/test_results.json"))
    ]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Former mnasnet_a1 implementation is topology-equivalent to torchvision.mnasnet1_0 and is relabeled as MnasNet-B1.",
        "old_model_id": OLD_ID,
        "new_model_id": NEW_ID,
        "result_directory": str(new_results.relative_to(PROJECT_ROOT)),
        "checkpoint_directory": str(new_checkpoints.relative_to(PROJECT_ROOT)),
        "json_files": json_records,
        "checkpoints": checkpoint_records,
        "unchanged_training_log_sha256": {
            path.replace(OLD_ID, NEW_ID): digest for path, digest in training_logs.items()
        },
        "test_accuracy_by_seed": {
            str(row["seed"]): row["test"]["accuracy"] for row in results
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
