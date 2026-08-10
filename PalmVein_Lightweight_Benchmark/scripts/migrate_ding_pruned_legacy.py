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

from src.models.ding_legacy import build_ding_pruned_legacy


OLD_ID = "ding_pruned"
NEW_ID = "ding_pruned_legacy_parameter_matched_v1"


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
    before = file_sha256(path)
    payload = replace_identifier(json.loads(path.read_text(encoding="utf-8")))
    payload["architecture_status"] = (
        "legacy five-block parameter-matched approximation; excluded from benchmark summaries"
    )
    if path.name == "test_results.json":
        payload["best_checkpoint"] = str(
            Path("artifacts/checkpoints/scratch")
            / NEW_ID
            / f"seed_{payload['seed']}"
            / "best.pth"
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "before_file_sha256": before,
        "after_file_sha256": file_sha256(path),
    }


def migrate_checkpoint(path: Path) -> dict:
    before_file = file_sha256(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    before_tensors = state_tensor_sha256(state["model_state"])
    metadata = dict(state.get("metadata") or {})
    if metadata.get("model") != OLD_ID:
        raise RuntimeError(f"Unexpected model metadata in {path}: {metadata.get('model')!r}")
    metadata["model"] = NEW_ID
    metadata["architecture_status"] = "legacy five-block parameter-matched approximation"
    state["metadata"] = metadata
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)

    migrated = torch.load(path, map_location="cpu", weights_only=False)
    after_tensors = state_tensor_sha256(migrated["model_state"])
    if before_tensors != after_tensors:
        raise RuntimeError(f"Model tensor hash changed while migrating {path}")
    model = build_ding_pruned_legacy(num_classes=int(metadata["num_classes"]))
    model.load_state_dict(migrated["model_state"], strict=True)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "before_file_sha256": before_file,
        "after_file_sha256": file_sha256(path),
        "model_state_sha256": after_tensors,
        "strict_load_verified": True,
    }


def main() -> None:
    old_results = PROJECT_ROOT / "results/scratch" / OLD_ID
    new_results = PROJECT_ROOT / "results/legacy/scratch" / NEW_ID
    old_checkpoints = PROJECT_ROOT / "artifacts/checkpoints/scratch" / OLD_ID
    new_checkpoints = PROJECT_ROOT / "artifacts/checkpoints/legacy/scratch" / NEW_ID
    manifest_path = PROJECT_ROOT / "results/migrations/ding_pruned_to_legacy_v1.json"

    if not old_results.exists() and not old_checkpoints.exists():
        if new_results.exists() and new_checkpoints.exists() and manifest_path.exists():
            print(f"Migration already completed: {manifest_path}")
            return
        raise FileNotFoundError("Legacy Ding result/checkpoint directories were not found")
    if new_results.exists() or new_checkpoints.exists():
        raise FileExistsError("Migration destination already exists; refusing to merge directories")

    training_logs = {
        path.name + ":" + path.parent.name: file_sha256(path)
        for path in sorted(old_results.glob("seed_*/training_log.csv"))
    }
    new_results.parent.mkdir(parents=True, exist_ok=True)
    new_checkpoints.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_results), str(new_results))
    shutil.move(str(old_checkpoints), str(new_checkpoints))

    json_records = [migrate_json(path) for path in sorted(new_results.glob("seed_*/*.json"))]
    checkpoint_records = [
        migrate_checkpoint(path) for path in sorted(new_checkpoints.glob("seed_*/*.pth"))
    ]
    for path in sorted(new_results.glob("seed_*/training_log.csv")):
        key = path.name + ":" + path.parent.name
        if file_sha256(path) != training_logs[key]:
            raise RuntimeError(f"Training log changed during migration: {path}")

    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(new_results.glob("seed_*/test_results.json"))
    ]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Former DingPruned used a five-block depthwise-pointwise approximation. "
            "It is archived before the paper-constrained six-block reconstruction replaces the public ID."
        ),
        "old_model_id": OLD_ID,
        "new_model_id": NEW_ID,
        "result_directory": str(new_results.relative_to(PROJECT_ROOT)),
        "checkpoint_directory": str(new_checkpoints.relative_to(PROJECT_ROOT)),
        "json_files": json_records,
        "checkpoints": checkpoint_records,
        "unchanged_training_log_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in sorted(new_results.glob("seed_*/training_log.csv"))
        },
        "test_accuracy_by_seed": {
            str(row["seed"]): row["test"]["accuracy"] for row in results
        },
        "excluded_from_primary_summaries": True,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
