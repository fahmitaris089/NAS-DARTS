#!/usr/bin/env python3
"""Select the C10 ICD screening winner using validation metrics only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_candidate(name: str, directory: Path) -> dict:
    result_path = directory / "screening_results.json"
    checkpoint = directory / "best_screening.pth"
    config = directory / "config.json"
    for path in (result_path, checkpoint, config):
        if not path.exists():
            raise FileNotFoundError(path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("test_acc") is not None or result.get("test_loss") is not None:
        raise ValueError(
            f"{name} is not a validation-only screening result: {result_path}"
        )
    kd_config = result.get("kd_config", {})
    if name != "baseline_arcface" and not kd_config.get("skip_test_evaluation"):
        raise ValueError(f"{name} did not record --skip-test-evaluation")
    key = result["best_screening_key"]
    return {
        "name": name,
        "directory": str(directory),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "validation_errors": int(key["validation_errors"]),
        "validation_ce_loss": float(key["validation_ce_loss"]),
        "true_class_margin": -float(key["negative_true_class_margin"]),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(config),
        "initial_student_sha256": result.get("provenance", {}).get(
            "initial_student_sha256"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--fcd", required=True, type=Path)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidates = [
        read_candidate("baseline_arcface", args.baseline),
        read_candidate("fcd", args.fcd),
        read_candidate("icd_full", args.full),
    ]
    screening_initial_hashes = {
        candidate["initial_student_sha256"]
        for candidate in candidates
        if candidate["name"] in {"fcd", "icd_full"}
    }
    if None in screening_initial_hashes or len(screening_initial_hashes) != 1:
        raise ValueError(
            "FCD and full ICD must use exactly the same controlled initial-state hash"
        )
    for candidate in candidates:
        candidate["selection_key"] = [
            candidate["validation_errors"],
            candidate["validation_ce_loss"],
            -candidate["true_class_margin"],
        ]
    ordered = sorted(candidates, key=lambda item: tuple(item["selection_key"]))
    winner = ordered[0]
    status = "method_selected" if winner["name"] != "baseline_arcface" else "no_improvement"
    payload = {
        "partition": "validation_only",
        "test_loader_created": False,
        "selection_rule": "errors -> ordinary CE loss -> true-class margin",
        "controlled_initial_state_sha256": next(iter(screening_initial_hashes)),
        "status": status,
        "winner": winner,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status == "no_improvement":
        print("STOP: neither ICD candidate improved the frozen C10 ArcFace baseline.")


if __name__ == "__main__":
    main()
