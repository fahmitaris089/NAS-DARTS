#!/usr/bin/env python3
"""Select a robust C10 system strictly from deterministic validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_candidate(name: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("partition") != "validation_only":
        raise ValueError(f"{name} is not validation-only: {path}")
    if payload.get("test_loader_created") is not False:
        raise ValueError(f"{name} created a test loader: {path}")
    if payload.get("suite_complete") is not True:
        raise ValueError(f"{name} robustness suite is incomplete: {path}")
    winner = payload["winner"]
    return {
        "name": name,
        "robustness_result": str(path),
        "run_dir": payload["run_dir"],
        "config": payload["config"],
        "initial_student_sha256": payload.get("initial_student_sha256"),
        "checkpoint": winner["checkpoint"],
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "input_profile": winner["input_profile"],
        "stem_pool": winner["stem_pool"],
        "consistency_mode": winner["consistency_mode"],
        **winner["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--norm", required=True, type=Path)
    parser.add_argument("--consistency", required=True, type=Path)
    parser.add_argument("--avgpool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidates = [
        read_candidate("baseline_arcface", args.baseline),
        read_candidate("robust_norm", args.norm),
        read_candidate("robust_norm_js", args.consistency),
        read_candidate("robust_norm_avgpool_js", args.avgpool),
    ]
    controlled_hashes = {
        candidate["initial_student_sha256"] for candidate in candidates[1:]
    }
    if None in controlled_hashes or len(controlled_hashes) != 1:
        raise ValueError("E1-E3 must use one identical controlled initial-state hash")
    baseline = candidates[0]
    ordered = sorted(candidates, key=lambda item: tuple(item["selection_key"]))
    proposed = ordered[0]
    strictly_more_robust = (
        proposed["worst_case_corruption_errors"] < baseline["worst_case_corruption_errors"]
        or proposed["total_corruption_errors"] < baseline["total_corruption_errors"]
    )
    eligible = (
        proposed["name"] != "baseline_arcface"
        and proposed["clean_validation_errors"] == 0
        and strictly_more_robust
    )
    winner = proposed if eligible else baseline
    status = "method_selected" if eligible else "no_robust_system_improvement"
    payload = {
        "partition": "validation_only",
        "test_loader_created": False,
        "controlled_initial_state_sha256": next(iter(controlled_hashes)),
        "status": status,
        "acceptance_rule": (
            "834/834 clean validation and strict improvement in worst-case or "
            "total corruption errors relative to E0"
        ),
        "winner": winner,
        "proposed_best": proposed,
        "baseline": baseline,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not eligible:
        print("STOP: E1-E3 did not meet the frozen robustness acceptance rule.")


if __name__ == "__main__":
    main()
