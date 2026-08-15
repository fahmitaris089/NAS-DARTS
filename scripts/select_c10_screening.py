#!/usr/bin/env python3
"""Rank validation-only screening results using the frozen lexicographic rule."""

import argparse
import json
from pathlib import Path

BASELINE = {"errors": 1, "loss": 0.295620, "margin": 4.696615}


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = payload.get("best_screening_key")
    if not key:
        raise ValueError(f"Missing best_screening_key in {path}")
    return {
        "source": str(path),
        "epoch": payload.get("best_screening_epoch"),
        "errors": int(key["validation_errors"]),
        "loss": float(key["validation_ce_loss"]),
        "margin": -float(key["negative_true_class_margin"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/diagnostics/c10_screening_selection.json"))
    args = parser.parse_args()
    candidates = [load_result(path) for path in args.results]
    for candidate in candidates:
        candidate["selection_key"] = [candidate["errors"], candidate["loss"], -candidate["margin"]]
        candidate["beats_locked_baseline"] = (
            candidate["errors"] == 0
            or (
                candidate["errors"] == BASELINE["errors"]
                and candidate["loss"] < BASELINE["loss"]
                and candidate["margin"] > BASELINE["margin"]
            )
        )
    candidates.sort(key=lambda item: tuple(item["selection_key"]))
    payload = {
        "partition": "validation_only", "test_used_for_selection": False,
        "locked_baseline": BASELINE, "candidates": candidates,
        "winner": candidates[0] if candidates and candidates[0]["beats_locked_baseline"] else None,
        "next_action": (
            "confirm_seed_123_2026" if candidates and candidates[0]["beats_locked_baseline"]
            else "run_c12_pk_ce_if_budget_remains_or_stop_after_four_runs"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
