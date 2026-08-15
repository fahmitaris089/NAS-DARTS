#!/usr/bin/env python3
"""Apply the predeclared C12 capacity gate before the fourth screening run."""

import argparse
import json
from pathlib import Path

BASELINE = {"errors": 1, "loss": 0.295620, "margin": 4.696615}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="nas_results/retrain_l020_c12_pk_ce_300e/seed_42/screening_results.json")
    parser.add_argument("--output", default="results/diagnostics/c12_capacity_gate.json")
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    key = result.get("best_screening_key")
    if key is None:
        raise ValueError("C12 result lacks best_screening_key; rerun with the updated retraining script")
    observed = {"errors": int(key["validation_errors"]),
                "loss": float(key["validation_ce_loss"]),
                "margin": -float(key["negative_true_class_margin"])}
    allowed = observed["errors"] < BASELINE["errors"] or (
        observed["errors"] == BASELINE["errors"]
        and observed["loss"] < BASELINE["loss"]
        and observed["margin"] > BASELINE["margin"]
    )
    payload = {"partition": "validation_only", "test_used": False,
               "locked_c10_baseline": BASELINE, "c12_pk_ce": observed,
               "allow_method_run": allowed}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
