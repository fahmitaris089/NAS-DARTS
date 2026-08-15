#!/usr/bin/env python3
"""Exact paired McNemar test from two per-sample prediction CSV files."""

import argparse
import csv
import json
import math
from pathlib import Path


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"]: bool(int(row["correct"])) for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", default="results/diagnostics/mcnemar_exact.json")
    args = parser.parse_args()
    baseline, candidate = read(args.baseline), read(args.candidate)
    if baseline.keys() != candidate.keys():
        raise ValueError("Prediction manifests do not contain identical sample IDs")
    b = sum(baseline[key] and not candidate[key] for key in baseline)
    c = sum(not baseline[key] and candidate[key] for key in baseline)
    n = b + c
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n) if n else 1.0
    p_value = min(1.0, 2.0 * tail)
    payload = {"discordant_baseline_only_correct": b,
               "discordant_candidate_only_correct": c,
               "discordant_total": n, "exact_two_sided_p": p_value,
               "interpretation": "paired error comparison; not a guarantee of generalization"}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
