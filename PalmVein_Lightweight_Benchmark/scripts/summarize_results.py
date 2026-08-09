#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import load_json


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fp32_summary(protocol: str):
    validation = {}
    validation_path = PROJECT_ROOT / "results/model_validation/model_spec_validation.csv"
    if validation_path.exists():
        with validation_path.open("r", newline="", encoding="utf-8") as handle:
            validation = {row["model"]: row for row in csv.DictReader(handle)}
    onnx_sizes = defaultdict(list)
    for path in (PROJECT_ROOT / "results/deployment").glob("*_onnx_fp32.json"):
        payload = load_json(path)
        if payload.get("protocol") == protocol and payload.get("model"):
            onnx_sizes[payload["model"]].append(int(payload["onnx_bytes"]))
    grouped = defaultdict(list)
    for path in (PROJECT_ROOT / "results" / protocol).glob("*/seed_*/test_results.json"):
        row = load_json(path)
        grouped[row["model"]].append(row)
    output = []
    for model, rows in sorted(grouped.items()):
        accuracies = [float(row["test"]["accuracy"]) for row in rows]
        checkpoints = [Path(row["best_checkpoint"]) for row in rows]
        output.append({
            "model": model, "protocol": protocol, "seeds_completed": len(rows),
            "seeds": " ".join(str(row["seed"]) for row in sorted(rows, key=lambda item: item["seed"])),
            "accuracy_mean": statistics.mean(accuracies),
            "accuracy_sample_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "parameters": rows[0]["parameters"],
            "mmacs_224": validation.get(model, {}).get("mmacs_224", ""),
            "checkpoint_bytes_mean": statistics.mean(path.stat().st_size for path in checkpoints if path.exists()),
            "onnx_fp32_bytes_mean": statistics.mean(onnx_sizes[model]) if onnx_sizes[model] else "",
        })
    return output


def int8_summary():
    rows = []
    for path in (PROJECT_ROOT / "results/deployment").glob("*_quantization.json"):
        payload = load_json(path)
        test = payload.get("test") or {}
        rows.append({
            "model": payload.get("model", "unknown"), "protocol": payload.get("protocol", "unknown"),
            "seed": payload.get("seed", ""), "accuracy": test.get("accuracy", ""),
            "int8_bytes": payload.get("int8_bytes", ""), "format": payload.get("format", ""),
            "weights": payload.get("weights", ""), "activations": payload.get("activations", ""),
            "calibration_count": (payload.get("calibration") or {}).get("count", ""),
        })
    return sorted(rows, key=lambda row: (row["protocol"], row["model"], str(row["seed"])))


def latency_summary():
    rows = []
    for path in (PROJECT_ROOT / "results/deployment").glob("*_latency.json"):
        payload = load_json(path)
        rows.append({key: payload.get(key, "") for key in [
            "model", "protocol", "seed", "machine", "platform", "raspberry_pi_5_claimable",
            "threads", "batch_size", "warmup_iterations", "benchmark_iterations", "mean_ms", "median_ms", "p95_ms",
        ]})
    return sorted(rows, key=lambda row: (str(row["protocol"]), str(row["model"]), str(row["seed"])))


def main():
    summary = PROJECT_ROOT / "results/summary"
    fp32_fields = [
        "model", "protocol", "seeds_completed", "seeds", "accuracy_mean", "accuracy_sample_std",
        "parameters", "mmacs_224", "checkpoint_bytes_mean", "onnx_fp32_bytes_mean",
    ]
    write_csv(summary / "summary_scratch_fp32.csv", fp32_summary("scratch"), fp32_fields)
    write_csv(summary / "summary_pretrained_fp32.csv", fp32_summary("pretrained"), fp32_fields)
    write_csv(summary / "summary_int8.csv", int8_summary(), [
        "model", "protocol", "seed", "accuracy", "int8_bytes", "format", "weights", "activations", "calibration_count",
    ])
    write_csv(summary / "raspberry_pi_latency.csv", latency_summary(), [
        "model", "protocol", "seed", "machine", "platform", "raspberry_pi_5_claimable", "threads", "batch_size",
        "warmup_iterations", "benchmark_iterations", "mean_ms", "median_ms", "p95_ms",
    ])
    print(json.dumps({"summary_directory": str(summary), "files": sorted(path.name for path in summary.glob("*.csv"))}, indent=2))


if __name__ == "__main__":
    main()
