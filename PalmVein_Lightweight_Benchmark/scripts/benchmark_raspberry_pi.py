#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import load_json, save_json, sha256_file
from src.deployment.onnx_utils import create_session, validate_onnx_file


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    parser = argparse.ArgumentParser(description="Benchmark validated ONNX on Raspberry Pi 5 / CPU")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_json("configs/deployment.json")["runtime"]
    warmup = int(args.warmup if args.warmup is not None else config["warmup_iterations"])
    iterations = int(args.iterations if args.iterations is not None else config["benchmark_iterations"])
    threads = int(args.threads if args.threads is not None else config["intra_op_threads"])
    if warmup < 1 or iterations < 2:
        raise SystemExit("warmup must be >=1 and iterations >=2")
    validate_onnx_file(args.onnx)
    session = create_session(args.onnx, threads)
    input_meta = session.get_inputs()[0]
    shape = [1 if not isinstance(value, int) else value for value in input_meta.shape]
    input_data = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
    feed = {input_meta.name: input_data}
    for _ in range(warmup):
        session.run(None, feed)
    latencies = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        session.run(None, feed)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    machine = platform.machine()
    onnx_hash = sha256_file(args.onnx)
    payload = {
        "onnx": str(args.onnx.resolve()), "onnx_sha256": onnx_hash,
        "platform": platform.platform(), "machine": machine,
        "raspberry_pi_5_claimable": machine in {"aarch64", "arm64"} and "Linux" in platform.system(),
        "threads": threads, "batch_size": shape[0], "warmup_iterations": warmup,
        "benchmark_iterations": iterations, "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies), "p95_ms": percentile(latencies, 95),
        "minimum_ms": min(latencies), "maximum_ms": max(latencies),
    }
    resolved = str(args.onnx.resolve())
    metadata_paths = [
        *(PROJECT_ROOT / "results/deployment").glob("*_onnx_fp32.json"),
        *(PROJECT_ROOT / "results/deployment").glob("*_quantization.json"),
    ]
    for metadata_path in metadata_paths:
        try:
            candidate = load_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            continue
        candidate_hash = candidate.get("onnx_sha256") or candidate.get("int8_sha256")
        if (
            candidate.get("onnx_path") == resolved
            or candidate.get("int8_onnx") == resolved
            or candidate_hash == onnx_hash
        ):
            payload.update({key: candidate[key] for key in ("model", "protocol", "seed") if key in candidate})
            payload["precision"] = "INT8" if candidate.get("int8_onnx") else "FP32"
            if (candidate.get("test") or {}).get("accuracy") is not None:
                payload["accuracy"] = candidate["test"]["accuracy"]
            break
    if payload.get("model") and payload.get("protocol") and payload.get("seed") is not None:
        test_path = (
            PROJECT_ROOT / "results" / str(payload["protocol"]) / str(payload["model"])
            / f"seed_{payload['seed']}" / "test_results.json"
        )
        if "accuracy" not in payload and test_path.exists():
            payload["accuracy"] = (load_json(test_path).get("test") or {}).get("accuracy")
    output = args.output or PROJECT_ROOT / "results/deployment" / f"{args.onnx.stem}_latency.json"
    save_json(payload, output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
