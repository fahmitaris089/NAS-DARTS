#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(path: str) -> Path:
    target = ROOT / path
    if not target.exists():
        raise AssertionError(f"Missing required path: {path}")
    return target


def main() -> int:
    json_files = [
        "dataset/splits/split_info.json",
        "configs/search/thesis_lambdas.json",
        "configs/retraining/final_student.json",
        "configs/kd/final_kd.json",
        "results/lut/latency_lut_pi_int8_corrected.json",
        "results/search/search_hwint8_l0.05/genotype_final.json",
        "checkpoints/student/retraining/config.json",
        "checkpoints/student/kd/config.json",
    ]
    for path in json_files:
        with require(path).open(encoding="utf-8") as handle:
            json.load(handle)

    required_artifacts = [
        "checkpoints/teacher/EfficientNetV2M_best_model.pth",
        "checkpoints/student/retraining/L0.05_C12_cells10_best_model.pth",
        "checkpoints/student/kd/L0.05_C12_cells10_T20_A05_best_model.pth",
        "models/onnx_fp32/NAS_L0.05_C12_cells10_KD.onnx",
        "models/onnx_int8/NAS_L0.05_C12_cells10_KD_INT8.onnx",
    ]
    for path in required_artifacts:
        require(path)

    with require("results/thesis_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 10:
        raise AssertionError("Thesis manifest is incomplete")
    for row in rows:
        require(row["evidence_path"])
        if row["artifact_path"]:
            require(row["artifact_path"])

    try:
        import onnxruntime as ort
    except ImportError:
        print("WARN: onnxruntime unavailable; ONNX session checks skipped")
    else:
        for path in required_artifacts[-2:]:
            session = ort.InferenceSession(
                str(ROOT / path), providers=["CPUExecutionProvider"]
            )
            if not session.get_inputs() or not session.get_outputs():
                raise AssertionError(f"Invalid ONNX I/O: {path}")

    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

