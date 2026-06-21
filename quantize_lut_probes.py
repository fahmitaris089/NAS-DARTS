#!/usr/bin/env python3
"""
Quantize per-operator latency-LUT probes to INT8 (static, per-channel).

Why: the FP32 LUT (latency_lut_pi.json) does NOT capture the INT8 deployment
cost. Reparameterizable conv fuses to a single dense conv whose INT8 GEMM is
genuinely faster, while depthwise-separable ops (mbconv/sep/dil) gain little or
even regress under QDQ on ARM. To make hardware-aware NAS optimise the REAL
INT8 latency, we quantise every op-probe with the SAME recipe as deployment
(per-channel QDQ QInt8 + quant_pre_process) and then measure those INT8 graphs
on the Pi.

Pipeline (3 steps):
  1. (Mac) export FP32 probes  : python build_latency_lut.py --mode export --onnx-dir lut_onnx
  2. (Mac) quantize probes     : python quantize_lut_probes.py --in-dir lut_onnx --out-dir lut_onnx_int8
  3. (Pi)  measure INT8 LUT    : python build_latency_lut.py --mode measure \
                                     --onnx-dir lut_onnx_int8 --out latency_lut_pi_int8.json

Then search:
  python search.py --oplat_lambda 0.05 --latency_lut latency_lut_pi_int8.json ...

NOTE: calibration uses random tensors of each probe's input shape. This is
intentional and valid here — we measure LATENCY only (which kernels run), not
accuracy of these synthetic single-op graphs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx

try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"onnxruntime quantization unavailable: {exc}")


class RandomCalibReader(CalibrationDataReader):
    """Yield a few random tensors of the probe's input shape (latency-only)."""

    def __init__(self, input_name: str, shape: list[int], num_samples: int = 16, seed: int = 42):
        rng = np.random.default_rng(seed)
        self._data = [
            {input_name: rng.standard_normal(shape).astype(np.float32)}
            for _ in range(num_samples)
        ]
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self._data):
            return None
        item = self._data[self._idx]
        self._idx += 1
        return item


def input_name_of(onnx_path: Path) -> str:
    model = onnx.load(str(onnx_path))
    return model.graph.input[0].name


def quantize_probe(fp32_path: Path, int8_path: Path, shape: list[int]) -> str:
    """Return 'int8' on success, or 'copied_fp32' if no quantizable nodes."""
    iname = input_name_of(fp32_path)
    pre_path = int8_path.with_name(int8_path.stem + "_pre.onnx")
    try:
        quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=False)
        src = pre_path
    except Exception:
        src = fp32_path

    try:
        quantize_static(
            model_input=str(src),
            model_output=str(int8_path),
            calibration_data_reader=RandomCalibReader(iname, shape),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        status = "int8"
    except Exception as exc:
        # Ops with no quantizable nodes (none/skip_connect/pool) → keep FP32 graph.
        shutil.copyfile(fp32_path, int8_path)
        status = f"copied_fp32 ({type(exc).__name__})"
    finally:
        pre_path.unlink(missing_ok=True)
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize latency-LUT op probes to INT8")
    ap.add_argument("--in-dir", type=Path, default=Path("lut_onnx"))
    ap.add_argument("--out-dir", type=Path, default=Path("lut_onnx_int8"))
    args = ap.parse_args()

    manifest_path = args.in_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path} (run build_latency_lut.py --mode export first)")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Quantizing {len(manifest)} probes  {args.in_dir} -> {args.out_dir}")
    n_int8 = n_copied = 0
    for entry in manifest:
        fp32_path = args.in_dir / entry["file"]
        int8_path = args.out_dir / entry["file"]
        if not fp32_path.exists():
            print(f"  [missing] {entry['file']}")
            continue
        status = quantize_probe(fp32_path, int8_path, entry["shape"])
        if status == "int8":
            n_int8 += 1
        else:
            n_copied += 1
        print(f"  {entry['file']:<40} {status}")

    # Copy manifest so build_latency_lut.py --mode measure works on the int8 dir.
    shutil.copyfile(manifest_path, args.out_dir / "manifest.json")
    print(f"\nDone. int8={n_int8}  copied_fp32={n_copied}")
    print(f"Manifest copied: {args.out_dir / 'manifest.json'}")
    print("Next (on Pi):")
    print(f"  python build_latency_lut.py --mode measure --onnx-dir {args.out_dir} --out latency_lut_pi_int8.json")


if __name__ == "__main__":
    main()
