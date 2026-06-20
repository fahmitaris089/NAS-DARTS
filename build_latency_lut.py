#!/usr/bin/env python3
"""
Build Device Latency LUT for Hardware-Aware NAS
================================================
Mengukur latency SETIAP operator kandidat (PRIMITIVES) di perangkat target
(Raspberry Pi), lalu menulis lookup table JSON {op_name: latency_ms} yang
dipakai search.py sebagai biaya penalti:

    L = L_CE + lambda * Σ_edge Σ_op softmax(alpha)[op] * LUT[op]

PENTING: PyTorch sering tidak tersedia / rusak di Raspberry Pi, sedangkan
onnxruntime jalan normal. Karena itu proses dipisah dua fase:

  FASE 1 — EXPORT (butuh torch; jalankan di Mac/GPU):
      python build_latency_lut.py --mode export --onnx-dir lut_onnx
    → menghasilkan file ONNX kecil per (op, config) + manifest.json

  FASE 2 — MEASURE (hanya butuh onnxruntime; jalankan DI PI):
      python build_latency_lut.py --mode measure --onnx-dir lut_onnx --out latency_lut_pi.json
    → mengukur latency tiap ONNX di Pi, menulis LUT JSON

Mesin yang punya torch + onnxruntime bisa pakai sekaligus:
      python build_latency_lut.py --mode both --onnx-dir lut_onnx --out lut.json

Konfigurasi (channel, spatial, stride) dipilih mewakili jaringan ter-deploy
(C_init=8 → channel sel {8,16,32}, spatial menyusut 56→28→14).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

# (C, H, stride) — mewakili sel normal (stride1) & reduction (stride2)
DEFAULT_CONFIGS = [
    (8, 56, 1),
    (16, 28, 1),
    (32, 14, 1),
    (16, 28, 2),
    (32, 14, 2),
]


# ─── FASE 1: EXPORT (butuh torch) ────────────────────────────────────────────

def export_op_onnx(onnx_dir: Path, configs):
    """Bangun ONNX kecil untuk tiap (op, config). Memerlukan torch + operations."""
    import torch
    import torch.nn as nn
    import inspect
    from operations import OPS, fuse_reparam_model
    from nas_config import PRIMITIVES

    # Newer PyTorch (2.x) defaults to the dynamo ONNX exporter which requires the
    # 'onnxscript' package. Force the legacy TorchScript exporter when available
    # so this works without installing onnxscript (matches export_mobilenetv3_onnx.py).
    _export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        _export_kwargs["dynamo"] = False

    class OpWrapper(nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x):
            return self.op(x)

    onnx_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for op_name in PRIMITIVES:
        for (C, H, stride) in configs:
            try:
                op = OPS[op_name](C, stride, True)  # affine=True (deploy)
            except Exception as e:
                print(f"  [skip build] {op_name} C={C} stride={stride}: {e}")
                continue
            module = OpWrapper(op).eval()
            fuse_reparam_model(module)  # RepConv multi-branch → single conv

            fname = f"{op_name}__C{C}_H{H}_s{stride}.onnx"
            fpath = onnx_dir / fname
            x = torch.randn(1, C, H, H)
            try:
                with torch.no_grad():
                    torch.onnx.export(module, x, str(fpath), opset_version=13,
                                      input_names=["input"], output_names=["output"],
                                      do_constant_folding=True, **_export_kwargs)
                manifest.append({
                    "file": fname, "op": op_name,
                    "C": C, "H": H, "stride": stride,
                    "shape": [1, C, H, H],
                })
                print(f"  exported {fname}")
            except Exception as e:
                print(f"  [skip export] {op_name} C={C} H={H} stride={stride}: {e}")

    (onnx_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {onnx_dir / 'manifest.json'} ({len(manifest)} entries)")
    print(f"Salin folder '{onnx_dir}' ke Pi, lalu jalankan --mode measure di Pi.")


# ─── FASE 2: MEASURE (hanya butuh onnxruntime) ───────────────────────────────

def measure_lut(onnx_dir: Path, out_path: Path, threads, iters, warmup, seed):
    """Ukur latency tiap ONNX di manifest, agregasi per operator → LUT JSON."""
    import onnxruntime as ort

    np.random.seed(seed)
    manifest = json.loads((onnx_dir / "manifest.json").read_text(encoding="utf-8"))
    providers = ["CPUExecutionProvider"]

    print("=" * 70)
    print("Device Latency LUT — MEASURE")
    print(f"  onnxruntime: {ort.__version__} | threads={threads} | iters={iters}")
    print(f"  entries: {len(manifest)}")
    print("=" * 70)

    per_op = {}
    for entry in manifest:
        fpath = onnx_dir / entry["file"]
        if not fpath.exists():
            print(f"  [missing] {entry['file']}")
            continue
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(fpath), sess_options=so, providers=providers)
        iname = sess.get_inputs()[0].name
        onames = [o.name for o in sess.get_outputs()]
        x = np.random.randn(*entry["shape"]).astype(np.float32)

        for _ in range(warmup):
            sess.run(onames, {iname: x})
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            sess.run(onames, {iname: x})
            times.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(times)
        per_op.setdefault(entry["op"], []).append(med)

    lut = {op: round(float(statistics.mean(v)), 5) for op, v in per_op.items()}
    for op in sorted(lut):
        print(f"  {op:<16} {lut[op]:8.4f} ms")

    meta = {
        "device_note": "measured on target device; ms = median per config, mean over configs",
        "threads": threads, "iters": iters,
        "cost": lut,
    }
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nLUT tersimpan: {out_path}")
    print(f"Pakai di search:  python search.py --oplat_lambda 0.05 --latency_lut {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Build device latency LUT for NAS")
    ap.add_argument("--mode", choices=["export", "measure", "both"], default="both",
                    help="export=buat ONNX (butuh torch); measure=ukur di Pi (butuh onnxruntime)")
    ap.add_argument("--onnx-dir", type=Path, default=Path("lut_onnx"))
    ap.add_argument("--out", type=Path, default=Path("latency_lut_pi.json"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode in ("export", "both"):
        export_op_onnx(args.onnx_dir, DEFAULT_CONFIGS)
    if args.mode in ("measure", "both"):
        measure_lut(args.onnx_dir, args.out, args.threads, args.iters, args.warmup, args.seed)


if __name__ == "__main__":
    main()
