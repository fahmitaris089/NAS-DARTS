#!/usr/bin/env python3
"""
ONNX Runtime Per-Operator Profiler
==================================
Mengukur breakdown waktu inference per *op type* (Conv, Add, Concat, ...) untuk
satu atau beberapa model ONNX, plus latency wall-clock end-to-end.

Tujuan: membuktikan secara empiris di hardware target (mis. Raspberry Pi) ke mana
waktu inference sebenarnya habis — apakah ke banyak Conv kecil, ke Add/Concat
(memory traffic), atau ke overhead operator lain.

Cara kerja:
  1. Ukur latency end-to-end (profiling OFF, agar tidak ada overhead instrumentasi).
  2. Jalankan ulang dengan ORT profiling ON, parse trace JSON, agregasi per op_type.
  3. Tulis ringkasan ke stdout + CSV per model + (opsional) tabel perbandingan.

Hanya butuh: onnxruntime + numpy. Ringan, aman dijalankan di Pi.

Contoh:
  # Satu model, 8 thread (Pi 4/5 punya 4 core — pakai --threads 4)
  python profile_onnx_operators.py --model nas_results/retrain_mobile_v2_C4_834cls/model_benchmark.onnx --threads 4

  # Bandingkan NAS vs MobileNet (apples-to-apples: runtime & thread sama)
  python profile_onnx_operators.py \
      --model nas_results/retrain_mobile_v2_C4_834cls/model_benchmark.onnx \
      --model MobileNetV3Large/mobilenetv3_benchmark.onnx \
      --threads 4 --iters 100 --warmup 20

  # Coba execution provider XNNPACK (kalau onnxruntime di Pi dibangun dengan XNNPACK)
  python profile_onnx_operators.py --model model.onnx --provider xnnpack --threads 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime belum terpasang. Install: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)


# ─── Provider selection ──────────────────────────────────────────────────────

PROVIDER_MAP = {
    "cpu": "CPUExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


def resolve_providers(name: str) -> list[str]:
    """Pilih execution provider, fallback ke CPU jika tidak tersedia."""
    available = ort.get_available_providers()
    target = PROVIDER_MAP.get(name.lower(), "CPUExecutionProvider")
    providers: list[str] = []
    if target in available:
        providers.append(target)
        if target != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")  # fallback untuk op tak didukung
    else:
        print(f"  [WARN] Provider '{target}' tidak tersedia ({available}); pakai CPUExecutionProvider")
        providers.append("CPUExecutionProvider")
    return providers


# ─── Session helpers ─────────────────────────────────────────────────────────

def make_session(model_path: str, providers: list[str], threads: int,
                 enable_profiling: bool, profile_prefix: str | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = enable_profiling
    if enable_profiling and profile_prefix:
        so.profile_file_prefix = profile_prefix
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def build_dummy_input(session: ort.InferenceSession, batch: int, input_size: int) -> dict[str, np.ndarray]:
    """Bangun input acak sesuai signature model. Dimensi dinamis diisi default."""
    feeds: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = []
        for i, dim in enumerate(inp.shape):
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            elif i == 0:
                shape.append(batch)          # batch dimension
            elif i in (2, 3):
                shape.append(input_size)     # H, W
            else:
                shape.append(3)              # channels fallback
        dtype = np.float32 if "float" in inp.type else np.float32
        feeds[inp.name] = np.random.randn(*shape).astype(dtype)
    return feeds


# ─── Latency measurement (profiling OFF) ─────────────────────────────────────

def measure_latency(model_path: str, providers: list[str], threads: int,
                    feeds: dict[str, np.ndarray], iters: int, warmup: int) -> dict[str, float]:
    session = make_session(model_path, providers, threads, enable_profiling=False)
    out_names = [o.name for o in session.get_outputs()]

    for _ in range(warmup):
        session.run(out_names, feeds)

    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(out_names, feeds)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "p50_ms": times_ms[len(times_ms) // 2],
        "p90_ms": times_ms[int(len(times_ms) * 0.9)],
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
    }


# ─── Per-operator profiling (profiling ON) ───────────────────────────────────

def profile_operators(model_path: str, providers: list[str], threads: int,
                      feeds: dict[str, np.ndarray], iters: int, warmup: int) -> dict[str, Any]:
    tmpdir = tempfile.mkdtemp(prefix="ort_profile_")
    prefix = os.path.join(tmpdir, "prof")
    session = make_session(model_path, providers, threads,
                           enable_profiling=True, profile_prefix=prefix)
    out_names = [o.name for o in session.get_outputs()]

    for _ in range(warmup):
        session.run(out_names, feeds)
    for _ in range(iters):
        session.run(out_names, feeds)

    profile_path = session.end_profiling()

    with open(profile_path, "r") as f:
        events = json.load(f)

    # Agregasi waktu eksekusi node per op_type.
    # ORT mengeluarkan event "<node>_kernel_time" dengan args.op_name = tipe op.
    per_type_dur: dict[str, float] = defaultdict(float)   # total microseconds
    per_type_calls: dict[str, int] = defaultdict(int)     # jumlah eksekusi node

    for ev in events:
        if ev.get("cat") != "Node":
            continue
        name = ev.get("name", "")
        if not name.endswith("_kernel_time"):
            continue
        args = ev.get("args", {})
        op_type = args.get("op_name")
        if not op_type:
            continue
        per_type_dur[op_type] += float(ev.get("dur", 0.0))
        per_type_calls[op_type] += 1

    total_dur_us = sum(per_type_dur.values())

    rows = []
    for op_type in sorted(per_type_dur, key=per_type_dur.get, reverse=True):
        dur_us = per_type_dur[op_type]
        calls = per_type_calls[op_type]
        rows.append({
            "op_type": op_type,
            "calls_total": calls,
            "calls_per_infer": calls / max(iters, 1),
            "total_ms": dur_us / 1000.0,
            "avg_per_infer_ms": (dur_us / max(iters, 1)) / 1000.0,
            "avg_per_call_us": dur_us / max(calls, 1),
            "pct_time": 100.0 * dur_us / total_dur_us if total_dur_us > 0 else 0.0,
        })

    try:
        os.remove(profile_path)
        os.rmdir(tmpdir)
    except OSError:
        pass

    return {
        "rows": rows,
        "total_compute_ms_per_infer": (total_dur_us / max(iters, 1)) / 1000.0,
        "total_node_calls_per_infer": sum(per_type_calls.values()) / max(iters, 1),
    }


# ─── Reporting ───────────────────────────────────────────────────────────────

def print_report(model_path: str, latency: dict[str, float], prof: dict[str, Any]) -> None:
    name = Path(model_path).name
    print("\n" + "=" * 78)
    print(f"MODEL: {name}")
    print("=" * 78)
    print(f"  End-to-end latency : {latency['mean_ms']:.3f} ± {latency['std_ms']:.3f} ms "
          f"(p50={latency['p50_ms']:.3f}, p90={latency['p90_ms']:.3f}, "
          f"min={latency['min_ms']:.3f}, max={latency['max_ms']:.3f})")
    print(f"  Node calls / infer : {prof['total_node_calls_per_infer']:.0f}")
    print(f"  Sum kernel time    : {prof['total_compute_ms_per_infer']:.3f} ms/infer "
          f"(jumlah waktu semua node; bisa < end-to-end karena ada overhead di luar kernel)")
    print()
    print(f"  {'op_type':<22}{'calls/infer':>12}{'%time':>9}{'ms/infer':>11}{'us/call':>11}")
    print(f"  {'-'*22}{'-'*12}{'-'*9}{'-'*11}{'-'*11}")
    for r in prof["rows"]:
        print(f"  {r['op_type']:<22}{r['calls_per_infer']:>12.1f}{r['pct_time']:>8.1f}%"
              f"{r['avg_per_infer_ms']:>11.3f}{r['avg_per_call_us']:>11.1f}")


def write_csv(model_path: str, latency: dict[str, float], prof: dict[str, Any],
              output_dir: Path) -> Path:
    import csv
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (Path(model_path).stem + "_op_profile.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op_type", "calls_per_infer", "calls_total", "pct_time",
                    "ms_per_infer", "avg_per_call_us", "total_ms"])
        for r in prof["rows"]:
            w.writerow([r["op_type"], f"{r['calls_per_infer']:.2f}", r["calls_total"],
                        f"{r['pct_time']:.2f}", f"{r['avg_per_infer_ms']:.4f}",
                        f"{r['avg_per_call_us']:.2f}", f"{r['total_ms']:.3f}"])
        w.writerow([])
        w.writerow(["end_to_end_mean_ms", f"{latency['mean_ms']:.4f}"])
        w.writerow(["end_to_end_std_ms", f"{latency['std_ms']:.4f}"])
        w.writerow(["end_to_end_p50_ms", f"{latency['p50_ms']:.4f}"])
        w.writerow(["end_to_end_p90_ms", f"{latency['p90_ms']:.4f}"])
    return out_path


def print_comparison(results: list[dict[str, Any]]) -> None:
    if len(results) < 2:
        return
    print("\n" + "=" * 78)
    print("PERBANDINGAN END-TO-END")
    print("=" * 78)
    print(f"  {'model':<42}{'mean_ms':>11}{'node_calls':>13}")
    print(f"  {'-'*42}{'-'*11}{'-'*13}")
    for res in results:
        print(f"  {Path(res['model']).name:<42}{res['latency']['mean_ms']:>11.3f}"
              f"{res['prof']['total_node_calls_per_infer']:>13.0f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX Runtime per-operator profiler")
    p.add_argument("--model", action="append", required=True,
                   help="Path model ONNX (boleh diulang untuk membandingkan beberapa model)")
    p.add_argument("--provider", default="cpu",
                   choices=list(PROVIDER_MAP.keys()),
                   help="Execution provider (default: cpu). Pakai 'xnnpack' jika tersedia di Pi.")
    p.add_argument("--threads", type=int, default=4,
                   help="intra_op_num_threads (Pi 4/5 = 4 core; default 4)")
    p.add_argument("--iters", type=int, default=100, help="Jumlah iterasi terukur (default 100)")
    p.add_argument("--warmup", type=int, default=20, help="Iterasi warmup (default 20)")
    p.add_argument("--batch", type=int, default=1, help="Batch size (default 1)")
    p.add_argument("--input-size", type=int, default=224, help="Ukuran H/W input (default 224)")
    p.add_argument("--output-dir", type=Path, default=Path("profiling_results"),
                   help="Folder output CSV (default: profiling_results)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # Expand globs & validasi
    model_paths: list[str] = []
    for m in args.model:
        matches = glob.glob(m)
        if not matches:
            print(f"ERROR: model tidak ditemukan: {m}", file=sys.stderr)
            sys.exit(1)
        model_paths.extend(sorted(matches))

    providers = resolve_providers(args.provider)

    print("=" * 78)
    print("ONNX Runtime Per-Operator Profiler")
    print("=" * 78)
    print(f"  onnxruntime : {ort.__version__}")
    print(f"  providers   : {providers}")
    print(f"  threads     : {args.threads}  | iters: {args.iters} | warmup: {args.warmup}")
    print(f"  input       : batch={args.batch}, size={args.input_size}")

    results = []
    for model_path in model_paths:
        # Sesi sementara hanya untuk membangun input dengan signature yang benar
        probe = make_session(model_path, providers, args.threads, enable_profiling=False)
        feeds = build_dummy_input(probe, args.batch, args.input_size)
        del probe

        latency = measure_latency(model_path, providers, args.threads,
                                  feeds, args.iters, args.warmup)
        prof = profile_operators(model_path, providers, args.threads,
                                 feeds, args.iters, args.warmup)

        print_report(model_path, latency, prof)
        csv_path = write_csv(model_path, latency, prof, args.output_dir)
        print(f"\n  CSV tersimpan: {csv_path}")

        results.append({"model": model_path, "latency": latency, "prof": prof})

    print_comparison(results)
    print()


if __name__ == "__main__":
    main()
