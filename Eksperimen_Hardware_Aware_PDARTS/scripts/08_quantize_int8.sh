#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src/nas:${PYTHONPATH:-}"

python3 "$ROOT/src/deployment/export_kd_onnx_int8.py" \
  --model-dir "$ROOT/checkpoints/student/kd" \
  --calib-dir "$ROOT/dataset/calibration" \
  --num-calib 834 \
  --threads 4 \
  --warmup 20 \
  --runs 100 \
  "$@"

cp "$ROOT/checkpoints/student/kd/model_benchmark.onnx" \
  "$ROOT/models/onnx_fp32/NAS_L0.05_C12_cells10_KD_regenerated.onnx"
cp "$ROOT/checkpoints/student/kd/model_benchmark_int8_static.onnx" \
  "$ROOT/models/onnx_int8/NAS_L0.05_C12_cells10_KD_INT8_regenerated.onnx"
