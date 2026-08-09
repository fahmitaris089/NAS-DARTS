#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$ROOT/src/deployment/benchmark_compare_onnx_pi.py" \
  --data-dir "$ROOT/dataset/preprocessed" \
  --split-path "$ROOT/dataset/splits/split_info.json" \
  --model-a "$ROOT/models/onnx_fp32/NAS_L0.05_C12_cells10_KD.onnx" \
  --label-a "Final NAS KD FP32" \
  --model-b "$ROOT/models/onnx_int8/NAS_L0.05_C12_cells10_KD_INT8.onnx" \
  --label-b "Final NAS KD INT8" \
  --threads 4 \
  --warmup 20 \
  --save-path "$ROOT/results/deployment/benchmark_raspberry_pi_regenerated.json" \
  "$@"

