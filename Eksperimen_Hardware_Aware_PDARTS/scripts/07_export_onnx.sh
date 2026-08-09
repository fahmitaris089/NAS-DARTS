#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src/nas:${PYTHONPATH:-}"

python3 "$ROOT/src/deployment/export_kd_onnx.py" \
  --model-dir "$ROOT/checkpoints/student/kd" \
  --opset 13 \
  --output "$ROOT/models/onnx_fp32/NAS_L0.05_C12_cells10_KD_regenerated.onnx" \
  "$@"

