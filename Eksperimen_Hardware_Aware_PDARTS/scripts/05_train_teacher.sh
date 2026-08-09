#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src/teacher:${PYTHONPATH:-}"

python3 "$ROOT/src/teacher/train_model.py" \
  --model EfficientNetV2M \
  --data_dir "$ROOT/dataset/preprocessed" \
  --output_dir "$ROOT/results/teacher/EfficientNetV2M_regenerated" \
  "$@"

