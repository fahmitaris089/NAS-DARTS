#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src/nas:$ROOT/src/teacher:$ROOT/src/kd:${PYTHONPATH:-}"

python3 "$ROOT/src/kd/kd_train.py" \
  --teacher_arch efficientnet_v2_m \
  --teacher_weights "$ROOT/checkpoints/teacher/EfficientNetV2M_best_model.pth" \
  --student_config "$ROOT/checkpoints/student/retraining/config.json" \
  --student_weights "$ROOT/checkpoints/student/retraining/L0.05_C12_cells10_best_model.pth" \
  --data_dir "$ROOT/dataset/preprocessed" \
  --split_path "$ROOT/dataset/splits/split_info.json" \
  --temperature 20 \
  --alpha 0.5 \
  --output_dir "$ROOT/results/kd/L0.05_C12_cells10_T20_A05_regenerated" \
  "$@"

