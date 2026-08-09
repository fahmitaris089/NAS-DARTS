#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src/nas:${PYTHONPATH:-}"

python3 "$ROOT/src/nas/retrain.py" \
  --genotype "$ROOT/results/search/search_hwint8_l0.05/genotype_final.json" \
  --data_dir "$ROOT/dataset/preprocessed" \
  --split_path "$ROOT/dataset/splits/split_info.json" \
  --output_dir "$ROOT/results/retraining/L0.05_C12_cells10_regenerated" \
  --C_init 12 \
  --num_cells 10 \
  --stem_downsample 8 \
  "$@"

