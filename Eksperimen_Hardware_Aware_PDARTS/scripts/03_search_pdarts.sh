#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAMBDA="${1:-0.05}"
shift || true
export PYTHONPATH="$ROOT/src/nas:${PYTHONPATH:-}"

python3 "$ROOT/src/nas/search.py" \
  --data_dir "$ROOT/dataset/preprocessed" \
  --split_path "$ROOT/dataset/splits/split_info.json" \
  --latency_lut "$ROOT/results/lut/latency_lut_pi_int8_corrected.json" \
  --oplat_lambda "$LAMBDA" \
  --output_dir "$ROOT/results/search/search_regenerated_l${LAMBDA}" \
  "$@"

