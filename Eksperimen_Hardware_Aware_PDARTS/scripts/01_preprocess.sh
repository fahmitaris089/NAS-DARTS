#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/src/preprocessing/preprocessing.py" \
  --input-dir "$ROOT/dataset/raw" \
  --output-dir "$ROOT/dataset/preprocessed" \
  "$@"

