#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-measure}"
shift || true
export PYTHONPATH="$ROOT/src/lut:${PYTHONPATH:-}"

case "$MODE" in
  export)
    python3 "$ROOT/src/lut/build_latency_lut.py" \
      --mode export --onnx-dir "$ROOT/results/lut/probes_fp32" "$@"
    python3 "$ROOT/src/lut/quantize_lut_probes.py" \
      --in-dir "$ROOT/results/lut/probes_fp32" \
      --out-dir "$ROOT/results/lut/probes_int8"
    ;;
  measure)
    python3 "$ROOT/src/lut/build_latency_lut.py" \
      --mode measure \
      --onnx-dir "$ROOT/results/lut/probes_int8" \
      --out "$ROOT/results/lut/latency_lut_pi_int8_regenerated.json" \
      --threads 4 --warmup 20 --iters 100 "$@"
    ;;
  *)
    echo "Usage: $0 {export|measure} [additional arguments]" >&2
    exit 2
    ;;
esac
