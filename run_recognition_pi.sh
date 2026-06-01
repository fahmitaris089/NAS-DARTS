#!/usr/bin/env bash
# run_recognition_pi.sh — One-line launcher for palm vein recognition on Pi
#
# Pilih setting sesuai jarak kamera ke telapak:
#
#   ./run_recognition_pi.sh close    # 22-25cm
#   ./run_recognition_pi.sh mid      # 27-30cm  (DEFAULT)
#   ./run_recognition_pi.sh far      # 32cm+
#   ./run_recognition_pi.sh auto     # mid settings + relaxed thresholds
#
# Tambahan flags langsung diteruskan ke prototype script, misal:
#   ./run_recognition_pi.sh mid --preview
#   ./run_recognition_pi.sh mid --decision-mode verification
#   ./run_recognition_pi.sh close --save-rejected

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/nas_results/retrain_run7_robust"
PROTOTYPE="${SCRIPT_DIR}/prototype_nas_recognition_onnx.py"

DIST="${1:-mid}"
shift || true  # remove first arg, rest passed through

# ── Camera base settings (common to all distances) ──────────────────────────
BASE_ARGS=(
    --model-dir "$MODEL_DIR"
    --size 1920x1080
    --fps 30
    --awbgains 1.0,1.0
    --saturation 0
    --brightness -0.04
    --stable-frames 10
    --burst-frames 5
    --rearm-empty-frames 10
    --cooldown-seconds 2.0
)

# ── Per-distance camera tuning ───────────────────────────────────────────────
case "$DIST" in
    close|22cm|25cm)
        # 22-25cm: shorter distance → more light → lower exposure
        DIST_ARGS=(
            --exposure-us 6000
            --gain 1.0
            --contrast 1.5
        )
        echo "[run_recognition_pi] Distance preset: CLOSE (22-25cm)"
        ;;
    mid|27cm|30cm|"")
        # 27-30cm: training center distance — default settings
        DIST_ARGS=(
            --exposure-us 8000
            --gain 1.1
            --contrast 1.3
        )
        echo "[run_recognition_pi] Distance preset: MID (27-30cm)"
        ;;
    far|32cm)
        # 32cm+: farther → more exposure needed
        DIST_ARGS=(
            --exposure-us 9000
            --gain 1.2
            --contrast 1.3
        )
        echo "[run_recognition_pi] Distance preset: FAR (32cm+)"
        ;;
    auto)
        # Mid settings + relaxed detection thresholds (good for quick testing)
        DIST_ARGS=(
            --exposure-us 8000
            --gain 1.1
            --contrast 1.3
            --relaxed
        )
        echo "[run_recognition_pi] Distance preset: AUTO (mid + relaxed)"
        ;;
    *)
        echo "Unknown distance preset: '$DIST'"
        echo "Use: close | mid | far | auto"
        exit 1
        ;;
esac

echo "[run_recognition_pi] Model: $MODEL_DIR"
echo "[run_recognition_pi] Extra args: $*"
echo ""

exec python3 "$PROTOTYPE" \
    "${BASE_ARGS[@]}" \
    "${DIST_ARGS[@]}" \
    "$@"
