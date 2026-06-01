#!/usr/bin/env bash
# run_recognition_pi.sh — Launcher palm vein recognition di Pi
#
# Setting kamera disamakan dengan capture_on_hand_detect.py yang dipakai
# saat akuisisi dataset, supaya kondisi imaging konsisten.
#
# Usage:
#   ./run_recognition_pi.sh close    # 22-25cm
#   ./run_recognition_pi.sh mid      # 27-30cm  (DEFAULT)
#   ./run_recognition_pi.sh far      # 32cm+
#   ./run_recognition_pi.sh auto     # mid + relaxed thresholds
#
# Extra flags langsung diteruskan ke prototype, contoh:
#   ./run_recognition_pi.sh mid --preview
#   ./run_recognition_pi.sh mid --decision-mode verification
#   ./run_recognition_pi.sh mid --out-dir /tmp/my_results
#
# Output tersimpan di:
#   recognition_results/accepted/  — recognized events
#   recognition_results/rejected/  — rejected events (debug)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/nas_results/retrain_run7_robust"
PROTOTYPE="${SCRIPT_DIR}/prototype_nas_recognition_onnx.py"
OUT_DIR="${SCRIPT_DIR}/recognition_results"

DIST="${1:-mid}"
shift || true  # remove first arg, rest passed through

# ── Camera + capture settings (mirrored dari capture_on_hand_detect.py) ─────
BASE_ARGS=(
    --model-dir  "$MODEL_DIR"
    --out-dir    "$OUT_DIR"
    # Camera
    --size       1920x1080
    --fps        30
    --awbgains   1.0,1.0
    --brightness -0.04
    --saturation 0
    # Capture behavior — same as dataset acquisition
    --stable-frames      12
    --burst-frames       10
    --rearm-empty-frames 8
    --cooldown-seconds   2.0
    # Quality gate — same as --quality-filter in capture_on_hand_detect.py
    --quality-filter
    # Save rejected frames for debugging
    --save-rejected
)

# ── Per-distance camera tuning ───────────────────────────────────────────────
case "$DIST" in
    close|22cm|25cm)
        # 22-25cm: closer → more light → lower exposure
        DIST_ARGS=(
            --exposure-us 6000
            --gain        1.0
            --contrast    1.5
        )
        echo "[run_recognition_pi] Distance preset: CLOSE (22-25cm)"
        ;;
    mid|27cm|30cm|"")
        # 27-30cm: training center distance — matches dataset capture settings
        DIST_ARGS=(
            --exposure-us 8000
            --gain        1.1
            --contrast    1.3
        )
        echo "[run_recognition_pi] Distance preset: MID (27-30cm)"
        ;;
    far|32cm)
        # 32cm+: farther → needs more exposure
        DIST_ARGS=(
            --exposure-us 9000
            --gain        1.2
            --contrast    1.3
        )
        echo "[run_recognition_pi] Distance preset: FAR (32cm+)"
        ;;
    auto)
        # Mid settings + relaxed detection thresholds (good for initial tuning)
        DIST_ARGS=(
            --exposure-us 8000
            --gain        1.1
            --contrast    1.3
            --relaxed
        )
        echo "[run_recognition_pi] Distance preset: AUTO (mid + relaxed)"
        ;;
    *)
        echo "Unknown preset: '$DIST'  →  use: close | mid | far | auto"
        exit 1
        ;;
esac

echo "[run_recognition_pi] Model  : $MODEL_DIR"
echo "[run_recognition_pi] Output : $OUT_DIR"
echo "[run_recognition_pi] Extra  : $*"
echo ""

exec python3 "$PROTOTYPE" \
    "${BASE_ARGS[@]}" \
    "${DIST_ARGS[@]}" \
    "$@"
