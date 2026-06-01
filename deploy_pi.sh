#!/usr/bin/env bash
# deploy_pi.sh — Copy run7 ONNX model + recognition pipeline to Raspberry Pi
#
# Usage:
#   chmod +x deploy_pi.sh
#   ./deploy_pi.sh pi@raspberrypi.local
#   ./deploy_pi.sh pi@192.168.x.x
#   ./deploy_pi.sh pi@192.168.x.x --dest /home/pi/palm-nas
#
# What gets copied:
#   - ONNX model + metadata (run7)
#   - prototype recognition script
#   - preprocessing pipeline
#   - camera capture utilities
#   - config + requirements

set -euo pipefail

PI_HOST="${1:-}"
DEST="${2:-/home/pi/palm-nas}"

if [[ -z "$PI_HOST" ]]; then
    echo "Usage: $0 <user@pi-host> [--dest /remote/path]"
    echo "  e.g. $0 pi@raspberrypi.local"
    exit 1
fi

# Handle --dest flag
if [[ "${2:-}" == "--dest" ]]; then
    DEST="${3:-/home/pi/palm-nas}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Deploying to ${PI_HOST}:${DEST}"

# Create remote directory structure
ssh "$PI_HOST" "mkdir -p ${DEST}/nas_results/retrain_run7_robust"

# Model artifacts
echo "--> Copying ONNX model..."
scp "${SCRIPT_DIR}/nas_results/retrain_run7_robust/model_benchmark.onnx" \
    "${SCRIPT_DIR}/nas_results/retrain_run7_robust/model_benchmark_metadata.json" \
    "${PI_HOST}:${DEST}/nas_results/retrain_run7_robust/"

# Python files
echo "--> Copying Python scripts..."
scp \
    "${SCRIPT_DIR}/prototype_nas_recognition_onnx.py" \
    "${SCRIPT_DIR}/preprocess_final_dataset_adaptive.py" \
    "${SCRIPT_DIR}/palm_preprocessing.py" \
    "${SCRIPT_DIR}/nas_config.py" \
    "${SCRIPT_DIR}/capture_on_hand_detect.py" \
    "${SCRIPT_DIR}/requirements_pi.txt" \
    "${PI_HOST}:${DEST}/"

echo ""
echo "==> Done. On the Pi, run:"
echo "    cd ${DEST}"
echo "    pip install -r requirements_pi.txt"
echo ""
echo "  Live recognition — close distance 22-25cm (lower exposure):"
echo "    python3 prototype_nas_recognition_onnx.py \\"
echo "        --model-dir nas_results/retrain_run7_robust \\"
echo "        --size 1920x1080 \\"
echo "        --fps 30 \\"
echo "        --exposure-us 6000 \\"
echo "        --gain 1.0 \\"
echo "        --awbgains 1.0,1.0 \\"
echo "        --brightness -0.04 \\"
echo "        --contrast 1.5 \\"
echo "        --saturation 0 \\"
echo "        --relaxed"
echo ""
echo "  Live recognition — mid distance 27-30cm (standard settings):"
echo "    python3 prototype_nas_recognition_onnx.py \\"
echo "        --model-dir nas_results/retrain_run7_robust \\"
echo "        --size 1920x1080 \\"
echo "        --fps 30 \\"
echo "        --exposure-us 8000 \\"
echo "        --gain 1.1 \\"
echo "        --awbgains 1.0,1.0 \\"
echo "        --brightness -0.04 \\"
echo "        --contrast 1.3 \\"
echo "        --saturation 0 \\"
echo "        --relaxed"
echo ""
echo "  Static test (with a raw palm image, NOT a preprocessed BMP):"
echo "    python3 prototype_nas_recognition_onnx.py \\"
echo "        --model-dir nas_results/retrain_run7_robust \\"
echo "        --test-image /path/to/raw_capture.jpg"
echo ""
echo "  Benchmark inference latency:"
echo "    python3 -c \""
echo "    import onnxruntime as ort, numpy as np, time"
echo "    sess = ort.InferenceSession('nas_results/retrain_run7_robust/model_benchmark.onnx')"
echo "    dummy = np.random.randn(1,3,224,224).astype('float32')"
echo "    [sess.run(None, {'input': dummy}) for _ in range(5)]  # warmup"
echo "    t = time.perf_counter()"
echo "    for _ in range(20): sess.run(None, {'input': dummy})"
echo "    print(f'avg latency: {(time.perf_counter()-t)/20*1000:.1f} ms')"
echo "    \""
