#!/bin/bash
# =============================================================================
#  Palm Vein Recognition — Train All Models Sequentially
# =============================================================================
#
#  Usage:
#    chmod +x run_all.sh
#    ./run_all.sh                  # with augmentation (default)
#    ./run_all.sh --no_augmentation  # without augmentation
#
#  Each model trains for 300 epochs (5 freeze + 295 unfreeze).
#  Results saved in training_results/<ModelName>/
# =============================================================================

set -e  # exit on error

EPOCHS="${1:-300}"  # default 300 epoch, bisa override dengan argument
EXTRA_ARGS="${@:2}"  # ambil argument setelah yang pertama

MODELS=(
    "InceptionV3"
    "ResNet50"
    "VGG16"
    "DenseNet121"
    "EfficientNetB4"
    "EfficientNetV2M"
    "MobileNetV3Large"
    "GhostNet_050"
    "ConvNeXtBase"
    "RegNetY16GF"
)

TOTAL=${#MODELS[@]}
START_TIME=$(date +%s)

echo "============================================================"
echo "  PALM VEIN RECOGNITION — SEQUENTIAL TRAINING"
echo "  Models   : ${TOTAL}"
echo "  Epochs   : ${EPOCHS}"
echo "  Args     : ${EXTRA_ARGS:-"(default: with augmentation)"}"
echo "  Started  : $(date)"
echo "============================================================"
echo ""

for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[$i]}
    IDX=$((i + 1))

    echo "────────────────────────────────────────────────────────────"
    echo "  [${IDX}/${TOTAL}]  Training: ${MODEL}"
    echo "  Time: $(date)"
    echo "────────────────────────────────────────────────────────────"

    python3 train_model.py --model "${MODEL}" --epochs ${EPOCHS} ${EXTRA_ARGS}

    echo ""
    echo "  ✓ ${MODEL} complete."
    echo ""
done

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( (TRAIN_END - START_TIME) / 60 ))

echo "============================================================"
echo "  ALL ${TOTAL} MODELS TRAINED  (${TRAIN_ELAPSED} min total)"
echo "============================================================"
echo ""
echo "Running comparison analysis..."
echo ""

python3 evaluate_all.py

END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE — ${TOTAL_ELAPSED} min total"
echo "  Results: training_results/"
echo "  Finished: $(date)"
echo "============================================================"
