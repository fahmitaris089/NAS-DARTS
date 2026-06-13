#!/bin/bash
# =============================================================
#  KD Re-run: run5_efficientNetV2M_t10_a0.5_e300
#  Student : NAS-PDARTS retrain_run5 (C_init=4, N=8)
#  Teacher : EfficientNet-V2-M
#  Config  : T=10, alpha=0.5, no mixup/cutmix, epochs=300
#
#  Cara pakai di Vast.ai:
#    chmod +x run_kd_rerun.sh
#    ./run_kd_rerun.sh
# =============================================================

set -e

PROJECT_DIR="/root/NAS-DARTS"
VENV_PATH="/venv/torch"
OUTPUT_DIR="knowledge_distilation/kd_results/run5_efficientNetV2M_t10_a0.5_e300_rerun"

echo ""
echo "=============================================="
echo "  KD Re-run: EfficientNetV2M → NAS-run5"
echo "  T=10, alpha=0.5, epochs=300, no-mix"
echo "=============================================="

cd "$PROJECT_DIR"

# ── Aktifkan venv ─────────────────────────────────────────────
source "$VENV_PATH/bin/activate"
echo "Python : $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ── Verifikasi file yang dibutuhkan ada ───────────────────────
echo ""
echo "[Check] Verifikasi file..."

if [ ! -f "nas_results/retrain_run5/best_model.pth" ]; then
    echo "ERROR: nas_results/retrain_run5/best_model.pth tidak ditemukan!"
    exit 1
fi
if [ ! -f "nas_results/retrain_run5/config.json" ]; then
    echo "ERROR: nas_results/retrain_run5/config.json tidak ditemukan!"
    exit 1
fi
if [ ! -f "knowledge_distilation/best_model.pth" ]; then
    echo "ERROR: knowledge_distilation/best_model.pth (teacher weights) tidak ditemukan!"
    exit 1
fi
if [ ! -f "split_info.json" ]; then
    echo "ERROR: split_info.json tidak ditemukan!"
    exit 1
fi

echo "  ✓ retrain_run5/best_model.pth"
echo "  ✓ retrain_run5/config.json"
echo "  ✓ knowledge_distilation/best_model.pth (teacher)"
echo "  ✓ split_info.json"

# ── Jalankan KD training ──────────────────────────────────────
echo ""
echo "[Train] Memulai KD training..."
echo "  Output : $OUTPUT_DIR"
echo ""

python3 knowledge_distilation/kd_train.py \
    --teacher_arch     efficientnet_v2_m \
    --student_config   nas_results/retrain_run5/config.json \
    --student_weights  nas_results/retrain_run5/best_model.pth \
    --temperature      10.0 \
    --alpha            0.5 \
    --epochs           300 \
    --lr               0.0003 \
    --lr_min           1e-6 \
    --weight_decay     0.02 \
    --warmup_epochs    5 \
    --scheduler        cosine \
    --drop_path        0.1 \
    --batch_size       64 \
    --seed             42 \
    --no_mix \
    --output_dir       "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "  Training selesai!"
echo "  Hasil ada di: $OUTPUT_DIR"
echo "=============================================="

# ── Print test results ────────────────────────────────────────
if [ -f "$OUTPUT_DIR/test_results.json" ]; then
    echo ""
    echo "[Results]"
    python3 -c "
import json
r = json.load(open('$OUTPUT_DIR/test_results.json'))
print(f'  Test Accuracy : {r[\"test_acc\"]*100:.2f}%')
print(f'  Best Val Acc  : {r[\"best_val_acc\"]*100:.2f}%')
print(f'  Best Epoch    : {r[\"best_epoch\"]}')
print(f'  AUC           : {r[\"test_auc\"]:.6f}')
"
fi
