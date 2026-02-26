#!/bin/bash
# =============================================================
#  Setup Environment — vast.ai RTX 5090
# =============================================================
#  Gunakan ini saat torch.cuda.is_available() = False di tmux.
#
#  Cara pakai (jalankan dari folder Student/):
#    chmod +x setup_vastai.sh
#    source setup_vastai.sh
#
#  PENTING: pakai "source" bukan "bash" supaya venv aktif
#  di sesi terminal yang sama setelah script selesai.
# =============================================================

set -e  # stop jika ada error

echo ""
echo "=============================================="
echo "  Setup P-DARTS — vast.ai RTX 5090"
echo "=============================================="

VENV_PATH="/venv/torch"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Step 1: Deactivate venv yang mungkin aktif ────────────────
echo ""
echo "[1/5] Deactivate venv lama (jika ada)..."
deactivate 2>/dev/null || true

# ── Step 2: Buat venv baru dengan Python 3.11 ─────────────────
echo ""
echo "[2/5] Membuat venv baru di $VENV_PATH..."
if [ -d "$VENV_PATH" ]; then
    echo "  Venv sudah ada, skip pembuatan."
else
    python3.11 -m venv "$VENV_PATH"
    echo "  Venv dibuat."
fi

# ── Step 3: Activate venv ─────────────────────────────────────
echo ""
echo "[3/5] Aktivasi venv..."
source "$VENV_PATH/bin/activate"
echo "  Python: $(which python3)"
echo "  Python version: $(python3 --version)"

# ── Step 4: Upgrade pip + install PyTorch nightly cu128 ───────
echo ""
echo "[4/5] Install PyTorch nightly cu128 (RTX 5090 / Blackwell)..."
pip install --upgrade pip --quiet
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ── Step 5: Install dependensi project ────────────────────────
echo ""
echo "[5/5] Install dependensi P-DARTS..."
pip install --upgrade \
    numpy \
    Pillow \
    matplotlib \
    seaborn \
    tqdm \
    thop \
    scipy \
    scikit-learn \
    --quiet

# ── Verifikasi CUDA ───────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Verifikasi CUDA"
echo "=============================================="
python3 -c "
import torch
print(f'PyTorch      : {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version : {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'GPU {i}        : {p.name} | {p.total_memory // 1024**3}GB | sm_{p.major}{p.minor}')
else:
    print('WARNING: CUDA tidak tersedia!')
"

# ── Selesai ───────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Setup selesai!"
echo "  Venv aktif: $VENV_PATH"
echo ""
echo "  Langkah selanjutnya:"
echo "    cd $PROJECT_DIR"
echo "    python search.py"
echo ""
echo "  NOTE: Setiap sesi tmux baru, aktifkan dulu:"
echo "    source $VENV_PATH/bin/activate"
echo "=============================================="
