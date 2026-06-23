"""
Re-generate plots dari training_log.csv yang sudah ada.
Usage: python replot.py ResNet50
"""
import sys
from pathlib import Path

# Import fungsi plot dari train_model
sys.path.insert(0, str(Path(__file__).parent))
from train_model import plot_training_curves, plot_confusion_matrix

model_name = sys.argv[1] if len(sys.argv) > 1 else "ResNet50"
save_dir = Path("training_results") / model_name
log_path = save_dir / "training_log.csv"

if not log_path.exists():
    print(f"❌ File tidak ditemukan: {log_path}")
    sys.exit(1)

print(f"📊 Replotting training curves for {model_name}...")
plot_training_curves(log_path, save_dir)
print(f"✅ Saved: {save_dir}/training_curves.png")
