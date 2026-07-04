"""Debug KD model: cek PyTorch accuracy vs ONNX accuracy.

Usage:
    python3 debug_kd_model.py \
        --model-dir knowledge_distilation/kd_results/kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    ort = None

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from genotypes import dict_to_genotype
from model_eval import EvalNetwork


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def build_model(kd_cfg: dict, student_cfg: dict, model_path: Path) -> EvalNetwork:
    """Build EvalNetwork from KD config."""
    genotype = dict_to_genotype(student_cfg["genotype"])
    
    c_init = int(student_cfg.get("C_init", kd_cfg.get("student_C_init", 4)))
    num_cells = int(student_cfg.get("num_cells", kd_cfg.get("student_num_cells", 8)))
    num_classes = int(kd_cfg.get("num_classes", 834))
    dropout = float(kd_cfg.get("student_dropout", 0.3))
    
    model = EvalNetwork(
        genotype=genotype,
        C_init=c_init,
        num_cells=num_cells,
        num_classes=num_classes,
        auxiliary=False,
        dropout=dropout,
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location="cpu")
    
    print(f"\n[DEBUG] State dict keys (first 10):")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {i}: {key}")
    
    # Check if wrapped
    if any(k.startswith("module.") for k in state_dict.keys()):
        print("\n[WARN] Found 'module.' prefix → unwrapping DataParallel/DDP")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Remove auxiliary head
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("_auxiliary_head")}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"\n[WARN] Missing keys: {missing[:5]}")
    if unexpected:
        print(f"\n[WARN] Unexpected keys: {unexpected[:5]}")
    
    model.eval()
    return model


def preprocess_image(img_path: Path, input_size: int = 224) -> torch.Tensor:
    """Preprocess untuk PyTorch (matching training pipeline)."""
    img = Image.open(img_path).convert("L").resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    
    # GrayscaleToRGB
    rgb = np.stack([arr, arr, arr], axis=0)
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = (rgb - mean) / std
    
    return torch.from_numpy(rgb).float().unsqueeze(0)


def preprocess_image_onnx(img_path: Path, input_size: int = 224) -> np.ndarray:
    """Preprocess untuk ONNX."""
    img = Image.open(img_path).convert("L").resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    
    # GrayscaleToRGB
    rgb = np.stack([arr, arr, arr], axis=0)
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = (rgb - mean) / std
    
    return np.expand_dims(rgb.astype(np.float32), axis=0)


def test_pytorch_model(model: nn.Module, test_dir: Path, input_size: int, num_samples: int = 50):
    """Test PyTorch model accuracy."""
    print(f"\n{'='*60}")
    print("  PYTORCH MODEL TEST")
    print(f"{'='*60}")
    
    model.eval()
    
    # Collect test images
    test_images = []
    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    label_map = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"  Found {len(class_names)} classes")
    
    for class_name in class_names[:20]:  # First 20 classes
        class_dir = test_dir / class_name
        images = sorted(class_dir.glob("*.bmp"))
        for img_path in images[:5]:  # 5 samples per class
            test_images.append((img_path, label_map[class_name]))
            if len(test_images) >= num_samples:
                break
        if len(test_images) >= num_samples:
            break
    
    print(f"  Testing on {len(test_images)} images")
    
    correct = 0
    with torch.no_grad():
        for img_path, true_label in test_images:
            x = preprocess_image(img_path, input_size)
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1)[0])
            if pred == true_label:
                correct += 1
    
    accuracy = correct / len(test_images)
    print(f"  PyTorch Accuracy: {accuracy:.4f} ({correct}/{len(test_images)})")
    return test_images, accuracy


def test_onnx_model(onnx_path: Path, test_images, input_size: int):
    """Test ONNX model accuracy."""
    print(f"\n{'='*60}")
    print("  ONNX MODEL TEST")
    print(f"{'='*60}")
    
    if not ort:
        print("  [SKIP] onnxruntime not available")
        return None
    
    if not onnx_path.exists():
        print(f"  [SKIP] ONNX not found: {onnx_path}")
        return None
    
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    
    print(f"  Testing on {len(test_images)} images")
    
    correct = 0
    for img_path, true_label in test_images:
        x = preprocess_image_onnx(img_path, input_size)
        logits = sess.run(None, {input_name: x})[0]
        pred = int(np.argmax(logits, axis=1)[0])
        if pred == true_label:
            correct += 1
    
    accuracy = correct / len(test_images)
    print(f"  ONNX Accuracy: {accuracy:.4f} ({correct}/{len(test_images)})")
    return accuracy


def compare_outputs(model: nn.Module, onnx_path: Path, img_path: Path, input_size: int):
    """Compare PyTorch vs ONNX output untuk satu gambar."""
    print(f"\n{'='*60}")
    print("  PYTORCH vs ONNX OUTPUT COMPARISON")
    print(f"{'='*60}")
    
    model.eval()
    
    # PyTorch
    x_torch = preprocess_image(img_path, input_size)
    with torch.no_grad():
        logits_torch = model(x_torch).numpy()[0]
    
    pred_torch = int(np.argmax(logits_torch))
    top5_torch = np.argsort(logits_torch)[-5:][::-1]
    
    print(f"\n  PyTorch:")
    print(f"    Top-1 pred: {pred_torch}")
    print(f"    Top-5 pred: {top5_torch}")
    print(f"    Logits (first 10): {logits_torch[:10]}")
    
    # ONNX
    if ort and onnx_path.exists():
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        x_onnx = preprocess_image_onnx(img_path, input_size)
        logits_onnx = sess.run(None, {input_name: x_onnx})[0][0]
        
        pred_onnx = int(np.argmax(logits_onnx))
        top5_onnx = np.argsort(logits_onnx)[-5:][::-1]
        
        print(f"\n  ONNX:")
        print(f"    Top-1 pred: {pred_onnx}")
        print(f"    Top-5 pred: {top5_onnx}")
        print(f"    Logits (first 10): {logits_onnx[:10]}")
        
        # Difference
        diff = np.abs(logits_torch - logits_onnx)
        print(f"\n  Difference:")
        print(f"    Max abs diff: {diff.max():.6f}")
        print(f"    Mean abs diff: {diff.mean():.6f}")
        print(f"    Match (pred): {pred_torch == pred_onnx}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, 
                        default=Path("/Users/fahmitaris/Downloads/NAS-DARTS-TEMP/preprocessed_results/test"))
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()
    
    model_dir = args.model_dir.resolve()
    config_path = model_dir / "config.json"
    model_path = model_dir / "best_model.pth"
    onnx_path = model_dir / "model_benchmark.onnx"
    
    print(f"\n{'='*60}")
    print("  KD MODEL DEBUG")
    print(f"{'='*60}")
    print(f"  Model dir: {model_dir}")
    
    # Load config
    kd_cfg = load_json(config_path)
    if "student_config_path" in kd_cfg:
        student_cfg = load_json(PROJECT_ROOT / kd_cfg["student_config_path"])
    elif "genotype" in kd_cfg:
        student_cfg = kd_cfg
    else:
        raise KeyError("Missing genotype in config")
    
    # Build model
    model = build_model(kd_cfg, student_cfg, model_path)
    
    # Test PyTorch
    test_images, pytorch_acc = test_pytorch_model(model, args.test_dir, args.input_size, args.num_samples)
    
    # Test ONNX
    if onnx_path.exists():
        onnx_acc = test_onnx_model(onnx_path, test_images, args.input_size)
    else:
        print(f"\n[SKIP] ONNX not found: {onnx_path}")
        onnx_acc = None
    
    # Compare single image
    if test_images:
        compare_outputs(model, onnx_path, test_images[0][0], args.input_size)
    
    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  PyTorch Acc: {pytorch_acc:.4f}")
    if onnx_acc is not None:
        print(f"  ONNX Acc   : {onnx_acc:.4f}")
        print(f"  Diff       : {abs(pytorch_acc - onnx_acc):.4f}")
    print()


if __name__ == "__main__":
    main()
