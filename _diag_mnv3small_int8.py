"""Diagnostic: find a working INT8 recipe for MobileNetV3Small.

Quantizes the current FP32 ONNX with several recipes and evaluates real
top-1 accuracy on the test split, so we fix the collapse empirically.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "Teacher/training_results/MobileNetV3Small"
FP32 = MODEL_DIR / "model_benchmark.onnx"
DATA = Path("/Users/fahmitaris/Downloads/NAS-DARTS-TEMP/preprocessed_results")
SPLIT = ROOT / "split_info.json"
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def prep(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    g = np.asarray(img, np.float32) / 255.0
    rgb = (np.stack([g, g, g], 0) - MEAN) / STD
    return np.expand_dims(rgb.astype(np.float32), 0)


split = json.loads(SPLIT.read_text())
subjects = sorted((str(s) for s in split["subjects"]), key=int)
label_map = {s: i for i, s in enumerate(subjects)}

samples = []
for subj, fname in split["test"]:
    p = DATA / str(subj) / fname
    if p.exists():
        samples.append((p, str(subj)))
print(f"test samples: {len(samples)} | classes: {len(subjects)}")
cached = [(prep(p), s) for p, s in samples]

# calibration images (first 200 bmp)
calib_paths = sorted(DATA.rglob("*.bmp"))[:200]
calib_cached = [prep(p) for p in calib_paths]


def sess(path: Path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def accuracy(path: Path) -> float:
    s = sess(path)
    iname = s.get_inputs()[0].name
    c = 0
    for arr, subj in cached:
        out = s.run(None, {iname: arr})
        if int(np.argmax(out[0][0])) == label_map[subj]:
            c += 1
    return c / len(cached)


class Reader(CalibrationDataReader):
    def __init__(self, iname):
        self.iname = iname
        self.i = 0

    def get_next(self):
        if self.i >= len(calib_cached):
            return None
        x = calib_cached[self.i]
        self.i += 1
        return {self.iname: x}


iname = sess(FP32).get_inputs()[0].name
pre = MODEL_DIR / "_diag_pre.onnx"
quant_pre_process(str(FP32), str(pre), skip_symbolic_shape=False)

print(f"\nFP32 acc: {accuracy(FP32)*100:.2f}%")

recipes = [
    ("A_perchan_QInt8act", dict(per_channel=True, activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)),
    ("B_perchan_QUInt8act", dict(per_channel=True, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)),
    ("C_QUInt8act_entropy", dict(per_channel=True, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.Entropy)),
    ("D_QUInt8act_percentile", dict(per_channel=True, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.Percentile)),
]

for name, kw in recipes:
    out = MODEL_DIR / f"_diag_{name}.onnx"
    rd = Reader(iname)
    try:
        quantize_static(str(pre), str(out), rd, quant_format=QuantFormat.QDQ, **kw)
        acc = accuracy(out)
        print(f"{name:28s} acc: {acc*100:6.2f}%  size: {out.stat().st_size/1e6:.2f} MB")
    except Exception as e:
        print(f"{name:28s} FAILED: {e}")
    finally:
        if out.exists():
            out.unlink()

pre.unlink(missing_ok=True)
