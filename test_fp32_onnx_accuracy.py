#!/usr/bin/env python3
"""Test FP32 ONNX accuracy to verify it matches PyTorch."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

def preprocess(path, size=224):
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    g = np.asarray(img, dtype=np.float32) / 255.0
    rgb = np.stack([g, g, g], axis=0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = (rgb - mean) / std
    return np.expand_dims(rgb.astype(np.float32), axis=0)

split = json.load(open('split_info.json'))
label_map = {str(s): i for i, s in enumerate(sorted((str(x) for x in split['subjects']), key=int))}
data_dir = Path('/Users/fahmitaris/Downloads/NAS-DARTS-TEMP/preprocessed_results')

sess = ort.InferenceSession('Teacher/training_results/MobileNetV3Small/model_benchmark_fixed_bn.onnx',
                             providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name

correct = 0
for subj, fname in split['test'][:834]:
    path = data_dir / str(subj) / fname
    if not path.exists():
        continue
    arr = preprocess(path)
    out = sess.run(None, {inp_name: arr})[0]
    pred = int(np.argmax(out))
    if pred == label_map[str(subj)]:
        correct += 1

print(f'FP32 ONNX accuracy: {correct}/834 = {correct/834*100:.2f}%')
