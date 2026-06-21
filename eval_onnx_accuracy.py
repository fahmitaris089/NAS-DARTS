"""
Measure top-1 accuracy of an ONNX model on the palm-vein TEST split.

Reuses the project's exact preprocessing (palm_vein_dataset.get_transforms("test"))
and numeric-sorted label map so the number is directly comparable to the
PyTorch FP32 accuracy reported in test_results.json.

Usage:
  .venv/bin/python eval_onnx_accuracy.py \
      --onnx MobileNetV3Large/mobilenetv3_benchmark.onnx \
      --data_dir /Users/fahmitaris/Downloads/NAS-DARTS-TEMP/preprocessed_results \
      --split_path split_info.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from palm_vein_dataset import (
    build_image_list,
    build_label_map,
    get_transforms,
    load_split,
)


def main() -> None:
    p = argparse.ArgumentParser(description="ONNX top-1 accuracy on test split")
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--split_path", default="split_info.json", type=Path)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    split = load_split(args.split_path)
    label_map = build_label_map(split["subjects"])
    samples = build_image_list(args.data_dir, split[args.split], label_map)
    if not samples:
        raise SystemExit(f"No images found under {args.data_dir} for split '{args.split}'")

    tfm = get_transforms(split="test", input_size=args.input_size)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(args.onnx), so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    # logits is the first output for all models in this project
    out_name = sess.get_outputs()[0].name

    correct = 0
    total = len(samples)
    for img_path, label in samples:
        from PIL import Image

        x = tfm(Image.open(img_path).convert("L")).unsqueeze(0).numpy().astype(np.float32)
        logits = sess.run([out_name], {in_name: x})[0]
        pred = int(np.argmax(logits, axis=1)[0])
        correct += int(pred == label)

    acc = correct / total
    print(f"  ONNX        : {args.onnx}")
    print(f"  split       : {args.split}  ({total} samples)")
    print(f"  top-1 acc   : {acc * 100:.2f}%  ({correct}/{total})")

    out = {
        "onnx": str(args.onnx),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "num_samples": total,
        "top1_correct": correct,
        "top1_accuracy": acc,
    }
    out_path = args.onnx.with_name(args.onnx.stem + "_acc.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  saved       : {out_path}")


if __name__ == "__main__":
    main()
