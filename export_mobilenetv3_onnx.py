"""
Export MobileNetV3Large classifier to ONNX.

This is intended to regenerate a correct 834-class ONNX file from
`MobileNetV3Large/best_model.pth`, because the existing ONNX may have been
overwritten by a different model.

Usage:
  python3 export_mobilenetv3_onnx.py

  python3 export_mobilenetv3_onnx.py \
      --model-dir MobileNetV3Large \
      --num-classes 834
"""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "MobileNetV3Large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MobileNetV3Large to ONNX")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=834)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def build_model(weights_path: Path, num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    state_dict = torch.load(weights_path, map_location="cpu")
    if all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    weights_path = args.weights or (model_dir / "best_model.pth")
    output_path = args.output_path or (model_dir / "mobilenetv3_benchmark.onnx")
    metadata_path = args.metadata_path or (model_dir / "mobilenetv3_benchmark_metadata.json")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = build_model(weights_path, args.num_classes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    dummy = torch.randn(1, 3, args.input_size, args.input_size, dtype=torch.float32)
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # Prefer the legacy exporter for broad ONNX Runtime compatibility on ARM/Pi.
    signature = inspect.signature(torch.onnx.export)
    if "dynamo" in signature.parameters:
        export_kwargs["dynamo"] = False
    if "external_data" in signature.parameters:
        export_kwargs["external_data"] = False

    torch.onnx.export(
        model,
        dummy,
        str(tmp_output_path),
        **export_kwargs,
    )

    # Validate the file before replacing the target artifact.
    try:
        import onnx

        onnx_model = onnx.load(str(tmp_output_path))
        onnx.checker.check_model(onnx_model)
    except Exception:
        tmp_output_path.unlink(missing_ok=True)
        raise

    tmp_output_path.replace(output_path)

    metadata = {
        "exported_at": datetime.now().isoformat(),
        "model": "MobileNetV3Large",
        "weights_path": str(weights_path),
        "onnx_path": str(output_path),
        "num_classes": int(args.num_classes),
        "input_size": int(args.input_size),
        "opset": int(args.opset),
        "backend": "onnxruntime",
        "output_names": ["logits"],
        "logits_output_name": "logits",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported ONNX   : {output_path}")
    print(f"Metadata        : {metadata_path}")
    print(f"Model size      : {output_path.stat().st_size / 1e6:.3f} MB")


if __name__ == "__main__":
    main()
