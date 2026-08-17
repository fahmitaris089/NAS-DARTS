#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from src.common import load_json, save_json, sha256_file
from src.data import load_dataset_config, validate_calibration_manifest
from src.data.dataset import GrayscaleToRGB, label_map_from_split, load_split
from src.deployment.onnx_utils import create_session, validate_onnx_file
from palm_input_preprocessing import ApplyInputProfile, input_profile_metadata


class PalmCalibrationReader(CalibrationDataReader):
    def __init__(self, dataset_config: dict, manifest: dict, input_name: str, input_profile: str):
        self.input_name = input_name
        self.root = Path(dataset_config["data_dir"])
        size = int(dataset_config["input_size"])
        self.transform = transforms.Compose([
            transforms.Resize((size, size)), ApplyInputProfile(input_profile),
            transforms.ToTensor(), GrayscaleToRGB(),
            transforms.Normalize(dataset_config["imagenet_mean"], dataset_config["imagenet_std"]),
        ])
        self.entries = list(manifest["entries"])
        self.index = 0

    def get_next(self):
        if self.index >= len(self.entries):
            return None
        entry = self.entries[self.index]
        self.index += 1
        with Image.open(self.root / entry["relative_path"]) as image:
            array = self.transform(image.convert("L")).unsqueeze(0).numpy().astype(np.float32)
        return {self.input_name: array}

    def rewind(self):
        self.index = 0


def evaluate_onnx(model_path: Path, dataset_config: dict, input_profile: str, batch_size: int = 64, threads: int = 4):
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import PalmVeinDataset, build_samples, build_transforms

    split = load_split(dataset_config["split_path"])
    labels = label_map_from_split(split)
    samples = build_samples(dataset_config["data_dir"], split["test"], labels)
    dataset = PalmVeinDataset(
        samples, build_transforms(
            dataset_config, {"input_profile": input_profile}, training=False,
        ),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    session = create_session(model_path, threads)
    input_name = session.get_inputs()[0].name
    correct = total = 0
    for images, targets in loader:
        logits = session.run(None, {input_name: images.numpy().astype(np.float32)})[0]
        predictions = np.argmax(logits, axis=1)
        correct += int(np.sum(predictions == targets.numpy()))
        total += int(targets.numel())
    return {"accuracy": correct / total, "correct": correct, "samples": total}


def main():
    parser = argparse.ArgumentParser(description="Static QDQ INT8 quantization with train-only calibration")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    args = parser.parse_args()
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    deploy = load_json("configs/deployment.json")
    dataset = load_dataset_config()
    manifest = load_json(dataset["calibration_manifest"])
    manifest_validation = validate_calibration_manifest(dataset, manifest)
    session = create_session(args.onnx, int(deploy["runtime"]["intra_op_threads"]))
    input_name = session.get_inputs()[0].name
    source_metadata = {}
    resolved_fp32 = str(args.onnx.resolve())
    for metadata_path in (PROJECT_ROOT / "results/deployment").glob("*_onnx_fp32.json"):
        candidate = load_json(metadata_path)
        if candidate.get("onnx_path") == resolved_fp32:
            source_metadata = candidate
            break
    input_profile = str(source_metadata.get("input_profile", "legacy"))
    output = args.output or PROJECT_ROOT / "artifacts/onnx_int8" / f"{args.onnx.stem}_int8_qdq.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    reader = PalmCalibrationReader(dataset, manifest, input_name, input_profile)
    quantize_static(
        str(args.onnx), str(output), reader, quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    validate_onnx_file(output)
    reader.rewind()
    sample = reader.get_next()
    int8_session = create_session(output, int(deploy["runtime"]["intra_op_threads"]))
    smoke_output = int8_session.run(None, sample)[0]
    if smoke_output.shape != (1, int(dataset["expected_classes"])):
        raise RuntimeError(f"INT8 smoke output shape is {smoke_output.shape}")
    evaluation = None if args.skip_test_evaluation else evaluate_onnx(
        output, dataset, input_profile,
        threads=int(deploy["runtime"]["intra_op_threads"])
    )
    source_metadata = {
        key: source_metadata[key]
        for key in ("model", "protocol", "seed", "parameters", "input_profile", "input_profile_metadata")
        if key in source_metadata
    }
    payload = {
        **source_metadata,
        "fp32_onnx": str(args.onnx.resolve()), "int8_onnx": str(output.resolve()),
        "int8_sha256": sha256_file(output), "int8_bytes": output.stat().st_size,
        "format": "QDQ", "weights": "QInt8 per-channel", "activations": "QUInt8",
        "input_profile": input_profile,
        "input_profile_metadata": input_profile_metadata(input_profile),
        "calibration": manifest_validation, "test": evaluation, "smoke_output_shape": list(smoke_output.shape),
    }
    save_json(payload, PROJECT_ROOT / "results/deployment" / f"{output.stem}_quantization.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
