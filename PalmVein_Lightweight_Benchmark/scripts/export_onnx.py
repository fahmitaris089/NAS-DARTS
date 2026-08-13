#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import load_json, save_json, set_seed, sha256_file
from src.deployment.onnx_utils import compare_outputs, validate_onnx_file
from src.models import build_model
from src.models.pdarts import fuse_reparam_model


def main():
    parser = argparse.ArgumentParser(description="Export a validated best checkpoint to ONNX FP32")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = state.get("metadata", {})
    name = metadata.get("model")
    protocol = metadata.get("protocol", "scratch")
    seed = int(metadata.get("seed", 42))
    if not name:
        raise SystemExit("Checkpoint is missing model metadata")
    config = load_json("configs/deployment.json")
    model = build_model(name, int(metadata.get("num_classes", 834)), pretrained=protocol == "pretrained")
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    set_seed(seed)
    sample = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        reference_output = model(sample)
    export_model = copy.deepcopy(model).eval()
    fused_blocks = fuse_reparam_model(export_model)
    with torch.inference_mode():
        fused_output = export_model(sample)
    fusion_error = (reference_output - fused_output).abs()
    fusion_passed = torch.allclose(reference_output, fused_output, atol=1e-4, rtol=1e-5)
    if name.startswith("pdarts_") and fused_blocks == 0:
        raise RuntimeError("P-DARTS export refused: no RepConvBN block was fused")
    if not fusion_passed:
        raise RuntimeError(
            f"RepConv fusion parity failed: max_abs_error={fusion_error.max().item():.3e}"
        )
    output = args.output or PROJECT_ROOT / "artifacts/onnx_fp32" / f"{protocol}_{name}_seed{seed}.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {config["input_name"]: {0: "batch"}, config["output_name"]: {0: "batch"}} if config["dynamic_batch"] else None
    torch.onnx.export(
        export_model, sample, str(output), input_names=[config["input_name"]], output_names=[config["output_name"]],
        dynamic_axes=dynamic_axes, opset_version=int(config["onnx_opset"]), do_constant_folding=True,
    )
    validate_onnx_file(output)
    with torch.inference_mode():
        torch_output = export_model(sample)
    comparison = compare_outputs(torch_output, output, sample.numpy().astype(np.float32))
    payload = {
        **metadata,
        "onnx_path": str(output.resolve()),
        "onnx_sha256": sha256_file(output),
        "onnx_bytes": output.stat().st_size,
        "opset": int(config["onnx_opset"]),
        "validation": comparison,
        "repconv_fusion": {
            "blocks_fused": fused_blocks,
            "max_abs_error": float(fusion_error.max().item()),
            "mean_abs_error": float(fusion_error.mean().item()),
            "passed": bool(fusion_passed),
        },
    }
    metadata_path = PROJECT_ROOT / "results/deployment" / f"{protocol}_{name}_seed{seed}_onnx_fp32.json"
    save_json(payload, metadata_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
