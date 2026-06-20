"""Export the retrained NAS run6+2-class model to ONNX.

This exporter targets the retrain result under
``nas_results/retrain_run6_plus2_e100`` by default, but keeps the model-dir
configurable. It loads the PyTorch checkpoint, reconstructs the EvalNetwork
from the saved retrain config and genotype, then writes:

- model_benchmark.onnx
- model_benchmark_metadata.json

Example:
    python3 export_retrain_run6_plus2_onnx.py
    python3 export_retrain_run6_plus2_onnx.py --model-dir nas_results/retrain_run6_plus2_e100
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from nas_config import INPUT_SIZE, RETRAIN_CFG


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "nas_results" / "retrain_run6_plus2_e100"


class EmbeddingExportWrapper(nn.Module):
    """Expose logits and penultimate embeddings in a single ONNX graph."""

    def __init__(self, model: EvalNetwork):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, embeddings = self.model.forward_with_embeddings(x)
        return logits, embeddings


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_repo_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def infer_subjects(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if args.subjects:
        return sorted([str(subject) for subject in args.subjects], key=int)

    split_path = resolve_repo_path(cfg.get("split_path"))
    # Fallback to default split_info.json at project root
    if split_path is None or not split_path.exists():
        default_split = PROJECT_ROOT / "split_info.json"
        if default_split.exists():
            split_path = default_split
        else:
            raise FileNotFoundError(
                "Could not infer subjects from retrain config. Pass --subjects explicitly."
            )

    split = load_json(split_path)
    return sorted([str(subject) for subject in split["subjects"]], key=int)


def build_model_bundle(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir
    config_path = args.config_path or (model_dir / "config.json")
    model_path = args.model_path or (model_dir / "best_model.pth")
    onnx_path = args.output_path or (model_dir / "model_benchmark.onnx")
    metadata_path = args.metadata_path or (model_dir / "model_benchmark_metadata.json")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Weights not found: {model_path}")

    cfg = load_json(config_path)
    genotype_path = args.genotype or resolve_repo_path(cfg.get("genotype_path"))
    if genotype_path is None:
        if "genotype" in cfg:
            genotype = dict_to_genotype(cfg["genotype"])
            genotype_source = "config.genotype"
        else:
            genotype_path = PROJECT_ROOT / "nas_results" / "search" / "genotype_final.json"
            genotype = dict_to_genotype(load_json(genotype_path))
            genotype_source = str(genotype_path)
    else:
        genotype = dict_to_genotype(load_json(genotype_path))
        genotype_source = str(genotype_path)

    subjects = infer_subjects(cfg, args)
    label_names = args.subject_names or subjects
    if len(label_names) != len(subjects):
        raise ValueError("--subject-names must have the same length as --subjects.")

    # Reconstruct stem/reduction settings from config (default-safe for old models)
    _stem_ds = int(cfg.get("stem_downsample", 2))
    _red_idx_raw = cfg.get("reduction_indices", None)
    _red_idx = None
    if _red_idx_raw:
        _red_idx = [int(x) for x in str(_red_idx_raw).split(",") if str(x).strip() != ""]

    model = EvalNetwork(
        genotype=genotype,
        C_init=int(cfg.get("C_init", RETRAIN_CFG["C_init"])),
        num_cells=int(cfg.get("num_cells", RETRAIN_CFG["num_cells"])),
        num_classes=len(subjects),
        auxiliary=False,
        dropout=float(RETRAIN_CFG["dropout"]),
        stem_downsample=_stem_ds,
        reduction_indices=_red_idx,
    )
    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = {
        key: value for key, value in state_dict.items()
        if not key.startswith("_auxiliary_head")
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Re-parameterization: collapse any RepConvBN multi-branch blocks into single
    # convs for deployment. No-op for MBConv/SepConv models (returns n=0).
    from operations import fuse_reparam_model
    _, n_fused = fuse_reparam_model(model)
    if n_fused:
        print(f"[export] Fused {n_fused} RepConvBN block(s) into single convs for inference.")

    return {
        "model": model,
        "config": cfg,
        "config_path": config_path,
        "model_path": model_path,
        "onnx_path": onnx_path,
        "metadata_path": metadata_path,
        "subjects": subjects,
        "label_names": label_names,
        "genotype_source": genotype_source,
        "embedding_dimension": int(model.classifier.in_features),
    }


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_size: int,
    opset: int,
    include_embeddings: bool,
) -> tuple[float, list[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
    export_model: torch.nn.Module = model
    output_names = ["logits"]
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    if include_embeddings:
        export_model = EmbeddingExportWrapper(model)
        output_names = ["logits", "embedding"]
        dynamic_axes["embedding"] = {0: "batch"}

    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    return output_path.stat().st_size / 1e6, output_names


def write_metadata(
    bundle: dict[str, Any],
    args: argparse.Namespace,
    size_mb: float,
    output_names: list[str],
) -> None:
    metadata = {
        "exported_at": datetime.now().isoformat(),
        "model_dir": str(args.model_dir),
        "model_path": str(bundle["model_path"]),
        "config_path": str(bundle["config_path"]),
        "onnx_path": str(bundle["onnx_path"]),
        "genotype_source": bundle["genotype_source"],
        "input_size": int(args.input_size),
        "opset": int(args.opset),
        "num_classes": len(bundle["subjects"]),
        "subjects": bundle["subjects"],
        "label_names": bundle["label_names"],
        "c_init": int(bundle["config"].get("C_init", RETRAIN_CFG["C_init"])),
        "num_cells": int(bundle["config"].get("num_cells", RETRAIN_CFG["num_cells"])),
        "model_size_mb": float(size_mb),
        "backend": "onnxruntime",
        "output_names": output_names,
        "logits_output_name": "logits",
        "embedding_output_name": "embedding" if args.include_embeddings else None,
        "embedding_dimension": bundle["embedding_dimension"] if args.include_embeddings else None,
        "includes_embeddings": bool(args.include_embeddings),
    }
    bundle["metadata_path"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export retrained NAS model to ONNX")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--genotype", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--subject-names", nargs="+", default=None)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--include-embeddings", dest="include_embeddings", action="store_true")
    parser.add_argument("--logits-only", dest="include_embeddings", action="store_false")
    parser.set_defaults(include_embeddings=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_model_bundle(args)
    size_mb, output_names = export_onnx(
        bundle["model"],
        bundle["onnx_path"],
        args.input_size,
        args.opset,
        args.include_embeddings,
    )
    write_metadata(bundle, args, size_mb, output_names)

    print(f"Exported ONNX   : {bundle['onnx_path']}")
    print(f"Metadata        : {bundle['metadata_path']}")
    print(f"Model size      : {size_mb:.3f} MB")
    print(f"Subjects        : {', '.join(bundle['subjects'])}")
    print(f"Outputs         : {', '.join(output_names)}")


if __name__ == "__main__":
    main()