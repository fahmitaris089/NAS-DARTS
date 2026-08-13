"""Export KD best_model.pth ke ONNX.

Membaca C_init, num_cells, genotype, dan num_classes langsung dari
config.json di folder KD — tidak bergantung pada RETRAIN_CFG default.

Contoh pemakaian:
    python3 export_kd_onnx.py \
        --model-dir knowledge_distilation/kd_results/run5_efficientNetV2M_t10_a0.5_e500

Output:
    <model-dir>/model_benchmark.onnx
    <model-dir>/model_benchmark_metadata.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from operations import fuse_reparam_model


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_reduction_indices(raw_value) -> list[int] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return [int(x) for x in raw_value]
    if isinstance(raw_value, str):
        return [int(x.strip()) for x in raw_value.split(",") if x.strip()]
    raise TypeError(f"Unsupported reduction_indices type: {type(raw_value)}")


def build_model(kd_cfg: dict, student_cfg: dict, model_path: Path) -> EvalNetwork:
    """Rebuild student EvalNetwork persis seperti saat KD training (auxiliary=False)."""
    genotype = dict_to_genotype(student_cfg["genotype"])

    # Ambil C_init & num_cells dari student config (config.json di folder retrain)
    c_init    = int(student_cfg.get("C_init",    kd_cfg.get("student_C_init", 4)))
    num_cells = int(student_cfg.get("num_cells", kd_cfg.get("student_num_cells", 8)))
    num_classes = int(kd_cfg.get("num_classes", 834))
    dropout   = float(kd_cfg.get("student_dropout", 0.3))
    stem_downsample = int(student_cfg.get("stem_downsample", 2))
    reduction_indices = parse_reduction_indices(student_cfg.get("reduction_indices"))

    print(f"  Architecture : C_init={c_init}, num_cells={num_cells}, "
          f"num_classes={num_classes}, auxiliary=False, "
          f"stem_downsample={stem_downsample}, reduction_indices={reduction_indices}")

    model = EvalNetwork(
        genotype          = genotype,
        C_init            = c_init,
        num_cells         = num_cells,
        num_classes       = num_classes,
        auxiliary         = False,   # KD selalu auxiliary=False
        dropout           = dropout,
        stem_downsample   = stem_downsample,
        reduction_indices = reduction_indices,
    )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    # Skip auxiliary head keys jika ada (backward compat)
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("_auxiliary_head")}
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fuse_with_parity_check(
    model: nn.Module,
    sample: torch.Tensor,
    atol: float,
) -> tuple[nn.Module, dict]:
    """Fuse RepConv branches and prove numerical equivalence before export."""
    model.eval()
    with torch.no_grad():
        reference = model(sample).detach().cpu()
    deploy_model = copy.deepcopy(model).eval()
    params_before = sum(p.numel() for p in deploy_model.parameters())
    _, n_fused = fuse_reparam_model(deploy_model)
    if n_fused <= 0:
        raise RuntimeError(
            "No RepConvBN block was fused. Refusing to export a supposedly "
            "deployment-ready P-DARTS model."
        )
    with torch.no_grad():
        fused = deploy_model(sample).detach().cpu()
    difference = (reference - fused).abs()
    parity = {
        "repconv_blocks_fused": int(n_fused),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "atol": float(atol),
        "passed": bool(torch.allclose(reference, fused, atol=atol, rtol=1e-5)),
        "parameters_training_graph": int(params_before),
        "parameters_deploy_graph": int(sum(p.numel() for p in deploy_model.parameters())),
    }
    if not parity["passed"]:
        raise RuntimeError(f"RepConv fusion parity failed: {parity}")
    return deploy_model, parity


def export_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: int = 224,
    opset: int = 13,
) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names    = ["input"],
        output_names   = ["logits"],
        dynamic_axes   = {"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version  = opset,
        do_constant_folding = True,
    )
    size_mb = output_path.stat().st_size / 1e6
    return size_mb


def validate_onnx(
    model: nn.Module,
    onnx_path: Path,
    sample: torch.Tensor,
    atol: float,
) -> dict:
    """Validate ONNX structure and ONNX Runtime output against fused PyTorch."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX validation requires onnx and onnxruntime; install requirements.txt."
        ) from exc

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    node_counts: dict[str, int] = {}
    for node in graph.graph.node:
        node_counts[node.op_type] = node_counts.get(node.op_type, 0) + 1

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(["logits"], {"input": sample.cpu().numpy()})[0]
    with torch.inference_mode():
        torch_output = model(sample).detach().cpu().numpy()
    difference = abs(torch_output - ort_output)
    result = {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "atol": float(atol),
        "passed": bool(torch.allclose(
            torch.from_numpy(torch_output), torch.from_numpy(ort_output),
            atol=atol, rtol=1e-4,
        )),
        "node_counts": dict(sorted(node_counts.items())),
    }
    if not result["passed"]:
        raise RuntimeError(f"ONNX Runtime parity failed: {result}")
    return result


def write_metadata(
    model_dir: Path,
    kd_cfg: dict,
    student_cfg: dict,
    onnx_path: Path,
    size_mb: float,
    input_size: int,
    opset: int,
    fusion_validation: dict,
    onnx_validation: dict,
) -> Path:
    c_init    = int(student_cfg.get("C_init",    kd_cfg.get("student_C_init", 4)))
    num_cells = int(student_cfg.get("num_cells", kd_cfg.get("student_num_cells", 8)))
    stem_downsample = int(student_cfg.get("stem_downsample", 2))
    reduction_indices = parse_reduction_indices(student_cfg.get("reduction_indices"))
    metadata = {
        "exported_at"  : datetime.now().isoformat(),
        "model_dir"    : str(model_dir),
        "model_path"   : str(model_dir / "best_model.pth"),
        "onnx_path"    : str(onnx_path),
        "input_size"   : input_size,
        "opset"        : opset,
        "num_classes"  : int(kd_cfg.get("num_classes", 834)),
        "c_init"       : c_init,
        "num_cells"    : num_cells,
        "stem_downsample": stem_downsample,
        "reduction_indices": reduction_indices,
        "auxiliary"    : False,
        "model_size_mb": round(size_mb, 4),
        "onnx_sha256": sha256_file(onnx_path),
        "backend"      : "onnxruntime",
        "deployment_graph": "RepConvBN branches fused to single Conv2d",
        "fusion_validation": fusion_validation,
        "onnx_validation": onnx_validation,
        "kd_config"    : {
            "teacher_arch": kd_cfg.get("teacher_arch"),
            "temperature" : kd_cfg.get("temperature"),
            "alpha"       : kd_cfg.get("alpha"),
            "epochs"      : kd_cfg.get("epochs"),
        },
    }
    meta_path = model_dir / "model_benchmark_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export KD model ke ONNX")
    parser.add_argument(
        "--model-dir", type=Path,
        default=PROJECT_ROOT / "knowledge_distilation/kd_results/run5_efficientNetV2M_t10_a0.5_e500",
        help="Folder KD yang berisi config.json dan best_model.pth",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--opset",      type=int, default=13)
    parser.add_argument("--fusion-atol", type=float, default=1e-4)
    parser.add_argument("--onnx-atol", type=float, default=1e-4)
    parser.add_argument("--output",     type=Path, default=None,
                        help="Path output ONNX. Default: <model-dir>/model_benchmark_fused.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir  = args.model_dir.resolve()
    kd_config_path = model_dir / "config.json"
    model_path     = model_dir / "best_model.pth"
    onnx_path      = args.output or (model_dir / "model_benchmark_fused.onnx")

    if not kd_config_path.exists():
        raise FileNotFoundError(f"KD config not found: {kd_config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Weights not found: {model_path}")

    print(f"\n{'='*55}")
    print(f"  Export KD Model → ONNX")
    print(f"{'='*55}")
    print(f"  Model dir  : {model_dir}")
    print(f"  Output     : {onnx_path}")

    kd_cfg = load_json(kd_config_path)

    # Load student config (retrain config.json) untuk dapat genotype + C_init aktual.
    # Dua skenario:
    #   1. KD dir  → config.json punya "student_config_path" menunjuk ke retrain config.
    #   2. Retrain dir → config.json sudah memuat "genotype" langsung (self-contained).
    if "student_config_path" in kd_cfg:
        student_config_path = PROJECT_ROOT / kd_cfg["student_config_path"]
        if not student_config_path.exists():
            raise FileNotFoundError(
                f"Student config not found: {student_config_path}\n"
                f"  (dari kd config: {kd_cfg['student_config_path']})"
            )
        student_cfg = load_json(student_config_path)
    elif "genotype" in kd_cfg:
        # config.json ini sudah merupakan student/retrain config itu sendiri.
        print("  [info] 'student_config_path' tidak ada — "
              "menggunakan config.json ini langsung sebagai student config.")
        student_cfg = kd_cfg
    else:
        raise KeyError(
            "config.json tidak memuat 'student_config_path' maupun 'genotype'. "
            "Tidak bisa merekonstruksi arsitektur model."
        )

    print(f"\n  Loading model...")
    training_model = build_model(kd_cfg, student_cfg, model_path)
    torch.manual_seed(42)
    sample = torch.randn(2, 3, args.input_size, args.input_size, dtype=torch.float32)
    model, fusion_validation = fuse_with_parity_check(
        training_model, sample, args.fusion_atol
    )
    print(
        f"  Fused        : {fusion_validation['repconv_blocks_fused']} RepConvBN block(s)\n"
        f"  Fusion error : max={fusion_validation['max_abs_error']:.3e}, "
        f"mean={fusion_validation['mean_abs_error']:.3e}"
    )

    print(f"\n  Exporting ONNX (opset={args.opset}, input={args.input_size}x{args.input_size})...")
    size_mb = export_onnx(model, onnx_path, args.input_size, args.opset)
    onnx_validation = validate_onnx(model, onnx_path, sample, args.onnx_atol)
    print(
        f"  ONNX parity  : max={onnx_validation['max_abs_error']:.3e}, "
        f"mean={onnx_validation['mean_abs_error']:.3e}\n"
        f"  ONNX nodes   : {onnx_validation['node_counts']}"
    )

    meta_path = write_metadata(
        model_dir, kd_cfg, student_cfg, onnx_path, size_mb, args.input_size, args.opset,
        fusion_validation, onnx_validation,
    )

    print(f"\n  ✓ ONNX exported : {onnx_path}")
    print(f"  ✓ Metadata      : {meta_path}")
    print(f"  ✓ Model size    : {size_mb:.3f} MB")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
