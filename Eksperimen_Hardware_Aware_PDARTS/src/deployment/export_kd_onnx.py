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
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "nas"))

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from adaface import replace_linear_with_adaface, replace_linear_with_arcface
from operations import fuse_reparam_model


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, REPOSITORY_ROOT / path, PROJECT_ROOT / path]
    return next((candidate for candidate in candidates if candidate.exists()), REPOSITORY_ROOT / path)


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

    loss_mode = student_cfg.get("loss_mode", "ce")
    if loss_mode in {"arcface", "subcenter_arcface"}:
        replace_linear_with_arcface(
            model, num_classes=num_classes,
            m=float(student_cfg.get("arcface_margin", 0.5)),
            s=float(student_cfg.get("arcface_scale", 64.0)),
            num_subcenters=int(student_cfg.get(
                "arcface_subcenters", 2 if loss_mode == "subcenter_arcface" else 1
            )),
        )
    elif loss_mode == "adaface" or kd_cfg.get("adaface"):
        replace_linear_with_adaface(
            model, num_classes=num_classes,
            m=float(student_cfg.get("adaface_m", kd_cfg.get("adaface_m", 0.4))),
            h=float(student_cfg.get("adaface_h", kd_cfg.get("adaface_h", 0.333))),
            s=float(student_cfg.get("adaface_s", kd_cfg.get("adaface_s", 64.0))),
            t_alpha=float(student_cfg.get("adaface_t_alpha", kd_cfg.get("adaface_t_alpha", 0.01))),
        )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "student" in state_dict:
        state_dict = state_dict["student"]
    # Skip auxiliary head keys jika ada (backward compat)
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("_auxiliary_head")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    material_missing = [key for key in missing if not key.startswith("_auxiliary_head")]
    if material_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch; missing={material_missing[:10]} unexpected={unexpected[:10]}"
        )

    model.eval()
    _, n_fused = fuse_reparam_model(model)
    if n_fused:
        print(f"  Fused        : {n_fused} RepConvBN block(s)")
    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"  Params       : {n_params:.1f}K")
    return model


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


def write_metadata(
    model_dir: Path,
    kd_cfg: dict,
    student_cfg: dict,
    onnx_path: Path,
    size_mb: float,
    input_size: int,
    opset: int,
    model_path: Path,
) -> Path:
    c_init    = int(student_cfg.get("C_init",    kd_cfg.get("student_C_init", 4)))
    num_cells = int(student_cfg.get("num_cells", kd_cfg.get("student_num_cells", 8)))
    stem_downsample = int(student_cfg.get("stem_downsample", 2))
    reduction_indices = parse_reduction_indices(student_cfg.get("reduction_indices"))
    metadata = {
        "exported_at"  : datetime.now().isoformat(),
        "model_dir"    : str(model_dir),
        "model_path"   : str(model_path),
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
        "backend"      : "onnxruntime",
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
        default=PROJECT_ROOT / "checkpoints" / "student" / "kd",
        help="Folder KD yang berisi config.json dan best_model.pth",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--opset",      type=int, default=13)
    parser.add_argument("--output",     type=Path, default=None,
                        help="Path output ONNX. Default: <model-dir>/model_benchmark.onnx")
    parser.add_argument("--weights", type=Path, default=Path("best_screening.pth"),
                        help="Checkpoint inside model-dir, or an absolute checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir  = args.model_dir.resolve()
    kd_config_path = model_dir / "config.json"
    model_path = args.weights if args.weights.is_absolute() else model_dir / args.weights
    onnx_path      = args.output or (model_dir / "model_benchmark.onnx")

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
        student_config_path = resolve_project_path(kd_cfg["student_config_path"])
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
    model = build_model(kd_cfg, student_cfg, model_path)

    print(f"\n  Exporting ONNX (opset={args.opset}, input={args.input_size}x{args.input_size})...")
    size_mb = export_onnx(model, onnx_path, args.input_size, args.opset)

    meta_path = write_metadata(
        model_dir, kd_cfg, student_cfg, onnx_path, size_mb, args.input_size, args.opset,
        model_path,
    )

    print(f"\n  ✓ ONNX exported : {onnx_path}")
    print(f"  ✓ Metadata      : {meta_path}")
    print(f"  ✓ Model size    : {size_mb:.3f} MB")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
