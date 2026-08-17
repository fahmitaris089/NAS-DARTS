"""Export KD model ke ONNX FP32 + INT8 (static quantization) dalam satu pipeline.

Pipeline lengkap:
1. Load KD best_model.pth
2. Export ke ONNX FP32 (opset 13+)
3. Quantize ke INT8 (per-channel, QDQ format)
4. Benchmark FP32 vs INT8 (latency + size)
5. Evaluate accuracy FP32 vs INT8

Usage:
    python3 export_kd_onnx_int8.py \
        --model-dir knowledge_distilation/kd_results/kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed \
        --num-calib 200 \
        --eval-accuracy

Output:
    <model-dir>/model_benchmark.onnx               # FP32 ONNX
    <model-dir>/model_benchmark_int8_static.onnx   # INT8 ONNX
    <model-dir>/model_benchmark_metadata.json      # Export metadata
    <model-dir>/benchmark_int8_static_results.json # Benchmark results
    <model-dir>/model_benchmark_acc.json           # FP32 accuracy (optional)
    <model-dir>/model_benchmark_int8_static_acc.json # INT8 accuracy (optional)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
except ImportError as e:
    raise SystemExit(f"onnxruntime quantization modules unavailable: {e}")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "nas"))

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from adaface import replace_linear_with_adaface, replace_linear_with_arcface
from operations import fuse_reparam_model
from palm_input_preprocessing import (
    input_profile_metadata,
    preprocess_path_to_imagenet_bchw,
    validate_input_profile,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved: {path.name}")


def annotate_onnx_metadata(path: Path, properties: dict[str, object]) -> None:
    """Persist deployment-critical external preprocessing metadata in ONNX."""
    import onnx

    model = onnx.load(str(path))
    existing = {item.key: item.value for item in model.metadata_props}
    existing.update({key: str(value) for key, value in properties.items()})
    onnx.helper.set_model_props(model, existing)
    onnx.save(model, str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path_value: str | Path | None, default: Path | None = None) -> Path | None:
    if path_value is None:
        return default
    path = Path(path_value)
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build & Load Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(kd_cfg: dict, student_cfg: dict, model_path: Path) -> EvalNetwork:
    """Rebuild student EvalNetwork persis seperti saat KD training (auxiliary=False)."""
    genotype = dict_to_genotype(student_cfg["genotype"])

    # Ambil C_init & num_cells dari student config
    c_init    = int(student_cfg.get("C_init",    kd_cfg.get("student_C_init", 4)))
    num_cells = int(student_cfg.get("num_cells", kd_cfg.get("student_num_cells", 8)))
    num_classes = int(kd_cfg.get("num_classes", 834))
    dropout   = float(kd_cfg.get("student_dropout", 0.3))
    stem_downsample = int(student_cfg.get("stem_downsample", 2))
    stem_pool = str(student_cfg.get("stem_pool", "max"))
    reduction_indices = parse_reduction_indices(student_cfg.get("reduction_indices"))

    print(f"  Architecture : C_init={c_init}, num_cells={num_cells}, "
          f"num_classes={num_classes}, auxiliary=False, dropout={dropout}, "
          f"stem_downsample={stem_downsample}, stem_pool={stem_pool}, "
          f"reduction_indices={reduction_indices}")

    model = EvalNetwork(
        genotype          = genotype,
        C_init            = c_init,
        num_cells         = num_cells,
        num_classes       = num_classes,
        auxiliary         = False,   # KD selalu auxiliary=False
        dropout           = dropout,
        stem_downsample   = stem_downsample,
        stem_pool         = stem_pool,
        reduction_indices = reduction_indices,
    )

    loss_mode = student_cfg.get("loss_mode", "ce")
    use_adaface = bool(loss_mode == "adaface" or kd_cfg.get("adaface"))
    if loss_mode in {"arcface", "subcenter_arcface"}:
        replace_linear_with_arcface(
            model, num_classes=num_classes,
            m=float(student_cfg.get("arcface_margin", 0.5)),
            s=float(student_cfg.get("arcface_scale", 64.0)),
            num_subcenters=int(student_cfg.get(
                "arcface_subcenters", 2 if loss_mode == "subcenter_arcface" else 1
            )),
        )
        print(f"  Classifier   : {loss_mode} cosine inference head (training margin disabled)")
    elif use_adaface:
        replace_linear_with_adaface(
            model, num_classes=num_classes,
            m=float(student_cfg.get("adaface_m", kd_cfg.get("adaface_m", 0.4))),
            h=float(student_cfg.get("adaface_h", kd_cfg.get("adaface_h", 0.333))),
            s=float(student_cfg.get("adaface_s", kd_cfg.get("adaface_s", 64.0))),
            t_alpha=float(student_cfg.get("adaface_t_alpha", kd_cfg.get("adaface_t_alpha", 0.01))),
        )
        print("  Classifier   : AdaFace cosine inference head (training margin disabled)")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "student" in state_dict:
        state_dict = state_dict["student"]
    # Skip auxiliary head keys jika ada
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("_auxiliary_head")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    material_missing = [key for key in missing if not key.startswith("_auxiliary_head")]
    if material_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch; missing={material_missing[:10]} unexpected={unexpected[:10]}"
        )
    if missing:
        print(f"  [warn] Missing keys  : {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"  [warn] Unexpected keys: {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")

    model.eval()
    _, n_fused = fuse_reparam_model(model)
    if n_fused:
        print(f"  Fused        : {n_fused} RepConvBN block(s)")
    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"  Params       : {n_params:.1f}K")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Export ONNX FP32
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx_fp32(
    model: nn.Module,
    output_path: Path,
    input_size: int = 224,
    opset: int = 13,
) -> float:
    """Export PyTorch model ke ONNX FP32."""
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
    print(f"  ✓ ONNX FP32  : {output_path.name} ({size_mb:.3f} MB)")
    return size_mb


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: INT8 Static Quantization
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_bmp(
    path: Path,
    input_size: int = 224,
    input_profile: str = "legacy",
) -> np.ndarray:
    """BMP grayscale -> normalized RGB tensor BCHW float32."""
    return preprocess_path_to_imagenet_bchw(
        str(path), input_size=input_size, profile=input_profile,
    )


def collect_calibration_images(calib_dir: Path, limit: int) -> List[Path]:
    images = sorted(calib_dir.rglob("*.bmp"))
    if not images:
        raise FileNotFoundError(f"No .bmp files found under {calib_dir}")
    return images[: min(limit, len(images))]


def ensure_min_opset(fp32_path: Path, min_opset: int = 13) -> Path:
    """Guarantee opset >= min_opset untuk per-channel quantization."""
    import onnx

    model = onnx.load(str(fp32_path))
    current = max((op.version for op in model.opset_import if op.domain in ("", "ai.onnx")), default=0)
    if current >= min_opset:
        return fp32_path

    print(f"  [opset] Upgrading from opset {current} → {min_opset} for per-channel quant")
    upgraded = onnx.version_converter.convert_version(model, min_opset)
    onnx.checker.check_model(upgraded)
    up_path = fp32_path.with_name(fp32_path.stem + f"_op{min_opset}.onnx")
    onnx.save(upgraded, str(up_path))
    print(f"  [opset] Saved: {up_path.name}")
    return up_path


class PalmVeinCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        image_paths: List[Path],
        input_name: str,
        input_size: int,
        input_profile: str,
    ):
        self.input_name = input_name
        self.input_size = input_size
        self._data = [
            preprocess_bmp(p, input_size, input_profile) for p in image_paths
        ]
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self._data):
            return None
        x = self._data[self._idx]
        self._idx += 1
        return {self.input_name: x}


def quantize_to_int8(
    fp32_path: Path,
    int8_path: Path,
    calib_images: List[Path],
    input_name: str,
    input_size: int,
    input_profile: str,
) -> dict:
    """Static INT8 quantization dengan per-channel QDQ."""
    
    # 1. Ensure opset >= 13
    quant_input_path = ensure_min_opset(fp32_path, min_opset=13)

    # 2. Pre-processing (symbolic shape inference + graph cleanup)
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre_path = quant_input_path.with_name(quant_input_path.stem + "_pre.onnx")
        quant_pre_process(str(quant_input_path), str(pre_path), skip_symbolic_shape=False)
        quant_input_path = pre_path
        print(f"  [pre] Quant pre-process done: {pre_path.name}")
    except Exception as exc:
        print(f"  [pre] Quant pre-process skipped ({exc})")

    # 3. Quantize
    reader = PalmVeinCalibrationReader(
        calib_images, input_name, input_size, input_profile,
    )
    quantize_static(
        model_input=str(quant_input_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    int8_size = int8_path.stat().st_size / 1e6
    print(f"  ✓ ONNX INT8  : {int8_path.name} ({int8_size:.3f} MB)")

    return {
        "per_channel": True,
        "quant_format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "quant_input_onnx": str(quant_input_path),
        "quant_pre_process": quant_input_path.name.endswith("_pre.onnx"),
        "num_calib_images": len(calib_images),
        "input_profile": input_profile,
        "input_profile_metadata": input_profile_metadata(input_profile),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def make_session(model_path: Path, threads: int):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def benchmark_onnx(model_path: Path, input_size: int, threads: int, warmup: int, runs: int):
    sess = make_session(model_path, threads)
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.array(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
    }


@torch.inference_mode()
def verify_torch_onnx_parity(
    model: nn.Module,
    onnx_path: Path,
    arrays: list[np.ndarray],
    threads: int,
) -> dict:
    session = make_session(onnx_path, threads)
    input_name = session.get_inputs()[0].name
    max_abs_error = 0.0
    mean_abs_errors = []
    top1_matches = 0
    for array in arrays:
        torch_logits = model(torch.from_numpy(array)).detach().cpu().numpy()
        onnx_logits = np.asarray(session.run(None, {input_name: array})[0])
        difference = np.abs(torch_logits - onnx_logits)
        max_abs_error = max(max_abs_error, float(difference.max()))
        mean_abs_errors.append(float(difference.mean()))
        top1_matches += int(
            np.array_equal(torch_logits.argmax(1), onnx_logits.argmax(1))
        )
    payload = {
        "samples": len(arrays),
        "top1_matches": top1_matches,
        "top1_parity": top1_matches == len(arrays),
        "max_abs_error": max_abs_error,
        "mean_abs_error": float(np.mean(mean_abs_errors)),
    }
    if not payload["top1_parity"]:
        raise RuntimeError(f"PyTorch/ONNX top-1 parity failed: {payload}")
    return payload


def verify_onnx_top1_parity(
    reference_path: Path,
    candidate_path: Path,
    arrays: list[np.ndarray],
    threads: int,
) -> dict:
    reference = make_session(reference_path, threads)
    candidate = make_session(candidate_path, threads)
    reference_name = reference.get_inputs()[0].name
    candidate_name = candidate.get_inputs()[0].name
    matches = 0
    for array in arrays:
        reference_logits = reference.run(None, {reference_name: array})[0]
        candidate_logits = candidate.run(None, {candidate_name: array})[0]
        matches += int(
            np.array_equal(reference_logits.argmax(1), candidate_logits.argmax(1))
        )
    payload = {
        "samples": len(arrays),
        "top1_matches": matches,
        "top1_parity": matches == len(arrays),
        "partition": "training_calibration_only",
    }
    if not payload["top1_parity"]:
        raise RuntimeError(f"FP32/INT8 calibration top-1 parity failed: {payload}")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Accuracy Evaluation (Optional)
# ─────────────────────────────────────────────────────────────────────────────

def build_label_map(subjects: list[str]) -> dict[str, int]:
    ordered = sorted((str(subject) for subject in subjects), key=int)
    return {subject: idx for idx, subject in enumerate(ordered)}


def build_test_samples(data_dir: Path, split: dict) -> list[tuple[Path, int]]:
    label_map = build_label_map(split["subjects"])
    samples: list[tuple[Path, int]] = []
    missing: list[str] = []

    for subject_id, filename in split["test"]:
        img_path = data_dir / str(subject_id) / filename
        if img_path.exists():
            samples.append((img_path, label_map[str(subject_id)]))
        else:
            missing.append(f"{subject_id}/{filename}")

    if missing:
        print(f"  [warn] {len(missing)} test files not found (skipped)")
    if not samples:
        raise FileNotFoundError(f"No test samples found under {data_dir}")
    return samples


def evaluate_onnx_accuracy(
    model_path: Path,
    data_dir: Path,
    split_path: Path,
    input_size: int,
    threads: int,
    input_profile: str,
) -> dict:
    """Evaluate ONNX model accuracy with the same split/label mapping as training."""
    sess = make_session(model_path, threads)
    input_name = sess.get_inputs()[0].name
    split = load_json(split_path)
    test_images = build_test_samples(data_dir, split)
    print(f"  Evaluating on {len(test_images)} images, {len(split['subjects'])} classes...")

    correct = 0
    total = 0
    for img_path, true_label in test_images:
        x = preprocess_bmp(img_path, input_size, input_profile)
        logits = sess.run(None, {input_name: x})[0]
        pred = int(np.argmax(logits, axis=1)[0])
        if pred == true_label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": round(accuracy, 6),
        "correct": correct,
        "total": total,
        "num_classes": len(split["subjects"]),
        "data_dir": str(data_dir),
        "split_path": str(split_path),
        "input_profile": input_profile,
        "input_profile_metadata": input_profile_metadata(input_profile),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export KD model ke ONNX FP32 + INT8")
    parser.add_argument(
        "--model-dir", type=Path, required=True,
        help="Folder KD yang berisi config.json dan best_model.pth",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("best_model.pth"),
        help="Checkpoint filename inside --model-dir, or an absolute checkpoint path",
    )
    parser.add_argument(
        "--output-stem", type=str, default="model_benchmark",
        help="Output basename; FP32 uses .onnx and INT8 uses _int8_static.onnx",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=13)
    
    # Quantization
    parser.add_argument("--calib-dir", type=Path, default=Path("dataset/calibration"),
                        help="Calibration image root")
    parser.add_argument("--num-calib", type=int, default=200,
                        help="Number of calibration images")
    parser.add_argument("--calibration-manifest", type=Path,
                        default=Path("PalmVein_Lightweight_Benchmark/dataset/calibration_manifest.json"),
                        help="Manifest whose every selected entry must have source_split=train")
    
    # Benchmark
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    
    # Accuracy evaluation
    parser.add_argument("--eval-accuracy", action="store_true",
                        help="Evaluate accuracy on test set")
    parser.add_argument("--acknowledge-observed-test", action="store_true",
                        help="Required with --eval-accuracy because this test split was previously observed")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset root directory. Default: infer from KD config")
    parser.add_argument("--split-path", type=Path, default=None,
                        help="Split JSON path. Default: infer from KD config")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_accuracy and not args.acknowledge_observed_test:
        raise SystemExit("--eval-accuracy requires --acknowledge-observed-test")
    model_dir  = args.model_dir.resolve()
    kd_config_path = model_dir / "config.json"
    model_path = args.weights if args.weights.is_absolute() else model_dir / args.weights
    fp32_path      = model_dir / f"{args.output_stem}.onnx"
    int8_path      = model_dir / f"{args.output_stem}_int8_static.onnx"
    calib_dir      = resolve_path(args.calib_dir)
    calibration_manifest = resolve_path(args.calibration_manifest)

    if not kd_config_path.exists():
        raise FileNotFoundError(f"KD config not found: {kd_config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Weights not found: {model_path}")
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")
    if calibration_manifest is None or not calibration_manifest.exists():
        raise FileNotFoundError(f"Training-only calibration manifest not found: {calibration_manifest}")

    print_section("KD MODEL → ONNX FP32 + INT8 PIPELINE")
    print(f"  Model dir  : {model_dir}")
    print(f"  FP32 ONNX  : {fp32_path.name}")
    print(f"  INT8 ONNX  : {int8_path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Load config & build model
    # ─────────────────────────────────────────────────────────────────────────
    print_section("1. LOAD MODEL")
    kd_cfg = load_json(kd_config_path)
    eval_data_dir = resolve_path(args.data_dir, resolve_path(kd_cfg.get("data_dir")))
    eval_split_path = resolve_path(args.split_path, resolve_path(kd_cfg.get("split_path"), PROJECT_ROOT / "split_info.json"))

    if "student_config_path" in kd_cfg:
        student_config_path = resolve_path(kd_cfg["student_config_path"])
        if not student_config_path.exists():
            raise FileNotFoundError(f"Student config not found: {student_config_path}")
        student_cfg = load_json(student_config_path)
    elif "genotype" in kd_cfg:
        print("  [info] Using config.json as student config (genotype found)")
        student_cfg = kd_cfg
    else:
        raise KeyError("config.json missing both 'student_config_path' and 'genotype'")

    model = build_model(kd_cfg, student_cfg, model_path)
    input_profile = str(student_cfg.get("input_profile", kd_cfg.get("input_profile", "legacy")))
    validate_input_profile(input_profile)
    print(f"  Input profile: {input_profile}")

    # ─────────────────────────────────────────────────────────────────────────
    # Export FP32 ONNX
    # ─────────────────────────────────────────────────────────────────────────
    print_section("2. EXPORT ONNX FP32")
    fp32_size_mb = export_onnx_fp32(model, fp32_path, args.input_size, args.opset)
    onnx_properties = {
        "input_profile": input_profile,
        "input_profile_parameters": json.dumps(
            input_profile_metadata(input_profile), sort_keys=True,
        ),
        "external_preprocessing_required": True,
        "stem_pool": student_cfg.get("stem_pool", "max"),
    }
    annotate_onnx_metadata(fp32_path, onnx_properties)
    fp32_size_mb = fp32_path.stat().st_size / 1e6

    # ─────────────────────────────────────────────────────────────────────────
    # Quantize to INT8
    # ─────────────────────────────────────────────────────────────────────────
    print_section("3. QUANTIZE TO INT8")
    calib_images = collect_calibration_images(calib_dir, args.num_calib)
    parity_arrays = [
        preprocess_bmp(path, args.input_size, input_profile)
        for path in calib_images[: min(16, len(calib_images))]
    ]
    torch_fp32_parity = verify_torch_onnx_parity(
        model, fp32_path, parity_arrays, args.threads,
    )
    manifest = load_json(calibration_manifest)
    manifest_entries = manifest.get("entries", [])
    if not manifest_entries or any(entry.get("source_split") != "train" for entry in manifest_entries):
        raise ValueError("Calibration manifest must contain only source_split=train entries")
    allowed_pairs = {
        (str(entry.get("subject")), str(entry.get("filename")))
        for entry in manifest_entries
    }
    actual_pairs = [(path.parent.name, path.name) for path in calib_images]
    unverified = [pair for pair in actual_pairs if pair not in allowed_pairs]
    validation_mode = "exact_source_manifest"
    if unverified:
        if eval_split_path is None or not eval_split_path.exists():
            raise ValueError(
                "Calibration images differ from the source manifest and the training "
                f"split is unavailable for independent validation: {unverified[:10]}"
            )
        split_payload = load_json(eval_split_path)
        training_pairs = {
            (str(subject), str(filename))
            for subject, filename in split_payload.get("train", [])
        }
        outside_train = [pair for pair in actual_pairs if pair not in training_pairs]
        if outside_train:
            raise ValueError(
                "Calibration contains images outside the training split: "
                f"{outside_train[:10]}"
            )
        validation_mode = "independently_validated_against_training_split"
        print(
            "  [info] Calibration selection differs from the source manifest; "
            "all selected files were independently verified against split.train"
        )

    used_manifest_path = model_dir / f"{args.output_stem}_calibration_manifest_used.json"
    used_manifest = {
        "count": len(calib_images),
        "source_split": "train",
        "input_profile": input_profile,
        "input_profile_metadata": input_profile_metadata(input_profile),
        "validation_mode": validation_mode,
        "source_manifest": str(calibration_manifest),
        "source_manifest_sha256": sha256_file(calibration_manifest),
        "split_path": str(eval_split_path),
        "split_sha256": sha256_file(eval_split_path),
        "entries": [
            {
                "subject": subject,
                "filename": path.name,
                "relative_path": str(path.relative_to(calib_dir)),
                "sha256": sha256_file(path),
                "source_split": "train",
            }
            for path, (subject, _) in zip(calib_images, actual_pairs)
        ],
    }
    save_json(used_manifest_path, used_manifest)
    print(f"  Calibration images: {len(calib_images)}")

    fp32_sess = make_session(fp32_path, args.threads)
    input_name = fp32_sess.get_inputs()[0].name

    quant_recipe = quantize_to_int8(
        fp32_path, int8_path, calib_images, input_name, args.input_size,
        input_profile,
    )
    annotate_onnx_metadata(int8_path, onnx_properties)
    fp32_int8_parity = verify_onnx_top1_parity(
        fp32_path, int8_path, parity_arrays, args.threads,
    )
    int8_size_mb = int8_path.stat().st_size / 1e6

    # ─────────────────────────────────────────────────────────────────────────
    # Benchmark
    # ─────────────────────────────────────────────────────────────────────────
    print_section("4. BENCHMARK FP32 vs INT8")
    fp32_stats = benchmark_onnx(fp32_path, args.input_size, args.threads, args.warmup, args.runs)
    int8_stats = benchmark_onnx(int8_path, args.input_size, args.threads, args.warmup, args.runs)

    speedup = fp32_stats["mean_ms"] / int8_stats["mean_ms"] if int8_stats["mean_ms"] > 0 else float("nan")

    print(f"  FP32 ONNX {args.threads}T : {fp32_stats['mean_ms']:.2f} ms "
          f"(median={fp32_stats['median_ms']:.2f}, p95={fp32_stats['p95_ms']:.2f})")
    print(f"  INT8 ONNX {args.threads}T : {int8_stats['mean_ms']:.2f} ms "
          f"(median={int8_stats['median_ms']:.2f}, p95={int8_stats['p95_ms']:.2f})")
    print(f"  Speedup        : {speedup:.2f}x")
    print(f"  Size reduction : {fp32_size_mb:.3f} MB → {int8_size_mb:.3f} MB "
          f"({int8_size_mb/fp32_size_mb:.2f}x)")

    # ─────────────────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────────────────
    c_init    = int(student_cfg.get("C_init",    kd_cfg.get("student_C_init", 4)))
    num_cells = int(student_cfg.get("num_cells", kd_cfg.get("student_num_cells", 8)))
    stem_downsample = int(student_cfg.get("stem_downsample", 2))
    stem_pool = str(student_cfg.get("stem_pool", "max"))
    reduction_indices = parse_reduction_indices(student_cfg.get("reduction_indices"))

    metadata = {
        "exported_at"  : datetime.now().isoformat(),
        "model_dir"    : str(model_dir),
        "model_path"   : str(model_path),
        "fp32_onnx"    : str(fp32_path),
        "int8_onnx"    : str(int8_path),
        "fp32_onnx_sha256": sha256_file(fp32_path),
        "int8_onnx_sha256": sha256_file(int8_path),
        "input_size"   : args.input_size,
        "opset"        : args.opset,
        "num_classes"  : int(kd_cfg.get("num_classes", 834)),
        "c_init"       : c_init,
        "num_cells"    : num_cells,
        "stem_downsample": stem_downsample,
        "stem_pool"      : stem_pool,
        "reduction_indices": reduction_indices,
        "auxiliary"    : False,
        "fp32_size_mb" : round(fp32_size_mb, 4),
        "int8_size_mb" : round(int8_size_mb, 4),
        "backend"      : "onnxruntime",
        "pytorch_fp32_onnx_parity": torch_fp32_parity,
        "fp32_int8_calibration_top1_parity": fp32_int8_parity,
        "input_profile": input_profile,
        "input_profile_metadata": input_profile_metadata(input_profile),
        "calibration_manifest": str(used_manifest_path),
        "calibration_manifest_sha256": sha256_file(used_manifest_path),
        "calibration_validation_mode": validation_mode,
        "calibration_source_split": "train",
        "input_profile": input_profile,
        "input_profile_metadata": input_profile_metadata(input_profile),
        "calibration_images": len(calib_images),
        "test_previously_observed_acknowledged": bool(args.acknowledge_observed_test),
        "kd_config"    : {
            "teacher_arch": kd_cfg.get("teacher_arch"),
            "temperature" : kd_cfg.get("temperature"),
            "alpha"       : kd_cfg.get("alpha"),
            "epochs"      : kd_cfg.get("epochs"),
        },
    }
    save_json(model_dir / "model_benchmark_metadata.json", metadata)

    benchmark_results = {
        "model_dir": str(model_dir),
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path),
        "calib_dir": str(calib_dir),
        "calibration_manifest": str(used_manifest_path),
        "calibration_manifest_sha256": sha256_file(used_manifest_path),
        "calibration_validation_mode": validation_mode,
        "calibration_source_split": "train",
        "threads": args.threads,
        "fp32_size_mb": round(fp32_size_mb, 4),
        "int8_size_mb": round(int8_size_mb, 4),
        "fp32_4t_ms": round(fp32_stats["mean_ms"], 4),
        "int8_4t_ms": round(int8_stats["mean_ms"], 4),
        "speedup_x": round(speedup, 4),
        "quant_recipe": quant_recipe,
        "pytorch_fp32_onnx_parity": torch_fp32_parity,
        "fp32_int8_calibration_top1_parity": fp32_int8_parity,
    }
    save_json(model_dir / "benchmark_int8_static_results.json", benchmark_results)

    # ─────────────────────────────────────────────────────────────────────────
    # Optional: Evaluate accuracy
    # ─────────────────────────────────────────────────────────────────────────
    if args.eval_accuracy:
        if eval_data_dir is None or not eval_data_dir.exists():
            print(f"\n  [warn] Data dir not found: {eval_data_dir}, skipping accuracy eval")
        elif eval_split_path is None or not eval_split_path.exists():
            print(f"\n  [warn] Split path not found: {eval_split_path}, skipping accuracy eval")
        else:
            print_section("5. EVALUATE ACCURACY")
            print(f"  Data dir   : {eval_data_dir}")
            print(f"  Split path : {eval_split_path}")
            
            print(f"\n  FP32 Accuracy:")
            fp32_acc = evaluate_onnx_accuracy(
                fp32_path, eval_data_dir, eval_split_path, args.input_size,
                args.threads, input_profile,
            )
            save_json(model_dir / "model_benchmark_acc.json", fp32_acc)
            
            print(f"\n  INT8 Accuracy:")
            int8_acc = evaluate_onnx_accuracy(
                int8_path, eval_data_dir, eval_split_path, args.input_size,
                args.threads, input_profile,
            )
            save_json(model_dir / "model_benchmark_int8_static_acc.json", int8_acc)
            
            acc_drop = fp32_acc["accuracy"] - int8_acc["accuracy"]
            print(f"\n  Accuracy drop: {acc_drop:.4f} ({acc_drop*100:.2f}%)")

    print_section("✓ PIPELINE COMPLETE")
    print(f"  FP32 ONNX : {fp32_path}")
    print(f"  INT8 ONNX : {int8_path}")
    print(f"  Metadata  : {model_dir / 'model_benchmark_metadata.json'}")
    print(f"  Benchmark : {model_dir / 'benchmark_int8_static_results.json'}")
    if (
        args.eval_accuracy
        and eval_data_dir is not None and eval_data_dir.exists()
        and eval_split_path is not None and eval_split_path.exists()
    ):
        print(f"  FP32 Acc  : {model_dir / 'model_benchmark_acc.json'}")
        print(f"  INT8 Acc  : {model_dir / 'model_benchmark_int8_static_acc.json'}")
    print()


if __name__ == "__main__":
    main()
