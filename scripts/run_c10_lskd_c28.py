#!/usr/bin/env python3
"""Run the controlled C28→C10 CE + Logit Standardization KD experiment.

Invoke this file with ``py -3.11`` on Windows.  Child processes reuse the
same interpreter, avoiding PowerShell-specific command construction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEACHER_CONFIG = ROOT / "nas_results/teacher_l020_c28_cells12_stem8_arcface_300e/seed_42/config.json"
TEACHER_CHECKPOINT = ROOT / "nas_results/teacher_l020_c28_cells12_stem8_arcface_300e/seed_42/best_screening.pth"
STUDENT_CONFIG = ROOT / "nas_results/retrain_l020_c10_pk_ce_300e/seed_42/config.json"
INITIAL_STATE = ROOT / "nas_results/controlled_initial_states/l020_c10_stem8_cells8_seed42.pth"
BASELINE_DIR = ROOT / "nas_results/retrain_l020_c10_pk_ce_300e/seed_42"
DATA_DIR = ROOT / "preprocessed_results"
SPLIT = ROOT / "PalmVein_Lightweight_Benchmark/dataset/splits/split_info.json"
RUN_DIR = ROOT / "knowledge_distilation/kd_results/lskd_t2_w9_l020_c10_c28ta_seed42"
SMOKE_DIR = ROOT / "knowledge_distilation/kd_results/lskd_t2_w9_l020_c10_c28ta_seed42_smoke"
DIAGNOSTIC_DIR = ROOT / "results/diagnostics/c10_lskd_c28_seed42"
SELECTION = DIAGNOSTIC_DIR / "selection.json"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path.relative_to(ROOT)}")


def validate_inputs() -> None:
    for path in (
        TEACHER_CONFIG,
        TEACHER_CHECKPOINT,
        STUDENT_CONFIG,
        INITIAL_STATE,
        BASELINE_DIR / "best_screening.pth",
        SPLIT,
    ):
        require_file(path)
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
    student_cfg = json.loads(STUDENT_CONFIG.read_text(encoding="utf-8"))
    if student_cfg.get("loss_mode") != "ce":
        raise ValueError("Matched student config must use ordinary CE")
    student_arch = (
        int(student_cfg.get("C_init", -1)),
        int(student_cfg.get("num_cells", -1)),
        int(student_cfg.get("stem_downsample", -1)),
    )
    if student_arch != (10, 8, 8):
        raise ValueError(f"Expected C10/cells8/stem8 student, got {student_arch}")
    baseline_smoothing = float(
        student_cfg.get(
            "label_smoothing",
            student_cfg.get("retrain_cfg", {}).get("label_smoothing", -1),
        )
    )
    if baseline_smoothing != 0.2:
        raise ValueError(
            f"Matched PK-CE control must use label smoothing 0.2, got {baseline_smoothing}"
        )
    teacher_cfg = json.loads(TEACHER_CONFIG.read_text(encoding="utf-8"))
    if teacher_cfg.get("loss_mode") not in {"arcface", "subcenter_arcface"}:
        raise ValueError("C28 teacher config must use an ArcFace-compatible head")
    teacher_arch = (
        int(teacher_cfg.get("C_init", -1)),
        int(teacher_cfg.get("num_cells", -1)),
        int(teacher_cfg.get("stem_downsample", -1)),
    )
    if teacher_arch != (28, 12, 8):
        raise ValueError(f"Expected C28/cells12/stem8 teacher, got {teacher_arch}")


def run(command: list[str]) -> None:
    display = subprocess.list2cmdline(command)
    print(f"\n> {display}\n", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def train(smoke: bool) -> None:
    validate_inputs()
    output = SMOKE_DIR if smoke else RUN_DIR
    if not smoke and (output / "screening_results.json").exists():
        raise FileExistsError(
            f"Completed output already exists: {output.relative_to(ROOT)}. "
            "Refusing to overwrite a provenance-bearing experiment."
        )
    epochs = 1 if smoke else 300
    warmup = 1 if smoke else 10
    workers = 0 if smoke else 4
    run([
        sys.executable,
        "knowledge_distilation/kd_train.py",
        "--teacher_arch", "nas_eval",
        "--teacher_config", str(TEACHER_CONFIG.relative_to(ROOT)),
        "--teacher_weights", str(TEACHER_CHECKPOINT.relative_to(ROOT)),
        "--student_config", str(STUDENT_CONFIG.relative_to(ROOT)),
        "--student_weights", str(INITIAL_STATE.relative_to(ROOT)),
        "--initial_student_weights", str(INITIAL_STATE.relative_to(ROOT)),
        "--no_pretrained_student",
        "--data_dir", str(DATA_DIR.relative_to(ROOT)),
        "--split_path", str(SPLIT.relative_to(ROOT)),
        "--kd_method", "logit_standardization",
        "--temperature", "2",
        "--ce_weight", "1",
        "--ls_kd_weight", "9",
        "--ls_eps", "0.0000001",
        "--epochs", str(epochs),
        "--batch_size", "64",
        "--lr", "0.001",
        "--lr_min", "0.000001",
        "--weight_decay", "0.05",
        "--warmup_epochs", str(warmup),
        "--augmentation_policy", "v4_robust_light",
        "--train_sampler", "pk",
        "--pk_p", "16",
        "--pk_k", "4",
        "--label_smoothing", "0.2",
        "--drop_path", "0",
        "--cutout_length", "0",
        "--no_mix",
        "--seed", "42",
        "--num_workers", str(workers),
        "--output_dir", str(output.relative_to(ROOT)),
        "--skip-test-evaluation",
    ])


def evaluate_validation(config: Path, checkpoint: Path, output: Path) -> None:
    run([
        sys.executable,
        "scripts/evaluate_frozen_identification.py",
        "--config", str(config.relative_to(ROOT)),
        "--checkpoint", str(checkpoint.relative_to(ROOT)),
        "--data-dir", str(DATA_DIR.relative_to(ROOT)),
        "--split-path", str(SPLIT.relative_to(ROOT)),
        "--partition", "val",
        "--output-dir", str(output.relative_to(ROOT)),
        "--batch-size", "64",
        "--num-workers", "0",
    ])


def select() -> None:
    validate_inputs()
    for path in (
        RUN_DIR / "config.json",
        RUN_DIR / "best_screening.pth",
        RUN_DIR / "screening_results.json",
    ):
        require_file(path)
    baseline_eval = DIAGNOSTIC_DIR / "pk_ce_validation"
    candidate_eval = DIAGNOSTIC_DIR / "lskd_validation"
    evaluate_validation(
        BASELINE_DIR / "config.json",
        BASELINE_DIR / "best_screening.pth",
        baseline_eval,
    )
    evaluate_validation(
        RUN_DIR / "config.json",
        RUN_DIR / "best_screening.pth",
        candidate_eval,
    )
    run([
        sys.executable,
        "scripts/select_c10_lskd_c28.py",
        "--baseline-eval", str((baseline_eval / "results.json").relative_to(ROOT)),
        "--candidate-eval", str((candidate_eval / "results.json").relative_to(ROOT)),
        "--candidate-dir", str(RUN_DIR.relative_to(ROOT)),
        "--initial-state", str(INITIAL_STATE.relative_to(ROOT)),
        "--teacher-config", str(TEACHER_CONFIG.relative_to(ROOT)),
        "--teacher-checkpoint", str(TEACHER_CHECKPOINT.relative_to(ROOT)),
        "--split", str(SPLIT.relative_to(ROOT)),
        "--output", str(SELECTION.relative_to(ROOT)),
    ])


def final_eval(acknowledge_observed_test: bool) -> None:
    if not acknowledge_observed_test:
        raise ValueError(
            "final_eval requires --acknowledge-observed-test because the split "
            "has already been observed in prior experiments"
        )
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection.get("status") != "method_selected":
        raise RuntimeError("LS-KD was not selected on validation; test evaluation is blocked")
    run([
        sys.executable,
        "scripts/evaluate_frozen_identification.py",
        "--config", str((RUN_DIR / "config.json").relative_to(ROOT)),
        "--checkpoint", str((RUN_DIR / "best_screening.pth").relative_to(ROOT)),
        "--data-dir", str(DATA_DIR.relative_to(ROOT)),
        "--split-path", str(SPLIT.relative_to(ROOT)),
        "--partition", "test",
        "--acknowledge-observed-test",
        "--output-dir", "results/final/c10_lskd_c28/seed_42/pytorch_test",
        "--batch-size", "64",
        "--num-workers", "0",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=["smoke", "train", "select", "final_eval"]
    )
    parser.add_argument("--acknowledge-observed-test", action="store_true")
    args = parser.parse_args()
    if args.mode == "smoke":
        train(smoke=True)
    elif args.mode == "train":
        train(smoke=False)
    elif args.mode == "select":
        select()
    else:
        final_eval(args.acknowledge_observed_test)


if __name__ == "__main__":
    main()
