#!/usr/bin/env python3
"""Select C28→C10 LS-KD against the matched C10 PK-CE control on validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation_candidate(name: str, result_path: Path) -> dict:
    result = load_json(result_path)
    if result.get("partition") != "val":
        raise ValueError(f"{name} is not a validation result: {result_path}")
    if result.get("test_previously_observed_acknowledged"):
        raise ValueError(f"{name} validation artifact claims test access")
    samples = int(result["samples"])
    correct = int(result["correct"])
    candidate = {
        "name": name,
        "correct": correct,
        "samples": samples,
        "validation_errors": samples - correct,
        "ordinary_ce_loss": float(result["ordinary_ce_loss"]),
        "true_class_margin": float(result["mean_true_class_margin"]),
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "config": result["config"],
        "config_sha256": result["config_sha256"],
    }
    candidate["selection_key"] = [
        candidate["validation_errors"],
        candidate["ordinary_ce_loss"],
        -candidate["true_class_margin"],
    ]
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-eval", required=True, type=Path)
    parser.add_argument("--candidate-eval", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--initial-state", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    runtime_config_path = args.candidate_dir / "config.json"
    screening_path = args.candidate_dir / "screening_results.json"
    checkpoint_path = args.candidate_dir / "best_screening.pth"
    runtime = load_json(runtime_config_path)
    screening = load_json(screening_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if screening.get("test_acc") is not None or screening.get("test_loss") is not None:
        raise ValueError("LS-KD screening artifact contains test metrics")
    if not screening.get("kd_config", {}).get("skip_test_evaluation"):
        raise ValueError("LS-KD run did not record --skip-test-evaluation")
    if runtime.get("test_loader_created") is not False:
        raise ValueError("LS-KD screening created or did not audit a test loader")

    expected = {
        "kd_method": "logit_standardization",
        "student_loss_mode": "ce",
        "temperature": 2.0,
        "ce_weight": 1.0,
        "ls_kd_weight": 9.0,
        "ls_eps": 1e-7,
        "label_smoothing": 0.2,
        "epochs": 300,
        "batch_size": 64,
        "lr": 0.001,
        "lr_min": 1e-6,
        "weight_decay": 0.05,
        "warmup_epochs": 10,
        "augmentation_policy": "v4_robust_light",
        "train_sampler": "pk",
        "pk_p": 16,
        "pk_k": 4,
        "drop_path_prob": 0.0,
        "cutout_length": 0,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "seed": 42,
    }
    mismatches = {
        key: {"observed": runtime.get(key), "expected": value}
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Locked LS-KD protocol mismatch: {mismatches}")

    provenance = screening.get("provenance", {})
    expected_hashes = {
        "initial_student_sha256": sha256_file(args.initial_state),
        "teacher_config_sha256": sha256_file(args.teacher_config),
        "teacher_sha256": sha256_file(args.teacher_checkpoint),
        "split_sha256": sha256_file(args.split),
    }
    provenance_mismatches = {
        key: {"observed": provenance.get(key), "expected": value}
        for key, value in expected_hashes.items()
        if provenance.get(key) != value
    }
    if provenance_mismatches:
        raise ValueError(f"LS-KD provenance mismatch: {provenance_mismatches}")

    baseline = evaluation_candidate("pk_ce", args.baseline_eval)
    candidate = evaluation_candidate("logit_standardization_kd", args.candidate_eval)
    if baseline["samples"] != candidate["samples"]:
        raise ValueError("Baseline and LS-KD validation sample counts differ")
    if candidate["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError("Candidate evaluation did not use best_screening.pth")

    improves = tuple(candidate["selection_key"]) < tuple(baseline["selection_key"])
    winner = candidate if improves else baseline
    payload = {
        "task": "closed_set_identification",
        "partition": "validation_only",
        "test_loader_created": False,
        "selection_rule": "errors -> ordinary CE loss -> true-class margin",
        "status": "method_selected" if improves else "no_improvement",
        "winner": winner,
        "candidate_strictly_improves_matched_pk_ce": improves,
        "locked_protocol": expected,
        "provenance": expected_hashes,
        "candidates": [baseline, candidate],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not improves:
        print("STOP: LS-KD did not improve the matched PK-CE validation control.")


if __name__ == "__main__":
    main()
