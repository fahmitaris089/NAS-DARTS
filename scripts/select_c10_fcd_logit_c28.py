#!/usr/bin/env python3
"""Select FCD+logit KD against frozen validation-only C10 controls."""

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


def read_candidate(name: str, directory: Path) -> dict:
    result_path = directory / "screening_results.json"
    checkpoint = directory / "best_screening.pth"
    config_path = directory / "config.json"
    for path in (result_path, checkpoint, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    runtime_config = json.loads(config_path.read_text(encoding="utf-8"))
    if result.get("test_acc") is not None or result.get("test_loss") is not None:
        raise ValueError(f"{name} is not validation-only: {result_path}")
    if name != "baseline_arcface":
        kd_config = result.get("kd_config", {})
        if not kd_config.get("skip_test_evaluation"):
            raise ValueError(f"{name} did not record --skip-test-evaluation")

    key = result["best_screening_key"]
    return {
        "name": name,
        "directory": str(directory),
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "validation_errors": int(key["validation_errors"]),
        "validation_ce_loss": float(key["validation_ce_loss"]),
        "true_class_margin": -float(key["negative_true_class_margin"]),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(config_path),
        "provenance": result.get("provenance", {}),
        "kd_config": result.get("kd_config", {}),
        "runtime_config": runtime_config,
    }


def controlled_provenance(candidate: dict) -> dict:
    provenance = candidate["provenance"]
    return {
        key: provenance.get(key)
        for key in (
            "teacher_sha256",
            "teacher_config_sha256",
            "student_config_sha256",
            "split_sha256",
            "initial_student_sha256",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--fcd", required=True, type=Path)
    parser.add_argument("--fcd-logit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline = read_candidate("baseline_arcface", args.baseline)
    fcd = read_candidate("fcd_c28", args.fcd)
    fcd_logit = read_candidate("fcd_logit_c28", args.fcd_logit)

    fcd_provenance = controlled_provenance(fcd)
    logit_provenance = controlled_provenance(fcd_logit)
    if any(value is None for value in fcd_provenance.values()):
        raise ValueError("FCD control has incomplete controlled provenance")
    if fcd_provenance != logit_provenance:
        raise ValueError(
            "FCD and FCD+logit must share teacher, student, split, and initial-state hashes"
        )
    label_map_sha256 = fcd_logit["provenance"].get("label_map_sha256")
    if not label_map_sha256:
        raise ValueError("FCD+logit candidate did not record its deterministic label-map hash")

    matched_fields = (
        "epochs", "batch_size", "lr", "lr_min", "weight_decay",
        "warmup_epochs", "augmentation_policy", "train_sampler", "pk_p", "pk_k",
        "label_smoothing", "drop_path_prob", "cutout_length", "mixup_alpha",
        "cutmix_alpha", "icd_mode", "icd_bank_size", "icd_valid_steps",
        "icd_delta", "icd_gamma", "icd_sdc_start_epoch", "icd_sdc_weight",
        "icd_classification_weight", "scheduler", "amp", "freeze_bn",
        "student_dropout", "input_size", "seed", "num_workers",
    )
    mismatches = {
        key: [fcd["runtime_config"].get(key), fcd_logit["runtime_config"].get(key)]
        for key in matched_fields
        if fcd["runtime_config"].get(key) != fcd_logit["runtime_config"].get(key)
    }
    if mismatches:
        raise ValueError(f"FCD matched-control configuration differs: {mismatches}")

    logit_cfg = fcd_logit["kd_config"]
    locked = {
        "kd_method": logit_cfg.get("kd_method"),
        "icd_mode": logit_cfg.get("icd_mode"),
        "temperature": float(logit_cfg.get("temperature", -1)),
        "logit_kd_weight": float(logit_cfg.get("logit_kd_weight", -1)),
        "icd_logit_warmup_epochs": int(
            logit_cfg.get("icd_logit_warmup_epochs", -1)
        ),
    }
    expected = {
        "kd_method": "icd_compactness",
        "icd_mode": "fcd",
        "temperature": 20.0,
        "logit_kd_weight": 0.05,
        "icd_logit_warmup_epochs": 20,
    }
    if locked != expected:
        raise ValueError(f"FCD+logit protocol mismatch: {locked} != {expected}")

    candidates = [baseline, fcd, fcd_logit]
    for candidate in candidates:
        candidate["selection_key"] = [
            candidate["validation_errors"],
            candidate["validation_ce_loss"],
            -candidate["true_class_margin"],
        ]
        candidate.pop("provenance", None)
        candidate.pop("kd_config", None)
        candidate.pop("runtime_config", None)

    ordered = sorted(candidates, key=lambda item: tuple(item["selection_key"]))
    winner = ordered[0]
    improves_fcd = tuple(fcd_logit["selection_key"]) < tuple(fcd["selection_key"])
    status = (
        "method_selected"
        if winner["name"] == "fcd_logit_c28" and improves_fcd
        else "no_improvement"
    )
    payload = {
        "partition": "validation_only",
        "test_loader_created": False,
        "selection_rule": "errors -> ordinary CE loss -> true-class margin",
        "status": status,
        "winner": winner,
        "fcd_logit_strictly_improves_fcd": improves_fcd,
        "controlled_provenance": fcd_provenance,
        "label_map_sha256": label_map_sha256,
        "locked_logit_protocol": expected,
        "matched_control_fields": list(matched_fields),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status == "no_improvement":
        print("STOP: FCD+logit did not improve the frozen FCD C28 control.")


if __name__ == "__main__":
    main()
