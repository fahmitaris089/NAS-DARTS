#!/usr/bin/env python3
"""Evaluate one frozen checkpoint without biometric verification metrics."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
NAS = ROOT / "Eksperimen_Hardware_Aware_PDARTS" / "src" / "nas"
sys.path[:0] = [str(ROOT), str(NAS)]
from adaface import replace_linear_with_adaface, replace_linear_with_arcface  # noqa: E402
from genotypes import dict_to_genotype  # noqa: E402
from model_eval import EvalNetwork  # noqa: E402
from palm_vein_dataset import (PalmVeinDataset, build_image_list, build_label_map,
                               get_transforms, load_split)  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_model(config, checkpoint, device):
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    if "genotype" not in cfg and cfg.get("student_config_path"):
        # Experiment configs may be produced on Windows and later audited on
        # POSIX systems.  Backslashes are ordinary filename characters on
        # POSIX, so normalize the persisted project-relative path first.
        student_config = Path(str(cfg["student_config_path"]).replace("\\", "/"))
        if not student_config.is_absolute():
            student_config = ROOT / student_config
        cfg = json.loads(student_config.read_text(encoding="utf-8"))
    reduction = cfg.get("reduction_indices")
    if isinstance(reduction, str):
        reduction = [int(value) for value in reduction.split(",") if value.strip()]
    model = EvalNetwork(
        genotype=dict_to_genotype(cfg["genotype"]), C_init=int(cfg["C_init"]),
        num_cells=int(cfg["num_cells"]), num_classes=834, auxiliary=False,
        dropout=float(cfg.get("retrain_cfg", {}).get("dropout", 0.3)),
        stem_downsample=int(cfg.get("stem_downsample", 8)), reduction_indices=reduction,
        stem_pool=cfg.get("stem_pool", "max"),
    )
    mode = cfg.get("loss_mode", "ce")
    if mode == "adaface":
        replace_linear_with_adaface(
            model, num_classes=834, m=float(cfg.get("adaface_m", 0.4)),
            h=float(cfg.get("adaface_h", 0.333)), s=float(cfg.get("adaface_s", 64)),
            t_alpha=float(cfg.get("adaface_t_alpha", 0.01)),
        )
    elif mode in {"arcface", "subcenter_arcface"}:
        replace_linear_with_arcface(
            model, num_classes=834, m=float(cfg.get("arcface_margin", 0.5)),
            s=float(cfg.get("arcface_scale", 64)),
            num_subcenters=int(cfg.get("arcface_subcenters", 2 if mode.startswith("subcenter") else 1)),
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "student" in state:
        state = state["student"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--partition", choices=["val", "test"], default="val")
    parser.add_argument("--acknowledge-observed-test", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.partition == "test" and not args.acknowledge_observed_test:
        parser.error("test evaluation requires --acknowledge-observed-test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(args.config, args.checkpoint, device)
    split = load_split(args.split_path)
    label_map = build_label_map(split["subjects"])
    samples = build_image_list(args.data_dir, split[args.partition], label_map)
    dataset = PalmVeinDataset(
        samples,
        get_transforms(
            "val", 224, input_profile=cfg.get("input_profile", "legacy"),
        ),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    rows, total_loss, cursor = [], 0.0, 0
    for images, labels in loader:
        images, labels_device = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels_device, reduction="sum").item()
        probs = torch.softmax(logits, 1)
        confidence, predictions = probs.max(1)
        true_logits = logits.gather(1, labels_device[:, None]).squeeze(1)
        masked = logits.clone(); masked.scatter_(1, labels_device[:, None], float("-inf"))
        margins = true_logits - masked.max(1).values
        for index in range(labels.numel()):
            path, identity = samples[cursor + index]
            rows.append({"sample_id": str(path), "identity": identity,
                         "true_class": int(labels[index]),
                         "prediction": int(predictions[index]),
                         "correct": int(predictions[index].cpu() == labels[index]),
                         "confidence": float(confidence[index]),
                         "true_class_margin": float(margins[index])})
        cursor += labels.numel()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    correct = sum(row["correct"] for row in rows)
    result = {
        "task": "closed_set_identification", "partition": args.partition,
        "test_previously_observed_acknowledged": bool(args.acknowledge_observed_test),
        "checkpoint_selection_occurred_on_test": False,
        "correct": correct, "samples": len(rows), "accuracy": correct / len(rows),
        "ordinary_ce_loss": total_loss / len(rows),
        "mean_true_class_margin": sum(row["true_class_margin"] for row in rows) / len(rows),
        "checkpoint": args.checkpoint, "checkpoint_sha256": sha(args.checkpoint),
        "config": args.config, "config_sha256": sha(args.config),
        "input_profile": cfg.get("input_profile", "legacy"),
        "stem_pool": cfg.get("stem_pool", "max"),
        "consistency_mode": cfg.get("consistency_mode", "none"),
        "split_sha256": sha(args.split_path), "reported_metrics": ["accuracy_crr", "correct_total"],
        "excluded_metrics": ["eer", "far", "frr", "biometric_auc"],
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
