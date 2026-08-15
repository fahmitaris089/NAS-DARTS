#!/usr/bin/env python3
"""Validation-only complementarity diagnosis and bounded C10 weight soup."""

import argparse
import csv
import itertools
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_center_relation import sha256_file  # noqa: E402
from kd_config import KDConfig  # noqa: E402
from kd_train import load_student  # noqa: E402
from palm_vein_dataset import (PalmVeinDataset, build_image_list, build_label_map,
                               get_transforms, load_split)  # noqa: E402


def logger():
    value = logging.getLogger("validation_soup")
    value.handlers.clear(); value.addHandler(logging.StreamHandler()); value.setLevel(logging.INFO)
    return value


@torch.no_grad()
def logits_for(model, loader, device):
    model.eval(); chunks, labels = [], []
    for images, target in loader:
        chunks.append(model(images.to(device)).cpu()); labels.append(target)
    return torch.cat(chunks), torch.cat(labels)


def metrics(logits, labels):
    predictions = logits.argmax(1)
    true = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone(); masked.scatter_(1, labels[:, None], float("-inf"))
    return {"errors": int(predictions.ne(labels).sum()),
            "loss": float(F.cross_entropy(logits, labels)),
            "margin": float((true - masked.max(1).values).mean())}


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    model.eval()
    batchnorm = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    for module in batchnorm:
        module.reset_running_stats(); module.train()
    for images, _ in loader:
        model(images.to(device))
    model.eval()


def weighted_state(states, weights):
    result = {}
    for key in states[0]:
        values = [state[key] for state in states]
        if torch.is_floating_point(values[0]):
            result[key] = sum(weight * value.float() for weight, value in zip(weights, values)).to(values[0].dtype)
        else:
            result[key] = values[0].clone()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/c10_error_audit.json")
    parser.add_argument("--output-dir", default="results/diagnostics/c10_complementarity_seed42")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    selected_names = {"c10_pk_ce", "c10_center_scratch", "c10_hybrid_scratch"}
    entries = [entry for entry in config["students"] if entry["name"] in selected_names]
    if len(entries) != 3:
        raise ValueError(f"Expected PK-CE, Center, Hybrid entries; found {[x['name'] for x in entries]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(config["split_path"]); label_map = build_label_map(split["subjects"])
    train_samples = build_image_list(config["data_dir"], split["train"], label_map)
    val_samples = build_image_list(config["data_dir"], split["val"], label_map)
    kwargs = {"batch_size": int(config["batch_size"]), "shuffle": False,
              "num_workers": int(config["num_workers"])}
    train_loader = DataLoader(PalmVeinDataset(
        train_samples, get_transforms("val", int(config["input_size"]))), **kwargs)
    val_loader = DataLoader(PalmVeinDataset(
        val_samples, get_transforms("val", int(config["input_size"]))), **kwargs)
    models, states, individual = [], [], []
    for entry in entries:
        model = load_student(KDConfig(student_config_path=entry["config"],
                                     student_weights=entry["weights"], num_classes=834),
                             device, logger())
        logits, labels = logits_for(model, val_loader, device)
        individual.append({"name": entry["name"], **metrics(logits, labels)})
        models.append(model); states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
    ensemble_logits = sum(logits_for(model, val_loader, device)[0] for model in models) / len(models)
    ensemble = metrics(ensemble_logits, labels)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    payload = {"partition": "validation_only", "test_loader_created": False,
               "individual": individual, "average_logits": ensemble,
               "soup_attempted": ensemble["errors"] == 0, "soups": []}
    if ensemble["errors"] == 0:
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        combinations = [weights for weights in itertools.product(grid, repeat=3)
                        if abs(sum(weights) - 1.0) < 1e-9]
        for weights in combinations:
            candidate = load_student(KDConfig(student_config_path=entries[0]["config"],
                                               student_weights=entries[0]["weights"], num_classes=834),
                                     device, logger())
            candidate.load_state_dict(weighted_state(states, weights), strict=True)
            recalibrate_bn(candidate, train_loader, device)
            candidate_logits, _ = logits_for(candidate, val_loader, device)
            result = {"weights": list(weights), **metrics(candidate_logits, labels)}
            payload["soups"].append(result)
        payload["soups"].sort(key=lambda row: (row["errors"], row["loss"], -row["margin"]))
        best = payload["soups"][0]
        candidate = load_student(KDConfig(student_config_path=entries[0]["config"],
                                           student_weights=entries[0]["weights"], num_classes=834),
                                 device, logger())
        candidate.load_state_dict(weighted_state(states, best["weights"]), strict=True)
        recalibrate_bn(candidate, train_loader, device)
        torch.save(candidate.state_dict(), output / "best_validation_soup.pth")
        payload["best_soup"] = best
    payload["checkpoint_hashes"] = {entry["name"]: sha256_file(entry["weights"]) for entry in entries}
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["soups"]:
        with (output / "soup_grid.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload["soups"][0])); writer.writeheader(); writer.writerows(payload["soups"])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
