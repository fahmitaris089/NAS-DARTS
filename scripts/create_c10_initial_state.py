#!/usr/bin/env python3
"""Create one hashed random C10 state shared by controlled ablations."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genotypes import dict_to_genotype  # noqa: E402
from model_eval import EvalNetwork  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C-init", type=int, default=None)
    parser.add_argument("--num-cells", type=int, default=None)
    parser.add_argument("--stem-downsample", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    config_path = Path(args.config)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() and not args.overwrite:
        if not manifest_path.exists():
            raise FileExistsError(
                f"Existing initial state lacks provenance manifest: {manifest_path}. "
                "Use --overwrite only after confirming it may be replaced."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "config_sha256": config_hash, "seed": args.seed,
            "C_init": args.C_init or manifest.get("C_init"),
            "num_cells": args.num_cells or manifest.get("num_cells"),
            "stem_downsample": args.stem_downsample or manifest.get("stem_downsample"),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Existing initial state provenance mismatch: {manifest} vs {expected}")
        state_hash = hashlib.sha256(output.read_bytes()).hexdigest()
        if manifest.get("state_sha256") != state_hash:
            raise ValueError("Existing initial state hash differs from its manifest")
        print(json.dumps({"output": str(output), "sha256": state_hash, "status": "reused"}, indent=2))
        return
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    reduction = cfg.get("reduction_indices")
    if isinstance(reduction, str):
        reduction = [int(x) for x in reduction.split(",") if x.strip()]
    c_init = args.C_init or int(cfg["C_init"])
    num_cells = args.num_cells or int(cfg["num_cells"])
    stem_downsample = args.stem_downsample or int(cfg.get("stem_downsample", 8))
    model = EvalNetwork(
        genotype=dict_to_genotype(cfg["genotype"]), C_init=c_init,
        num_cells=num_cells, num_classes=834, auxiliary=False,
        dropout=float(cfg.get("retrain_cfg", {}).get("dropout", 0.3)),
        stem_downsample=stem_downsample, reduction_indices=reduction,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    state_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "state_sha256": state_hash, "config": str(config_path),
        "config_sha256": config_hash, "seed": args.seed,
        "C_init": c_init, "num_cells": num_cells,
        "stem_downsample": stem_downsample,
        "reduction_indices": reduction, "num_classes": 834,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": state_hash, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
