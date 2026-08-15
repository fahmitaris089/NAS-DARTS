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
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    reduction = cfg.get("reduction_indices")
    if isinstance(reduction, str):
        reduction = [int(x) for x in reduction.split(",") if x.strip()]
    c_init = args.C_init or int(cfg["C_init"])
    num_cells = args.num_cells or int(cfg["num_cells"])
    stem_downsample = args.stem_downsample or int(cfg.get("stem_downsample", 8))
    architecture = {
        "genotype": cfg["genotype"], "C_init": c_init, "num_cells": num_cells,
        "stem_downsample": stem_downsample, "reduction_indices": reduction,
        "num_classes": 834,
    }
    architecture_hash = hashlib.sha256(
        json.dumps(architecture, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = EvalNetwork(
        genotype=dict_to_genotype(cfg["genotype"]), C_init=c_init,
        num_cells=num_cells, num_classes=834, auxiliary=False,
        dropout=float(cfg.get("retrain_cfg", {}).get("dropout", 0.3)),
        stem_downsample=stem_downsample, reduction_indices=reduction,
    )
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() and not args.overwrite:
        if not manifest_path.exists():
            raise FileExistsError(
                f"Existing initial state lacks provenance manifest: {manifest_path}. "
                "Use --overwrite only after confirming it may be replaced."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_semantics = {
            "seed": args.seed, "C_init": c_init, "num_cells": num_cells,
            "stem_downsample": stem_downsample, "reduction_indices": reduction,
            "num_classes": 834,
        }
        semantic_mismatch = {
            key: (manifest.get(key), value) for key, value in expected_semantics.items()
            if manifest.get(key) != value
        }
        if semantic_mismatch:
            raise ValueError(f"Existing initial-state architecture mismatch: {semantic_mismatch}")
        state_hash = hashlib.sha256(output.read_bytes()).hexdigest()
        if manifest.get("state_sha256") != state_hash:
            raise ValueError("Existing initial state hash differs from its manifest")
        state = torch.load(output, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "student" in state:
            state = state["student"]
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Existing initial state is not strictly compatible with the requested architecture"
            ) from error
        status = "reused" if manifest.get("config_sha256") == config_hash else "reused_semantic_match"
        compatible_hashes = set(manifest.get("compatible_config_sha256", []))
        compatible_hashes.update(filter(None, [manifest.get("config_sha256"), config_hash]))
        manifest["compatible_config_sha256"] = sorted(compatible_hashes)
        manifest["architecture_sha256"] = architecture_hash
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({
            "output": str(output), "sha256": state_hash, "status": status,
            "original_config_sha256": manifest.get("config_sha256"),
            "current_config_sha256": config_hash,
            "strict_state_dict_compatible": True,
        }, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    state_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "state_sha256": state_hash, "config": str(config_path),
        "config_sha256": config_hash, "seed": args.seed,
        "C_init": c_init, "num_cells": num_cells,
        "stem_downsample": stem_downsample,
        "reduction_indices": reduction, "num_classes": 834,
        "architecture_sha256": architecture_hash,
        "compatible_config_sha256": [config_hash],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": state_hash, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
