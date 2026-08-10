#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MODEL_NAMES, PRIMARY_MODEL_NAMES
from src.models.factory import PRETRAINED_MODELS


def main():
    parser = argparse.ArgumentParser(description="Run the benchmark experiment matrix sequentially")
    parser.add_argument("--protocol", choices=("scratch", "pretrained"), required=True)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    models = list(PRIMARY_MODEL_NAMES) if args.models == ["all"] else args.models
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")
    if args.protocol == "pretrained":
        unavailable = [name for name in models if name not in PRETRAINED_MODELS]
        if unavailable:
            raise SystemExit(f"Pretrained is explicitly N/A for: {unavailable}")
    for model in models:
        for seed in args.seeds:
            command = [
                sys.executable, str(PROJECT_ROOT / "scripts/train.py"), "--model", model,
                "--protocol", args.protocol, "--seed", str(seed), "--device", args.device,
            ]
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
