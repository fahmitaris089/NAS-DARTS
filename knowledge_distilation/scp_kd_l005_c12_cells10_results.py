#!/usr/bin/env python3
"""Copy KD experiment result folders from Vast.ai to this Mac via scp."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


EXPERIMENT_FOLDERS = [
    "kd_L0.10_C8_stem8_t20_a03_lr1e4_e160_nw0",
    "kd_L0.10_C8_stem8_t20_a07_lr1e4_e160_nw0",
    "kd_L0.20_C8_stem8_t10_a05_lr1e4_e160_nw0",
    "kd_L0.20_C8_stem8_t20_a05_lr1e4_e160_nw0",
    "kd_L0.20_C8_stem8_t30_a05_lr1e4_e160_nw0",
    "kd_L0.20_C8_stem8_t20_a03_lr1e4_e160_nw0",
    "kd_L0.20_C8_stem8_t20_a07_lr1e4_e160_nw0"
    # "kd_L0.05_C12_cells10_stem8_t10_a05_lr1e4_e160_nw0",
    # "kd_L0.05_C12_cells10_stem8_t30_a05_lr1e4_e160_nw0",
    # "kd_L0.05_C12_cells10_stem8_t20_a03_lr1e4_e160_nw0",
    # "kd_L0.05_C12_cells10_stem8_t20_a07_lr1e4_e160_nw0",
    # "kd_L0.10_C8_stem8_t10_a05_lr1e4_e160_nw0",
    # "kd_L0.10_C8_stem8_t20_a05_lr1e4_e160_nw0",
    # "kd_L0.10_C8_stem8_t30_a05_lr1e4_e160_nw0",
    # "kd_L0.05_C8_stem8_t10_a05_lr1e4_e160_nw0",
    # "kd_L0.05_C8_stem8_t20_a05_lr1e4_e160_nw0",
    # "kd_L0.05_C8_stem8_t30_a05_lr1e4_e160_nw0",
    # "kd_L0.05_C8_stem8_t20_a03_lr1e4_e160_nw0",
    # "kd_L0.05_C8_stem8_t20_a07_lr1e4_e160_nw0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loop scp for the listed KD result folders."
    )
    parser.add_argument("--host", default="47.186.21.5", help="Vast.ai host/IP.")
    parser.add_argument("--user", default="root", help="SSH username.")
    parser.add_argument("--port", default=57561, type=int, help="SSH/scp port.")
    parser.add_argument(
        "--remote-base",
        default="/workspace/NAS-DARTS/knowledge_distilation/kd_results",
        help="Remote directory containing KD result folders.",
    )
    parser.add_argument(
        "--local-dest",
        default="~/Downloads/NAS-DARTS/knowledge_distilation/kd_results_new",
        help="Local destination directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scp commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dest = Path(args.local_dest).expanduser()
    local_dest.mkdir(parents=True, exist_ok=True)

    for folder in EXPERIMENT_FOLDERS:
        remote_path = f"{args.user}@{args.host}:{args.remote_base}/{folder}"
        cmd = ["scp", "-P", str(args.port), "-r", remote_path, str(local_dest)]

        print("Running:", " ".join(cmd), flush=True)
        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)

    print(f"Done. Copied {len(EXPERIMENT_FOLDERS)} folders to {local_dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
