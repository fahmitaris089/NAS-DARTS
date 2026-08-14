#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_calibration_manifest, load_dataset_config, validate_calibration_manifest, validate_dataset


def main():
    parser = argparse.ArgumentParser(description="Validate thesis split and create the train-only INT8 manifest")
    parser.add_argument(
        "--dataset-config", type=Path, default=Path("configs/dataset.json"),
        help="Dataset configuration JSON to validate and prepare.",
    )
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()
    config = load_dataset_config(args.dataset_config)
    validation = validate_dataset(config, verify_images=not args.skip_image_check)
    manifest = create_calibration_manifest(config)
    calibration = validate_calibration_manifest(config, manifest)
    print(json.dumps({"dataset": validation, "calibration": calibration}, indent=2))


if __name__ == "__main__":
    main()
