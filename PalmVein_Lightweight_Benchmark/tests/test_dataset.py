from __future__ import annotations

import json
import unittest

from src.data import load_dataset_config, validate_calibration_manifest, validate_dataset


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_dataset_config()

    def test_split_integrity(self):
        report = validate_dataset(self.config, verify_images=True)
        self.assertEqual(report["counts"], {"train": 6672, "val": 834, "test": 834})
        self.assertEqual(report["classes"], 834)
        self.assertFalse(any(report["overlap"].values()))

    def test_calibration_is_training_only(self):
        with open(self.config["calibration_manifest"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        report = validate_calibration_manifest(self.config, manifest)
        self.assertEqual(report, {"valid": True, "count": 834, "train_only": True})


if __name__ == "__main__":
    unittest.main()
