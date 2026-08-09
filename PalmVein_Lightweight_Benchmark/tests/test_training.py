from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.ding import build_ding_pruned
from src.training.engine import run_training


def tiny_loaders():
    generator = torch.Generator().manual_seed(42)
    images = torch.randn(8, 3, 32, 32, generator=generator)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    dataset = TensorDataset(images, targets)
    return {name: DataLoader(dataset, batch_size=4, shuffle=name == "train") for name in ("train", "val", "test")}


def tiny_protocol(epochs: int):
    return {
        "protocol": "scratch", "epochs": epochs, "optimizer": "AdamW", "learning_rate": 1e-3,
        "weight_decay": 0.0, "warmup_epochs": 1, "warmup_start_factor": 0.1,
        "minimum_learning_rate": 1e-6, "label_smoothing": 0.0, "gradient_clip_norm": 1.0,
    }


class TrainingTests(unittest.TestCase):
    def test_forward_backward_optimizer_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = build_ding_pruned(4)
            result = run_training(
                model, tiny_loaders(), tiny_protocol(1), torch.device("cpu"),
                root / "checkpoints", root / "results", {"model": "ding_pruned", "protocol": "scratch", "seed": 42},
            )
            best = Path(result["best_checkpoint"])
            self.assertTrue(best.is_file())
            resumed = build_ding_pruned(4)
            second = run_training(
                resumed, tiny_loaders(), tiny_protocol(2), torch.device("cpu"),
                root / "checkpoints", root / "results", {"model": "ding_pruned", "protocol": "scratch", "seed": 42},
                resume=root / "checkpoints/last.pth",
            )
            self.assertIn("accuracy", second["test"])
            state = torch.load(root / "checkpoints/last.pth", map_location="cpu", weights_only=False)
            self.assertEqual(state["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
