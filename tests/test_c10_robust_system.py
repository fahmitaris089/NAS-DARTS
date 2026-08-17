import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from palm_input_preprocessing import (
    apply_input_profile_pil,
    preprocess_pil_to_imagenet_chw,
    robust_percentile_unit,
)
from palm_vein_dataset import create_retrain_dataloaders, get_transforms
from scripts.evaluate_c10_robustness import ValidationVariantDataset
from consistency_training import js_consistency_loss

import sys

NAS = Path(__file__).resolve().parents[1] / "Eksperimen_Hardware_Aware_PDARTS" / "src" / "nas"
if str(NAS) not in sys.path:
    sys.path.insert(0, str(NAS))
from adaface import replace_linear_with_arcface  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


class RobustInputProfileTest(unittest.TestCase):
    def test_percentile_formula_and_low_dynamic_range_are_deterministic(self):
        image = np.array([[10, 20], [30, 250]], dtype=np.uint8)
        first = robust_percentile_unit(image)
        second = robust_percentile_unit(image)
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

        flat = robust_percentile_unit(np.full((8, 8), 101, dtype=np.uint8))
        np.testing.assert_array_equal(flat, np.zeros_like(flat))

    def test_pil_and_onnx_preprocessing_share_profile(self):
        source = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
        image = Image.fromarray(source)
        profiled = apply_input_profile_pil(
            image.resize((16, 16), Image.Resampling.BILINEAR),
            "robust_percentile_v1",
        )
        expected_unit = np.asarray(profiled, dtype=np.float32) / 255.0
        actual = preprocess_pil_to_imagenet_chw(
            image, 16, "robust_percentile_v1",
        )
        restored = actual[0] * 0.229 + 0.485
        np.testing.assert_allclose(restored, expected_unit, atol=1e-6, rtol=0)

        pytorch_tensor = get_transforms(
            "val", input_size=16, input_profile="robust_percentile_v1",
        )(image)
        np.testing.assert_allclose(
            pytorch_tensor.numpy(), actual, atol=1e-6, rtol=0,
        )


class RobustTrainingComponentsTest(unittest.TestCase):
    def test_two_view_transform_returns_two_tensors_without_flip(self):
        transform = get_transforms(
            "train", input_size=32, use_augmentation=True,
            augmentation_policy="v4_robust_light",
            input_profile="robust_percentile_v1",
            consistency_mode="js_two_view",
        )
        views = transform(Image.fromarray(np.full((32, 32), 127, dtype=np.uint8)))
        self.assertEqual(len(views), 2)
        self.assertEqual(tuple(views[0].shape), (3, 32, 32))
        self.assertEqual(tuple(views[1].shape), (3, 32, 32))
        self.assertNotIn("HorizontalFlip", repr(transform))

    def test_js_loss_is_finite_and_zero_for_identical_logits(self):
        logits = torch.randn(3, 834, requires_grad=True)
        zero = js_consistency_loss(logits, logits, temperature=4.0)
        self.assertAlmostEqual(float(zero.detach()), 0.0, places=6)
        other = torch.randn(3, 834, requires_grad=True)
        loss = js_consistency_loss(logits, other, temperature=4.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(other.grad).all())

    def test_screening_loader_does_not_create_test_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for subject in ("1", "2"):
                (root / subject).mkdir()
                for index in range(1, 4):
                    Image.fromarray(np.full((16, 16), 50 * index, dtype=np.uint8)).save(
                        root / subject / f"{subject}_{index}.bmp"
                    )
            split = {
                "subjects": ["1", "2"],
                "train": [["1", "1_1.bmp"], ["2", "2_1.bmp"]],
                "val": [["1", "1_2.bmp"], ["2", "2_2.bmp"]],
                "test": [["1", "1_3.bmp"], ["2", "2_3.bmp"]],
            }
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            _, _, test_loader, info = create_retrain_dataloaders(
                data_dir=root, split_path=split_path, batch_size=2,
                num_workers=0, include_test=False,
                input_profile="robust_percentile_v1",
            )
            self.assertIsNone(test_loader)
            self.assertFalse(info["test_loader_created"])
            self.assertIsNone(info["test_size"])

    def test_corruption_variant_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.bmp"
            Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(path)
            dataset = ValidationVariantDataset(
                [(path, 0)], ("rotation_p15", "rotation", 15.0),
                "robust_percentile_v1", input_size=32,
            )
            first, _ = dataset[0]
            second, _ = dataset[0]
            torch.testing.assert_close(first, second, rtol=0, atol=0)


class StemPoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        genotype_payload = json.loads(
            (ROOT / "nas_results/search_hwint8_l0.20/genotype_final.json").read_text(
                encoding="utf-8"
            )
        )
        cls.genotype = dict_to_genotype(genotype_payload)

    def build(self, pool):
        model = EvalNetwork(
            genotype=self.genotype, C_init=10, num_cells=8, num_classes=834,
            auxiliary=False, dropout=0.3, stem_downsample=8,
            reduction_indices=[2, 5], stem_pool=pool,
        )
        replace_linear_with_arcface(
            model, num_classes=834, m=0.5, s=64.0, num_subcenters=1,
        )
        return model

    def test_max_avg_are_state_compatible_and_keep_parameter_budget(self):
        maximum = self.build("max")
        average = self.build("avg")
        average.load_state_dict(maximum.state_dict(), strict=True)
        maximum_params = sum(parameter.numel() for parameter in maximum.parameters())
        average_params = sum(parameter.numel() for parameter in average.parameters())
        self.assertEqual(maximum_params, 585250)
        self.assertEqual(average_params, maximum_params)
        with torch.inference_mode():
            self.assertEqual(tuple(average(torch.zeros(1, 3, 224, 224)).shape), (1, 834))


if __name__ == "__main__":
    unittest.main()
