from __future__ import annotations

import unittest

import torch

from src.models import MODEL_NAMES, build_model, count_parameters


class ModelTests(unittest.TestCase):
    def test_all_adapted_outputs(self):
        sample = torch.randn(2, 3, 224, 224)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, num_classes=834)
                model.eval()
                with torch.inference_mode():
                    output = model(sample)
                self.assertEqual(tuple(output.shape), (2, 834))

    def test_ding_reference_parameter_envelopes(self):
        targets = {
            "ding_baseline": (351_000, 0.03),
            "ding_pw": (165_000, 0.03),
            "ding_pruned": (93_000, 0.05),
        }
        for name, (target, tolerance) in targets.items():
            with self.subTest(model=name):
                observed = count_parameters(build_model(name, num_classes=500, input_channels=1))
                self.assertLessEqual(abs(observed - target) / target, tolerance)

    def test_mnasnet_a1_torchvision_shape_equivalence(self):
        from torchvision.models import mnasnet1_0

        local = build_model("mnasnet_a1", num_classes=1000)
        reference = mnasnet1_0(weights=None)
        self.assertEqual(count_parameters(local), 4_383_312)
        self.assertEqual(
            [tuple(value.shape) for value in local.state_dict().values()],
            [tuple(value.shape) for value in reference.state_dict().values()],
        )

    def test_pretrained_is_explicitly_unavailable_for_reconstructions(self):
        for name in ("ding_baseline", "ding_pw", "ding_pruned", "pdarts_l005_c12_cells10"):
            with self.subTest(model=name):
                with self.assertRaises(ValueError):
                    build_model(name, pretrained=True)


if __name__ == "__main__":
    unittest.main()
