from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.models import MODEL_NAMES, PRIMARY_MODEL_NAMES, build_model, count_parameters
from src.models.ding import (
    DING_BASELINE_SPECS,
    DING_PRUNED_SPECS,
    DING_PW_SPECS,
    DingPointwiseBlock,
)
from src.models.ampvnet import AMPVNetBottleneck
from src.models.mnasnet import MNASNET_A1_STAGES, SqueezeExcite
from src.models.ding_legacy import build_ding_pruned_legacy
from src.models.factory import PALMNET_VARIANTS, PRETRAINED_MODELS
from src.models.palmnet import (
    MBConvBlock,
    MobileNetV3Block,
    PAPER_VARIANT_CODES,
    ShuffleNetV2Block,
    build_palmnet,
    parse_variant_code,
)


class ModelTests(unittest.TestCase):
    def test_ampvnet_matches_figure_9_and_parameter_count(self):
        reference = build_model("ampvnet", num_classes=1100).eval()
        self.assertEqual(count_parameters(reference), 1_637_676)
        self.assertLess(abs(count_parameters(reference) - 1_610_000) / 1_610_000, 0.02)
        self.assertEqual(len(reference.stages), 4)
        self.assertTrue(
            all(isinstance(stage, AMPVNetBottleneck) for stage in reference.stages)
        )

        with torch.inference_mode():
            outputs = reference.forward_stages(torch.zeros(1, 3, 224, 224))
        self.assertEqual(
            [tuple(value.shape[1:]) for value in outputs],
            [
                (32, 56, 56),
                (64, 28, 28),
                (128, 14, 14),
                (256, 7, 7),
                (512, 3, 3),
            ],
        )

        adapted = build_model("ampvnet", num_classes=834)
        output = adapted(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (2, 834))
        output.square().mean().backward()
        with self.assertRaises(ValueError):
            build_model("ampvnet", pretrained=True)

    def test_all_adapted_outputs(self):
        sample = torch.randn(2, 3, 224, 224)
        for name in PRIMARY_MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, num_classes=834)
                model.eval()
                with torch.inference_mode():
                    output = model(sample)
                self.assertEqual(tuple(output.shape), (2, 834))

    def test_palmnet_parser_and_all_paper_variants(self):
        self.assertEqual(parse_variant_code("2413"), (2, 4, 1, 3))
        self.assertEqual(parse_variant_code("2411"), (2, 4, 1, 1))
        for invalid in ("2414", "241", "0000", "abcd"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_variant_code(invalid)

        for code in PAPER_VARIANT_CODES:
            with self.subTest(code=code):
                model = build_palmnet(
                    width_mult=0.5,
                    variant_code=code,
                    num_classes=834,
                    input_channels=3,
                ).eval()
                shuffle, mobile, mbconv, expansion = (int(value) for value in code)
                self.assertEqual(len(model.shuffle_stage), shuffle)
                self.assertEqual(len(model.mobilenetv3_stage), mobile)
                self.assertEqual(len(model.mbconv_stage), mbconv)
                self.assertEqual(model.spec.expansion_factor, expansion)
                self.assertTrue(all(isinstance(block, ShuffleNetV2Block) for block in model.shuffle_stage))
                self.assertTrue(all(isinstance(block, MobileNetV3Block) for block in model.mobilenetv3_stage))
                self.assertTrue(all(isinstance(block, MBConvBlock) for block in model.mbconv_stage))

    def test_palmnet_shapes_reference_mode_and_efficiency_ordering(self):
        model = build_model("palmnet_05x_2413", num_classes=834).eval()
        with torch.inference_mode():
            stages = model.forward_stages(torch.zeros(1, 3, 224, 224))
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual([tuple(value.shape[-2:]) for value in stages], [(56, 56), (28, 28), (14, 14), (7, 7), (7, 7)])
        self.assertEqual(tuple(output.shape), (1, 834))

        reference = build_palmnet(
            width_mult=0.5,
            variant_code="2413",
            num_classes=200,
            input_channels=1,
        ).eval()
        with torch.inference_mode():
            reference_output = reference(torch.zeros(1, 1, 224, 224))
        self.assertEqual(tuple(reference_output.shape), (1, 200))

        smaller = build_model("palmnet_05x_2411", num_classes=834)
        larger = build_model("palmnet_05x_2413", num_classes=834)
        self.assertLess(count_parameters(smaller), count_parameters(larger))

    def test_palmnet_forward_backward_checkpoint_and_metadata(self):
        model = build_model("palmnet_05x_2413", num_classes=4)
        output = model(torch.randn(2, 3, 64, 64))
        output.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        metadata = model.reconstruction_metadata()
        self.assertEqual(metadata["variant_code"], "2413")
        self.assertEqual(metadata["shuffle_blocks"], 2)
        self.assertEqual(metadata["mobilenetv3_blocks"], 4)
        self.assertEqual(metadata["mbconv_blocks"], 1)
        self.assertEqual(metadata["expansion_factor"], 3)
        self.assertEqual(metadata["reconstruction_status"], "paper-constrained independent reconstruction")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "palmnet.pth"
            torch.save(model.state_dict(), path)
            restored = build_model("palmnet_05x_2413", num_classes=4)
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)

    def test_palmnet_registry_and_pretrained_policy(self):
        self.assertIn("palmnet_05x_2413", PRIMARY_MODEL_NAMES)
        self.assertIn("palmnet_05x_2411", PRIMARY_MODEL_NAMES)
        self.assertNotIn("palmnet_05x_2412", PRIMARY_MODEL_NAMES)
        self.assertEqual(set(PALMNET_VARIANTS).issubset(MODEL_NAMES), True)
        self.assertTrue(set(PALMNET_VARIANTS).isdisjoint(PRETRAINED_MODELS))
        for name in PALMNET_VARIANTS:
            with self.subTest(model=name), self.assertRaises(ValueError):
                build_model(name, pretrained=True)

    def test_ding_matches_paper_constrained_six_block_topology(self):
        specifications = {
            "ding_baseline": DING_BASELINE_SPECS,
            "ding_pw": DING_PW_SPECS,
            "ding_pruned": DING_PRUNED_SPECS,
        }
        expected_channels = {
            "ding_baseline": [32, 32, 64, 64, 128, 128],
            "ding_pw": [32, 32, 64, 64, 128, 128],
            "ding_pruned": [22, 22, 44, 44, 89, 89],
        }
        for name, specs in specifications.items():
            with self.subTest(model=name):
                model = build_model(name, num_classes=834, input_channels=3).eval()
                self.assertEqual(len(model.blocks), 6)
                self.assertEqual(tuple(model.architecture_spec[1:]), tuple(specs[1:]))
                self.assertEqual(
                    [spec.out_channels for spec in model.architecture_spec],
                    expected_channels[name],
                )
                self.assertTrue(all(not spec.pool for spec in model.architecture_spec[-1:]))
                self.assertTrue(all(spec.pool for spec in model.architecture_spec[:5]))
                grouped = [
                    module for module in model.modules()
                    if isinstance(module, torch.nn.Conv2d) and module.groups != 1
                ]
                self.assertEqual(grouped, [])
                with torch.inference_mode():
                    outputs = model.forward_block_features(torch.zeros(1, 3, 224, 224))
                self.assertEqual(
                    [tuple(value.shape[-2:]) for value in outputs],
                    [(112, 112), (56, 56), (28, 28), (14, 14), (7, 7), (7, 7)],
                )

    def test_ding_pw_blocks_are_pointwise_standard_pointwise(self):
        for name in ("ding_pw", "ding_pruned"):
            with self.subTest(model=name):
                model = build_model(name, num_classes=834)
                self.assertTrue(all(isinstance(block, DingPointwiseBlock) for block in model.blocks[3:]))
                for block in model.blocks[3:]:
                    self.assertEqual(block.reduce.kernel_size, (1, 1))
                    self.assertEqual(block.spatial.kernel_size, (3, 3))
                    self.assertEqual(block.spatial.groups, 1)
                    self.assertEqual(block.expand.kernel_size, (1, 1))

    def test_ding_reference_input_and_classifier(self):
        for name in ("ding_baseline", "ding_pw", "ding_pruned"):
            with self.subTest(model=name):
                model = build_model(name, num_classes=500, input_channels=1).eval()
                with torch.inference_mode():
                    output = model(torch.zeros(1, 1, 224, 224))
                self.assertEqual(tuple(output.shape), (1, 500))

    def test_ding_forward_backward_and_checkpoint_round_trip(self):
        for name in ("ding_baseline", "ding_pw", "ding_pruned"):
            with self.subTest(model=name), TemporaryDirectory() as directory:
                model = build_model(name, num_classes=4)
                output = model(torch.randn(2, 3, 32, 32))
                output.square().mean().backward()
                self.assertTrue(
                    any(parameter.grad is not None for parameter in model.parameters())
                )
                path = Path(directory) / f"{name}.pth"
                torch.save(model.state_dict(), path)
                restored = build_model(name, num_classes=4)
                restored.load_state_dict(
                    torch.load(path, map_location="cpu", weights_only=True), strict=True
                )

    def test_migrated_ding_checkpoint_is_legacy_only(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "artifacts/checkpoints/legacy/scratch/ding_pruned_legacy_parameter_matched_v1/seed_42/best.pth"
        )
        if not path.exists():
            self.skipTest("Migrated legacy Ding checkpoint is not included in this checkout")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(
            state["metadata"]["model"], "ding_pruned_legacy_parameter_matched_v1"
        )
        legacy = build_ding_pruned_legacy(num_classes=834)
        legacy.load_state_dict(state["model_state"], strict=True)
        corrected = build_model("ding_pruned", num_classes=834)
        with self.assertRaises(RuntimeError):
            corrected.load_state_dict(state["model_state"], strict=True)

    def test_mnasnet_b1_torchvision_shape_equivalence(self):
        from torchvision.models import mnasnet1_0

        local = build_model("mnasnet_b1_torchvision", num_classes=1000).eval()
        reference = mnasnet1_0(weights=None).eval()
        self.assertEqual(count_parameters(local), 4_383_312)
        self.assertEqual(
            [tuple(value.shape) for value in local.state_dict().values()],
            [tuple(value.shape) for value in reference.state_dict().values()],
        )
        local.load_state_dict(
            {
                local_name: reference_value
                for (local_name, _), (_, reference_value) in zip(
                    local.state_dict().items(), reference.state_dict().items()
                )
            },
            strict=True,
        )
        sample = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            torch.testing.assert_close(local(sample), reference(sample), rtol=0, atol=0)

    def test_mnasnet_a1_matches_paper_stage_definition(self):
        expected = (
            ("ds", 1, 16, 1, 1, 3, 0.0),
            ("ir", 6, 24, 2, 2, 3, 0.0),
            ("ir", 3, 40, 3, 2, 5, 0.25),
            ("ir", 6, 80, 4, 2, 3, 0.0),
            ("ir", 6, 112, 2, 1, 3, 0.25),
            ("ir", 6, 160, 3, 2, 5, 0.25),
            ("ir", 6, 320, 1, 1, 3, 0.0),
        )
        observed = tuple(
            (spec.block, spec.expansion, spec.channels, spec.repeats, spec.stride, spec.kernel, spec.se_ratio)
            for spec in MNASNET_A1_STAGES
        )
        self.assertEqual(observed, expected)

        model = build_model("mnasnet_a1", num_classes=834).eval()
        with torch.inference_mode():
            outputs = model.forward_stages(torch.zeros(1, 3, 224, 224))
        self.assertEqual(
            [tuple(value.shape[1:]) for value in outputs],
            [
                (32, 112, 112),
                (16, 112, 112),
                (24, 56, 56),
                (40, 28, 28),
                (80, 14, 14),
                (112, 14, 14),
                (160, 7, 7),
                (320, 7, 7),
            ],
        )
        self.assertEqual(sum(isinstance(module, SqueezeExcite) for module in model.modules()), 8)

    def test_mnasnet_a1_parameter_counts_official_bn_and_timm_shape_oracle(self):
        import timm

        for classes, expected_parameters in ((1000, 3_887_038), (834, 3_674_392)):
            with self.subTest(classes=classes):
                local = build_model("mnasnet_a1", num_classes=classes).eval()
                reference = timm.create_model(
                    "semnasnet_100", pretrained=False, num_classes=classes
                ).eval()
                self.assertEqual(count_parameters(local), expected_parameters)
                self.assertEqual(
                    [tuple(value.shape) for value in local.state_dict().values()],
                    [tuple(value.shape) for value in reference.state_dict().values()],
                )
                batch_norms = [
                    module for module in local.modules() if isinstance(module, torch.nn.BatchNorm2d)
                ]
                self.assertTrue(batch_norms)
                self.assertTrue(all(module.eps == 1e-3 for module in batch_norms))
                self.assertTrue(all(module.momentum == 0.01 for module in batch_norms))

    def test_mnasnet_pretrained_policy(self):
        with self.assertRaises(ValueError):
            build_model("mnasnet_a1", pretrained=True)
        model = build_model("mnasnet_b1_torchvision", num_classes=834, pretrained=False)
        self.assertEqual(count_parameters(model), 4_170_666)

    def test_mnasnet_a1_forward_backward_and_checkpoint_round_trip(self):
        model = build_model("mnasnet_a1", num_classes=834)
        output = model(torch.randn(1, 3, 224, 224))
        output.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "a1.pth"
            torch.save(model.state_dict(), path)
            restored = build_model("mnasnet_a1", num_classes=834)
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)

    def test_migrated_legacy_checkpoint_belongs_to_b1(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "artifacts/checkpoints/pretrained/mnasnet_b1_torchvision/seed_42/best.pth"
        )
        if not path.exists():
            self.skipTest("Migrated legacy checkpoint is not included in this checkout")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(state["metadata"]["model"], "mnasnet_b1_torchvision")
        b1 = build_model("mnasnet_b1_torchvision", num_classes=834)
        b1.load_state_dict(state["model_state"], strict=True)
        a1 = build_model("mnasnet_a1", num_classes=834)
        with self.assertRaises(RuntimeError):
            a1.load_state_dict(state["model_state"], strict=True)

    def test_pretrained_is_explicitly_unavailable_for_reconstructions(self):
        for name in ("mnasnet_a1", "ding_baseline", "ding_pw", "ding_pruned", "pdarts_l005_c12_cells10"):
            with self.subTest(model=name):
                with self.assertRaises(ValueError):
                    build_model(name, pretrained=True)


if __name__ == "__main__":
    unittest.main()
