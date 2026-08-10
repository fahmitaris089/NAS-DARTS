from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from src.deployment.onnx_utils import compare_outputs, validate_onnx_file
from src.models.ding import build_ding_baseline, build_ding_pruned, build_ding_pw
from src.models.palmnet import build_palmnet


DING_BUILDERS = {
    "ding_baseline": build_ding_baseline,
    "ding_pw": build_ding_pw,
    "ding_pruned": build_ding_pruned,
}
PALMNET_BUILDERS = {
    "palmnet_05x_2413": lambda classes: build_palmnet(
        width_mult=0.5, variant_code="2413", num_classes=classes
    ),
    "palmnet_05x_2411": lambda classes: build_palmnet(
        width_mult=0.5, variant_code="2411", num_classes=classes
    ),
}


class OnnxTests(unittest.TestCase):
    def test_export_checker_and_runtime_parity(self):
        for name, builder in DING_BUILDERS.items():
            with self.subTest(model=name):
                model = builder(4).eval()
                sample = torch.randn(2, 3, 32, 32)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "smoke.onnx"
                    torch.onnx.export(
                        model, sample, str(path), input_names=["input"], output_names=["logits"],
                        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=13,
                    )
                    validate_onnx_file(path)
                    with torch.inference_mode():
                        expected = model(sample)
                    report = compare_outputs(
                        expected,
                        path,
                        sample.numpy().astype(np.float32),
                        atol=1e-4,
                        rtol=1e-3,
                    )
                    self.assertTrue(report["allclose"])

    def test_palmnet_export_checker_and_runtime_parity(self):
        for name, builder in PALMNET_BUILDERS.items():
            with self.subTest(model=name):
                model = builder(4).eval()
                sample = torch.randn(1, 3, 64, 64)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "palmnet.onnx"
                    torch.onnx.export(
                        model,
                        sample,
                        str(path),
                        input_names=["input"],
                        output_names=["logits"],
                        opset_version=13,
                    )
                    validate_onnx_file(path)
                    with torch.inference_mode():
                        expected = model(sample)
                    report = compare_outputs(
                        expected,
                        path,
                        sample.numpy().astype(np.float32),
                        atol=1e-4,
                        rtol=1e-3,
                    )
                    self.assertTrue(report["allclose"])

    def test_static_qdq_int8_smoke(self):
        class Reader(CalibrationDataReader):
            def __init__(self, arrays):
                self.arrays = iter(arrays)

            def get_next(self):
                try:
                    return {"input": next(self.arrays)}
                except StopIteration:
                    return None

        for name, builder in DING_BUILDERS.items():
            with self.subTest(model=name):
                model = builder(4).eval()
                sample = np.random.default_rng(42).standard_normal(
                    (1, 3, 32, 32)
                ).astype(np.float32)
                with tempfile.TemporaryDirectory() as directory:
                    fp32 = Path(directory) / "fp32.onnx"
                    int8 = Path(directory) / "int8.onnx"
                    torch.onnx.export(
                        model,
                        torch.from_numpy(sample),
                        str(fp32),
                        input_names=["input"],
                        output_names=["logits"],
                        opset_version=13,
                    )
                    quantize_static(
                        str(fp32),
                        str(int8),
                        Reader([sample, sample * 0.5]),
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=True,
                    )
                    validate_onnx_file(int8)
                    from src.deployment.onnx_utils import create_session

                    output = create_session(int8).run(None, {"input": sample})[0]
                    self.assertEqual(tuple(output.shape), (1, 4))

    def test_palmnet_static_qdq_int8_smoke(self):
        class Reader(CalibrationDataReader):
            def __init__(self, arrays):
                self.arrays = iter(arrays)

            def get_next(self):
                try:
                    return {"input": next(self.arrays)}
                except StopIteration:
                    return None

        for name, builder in PALMNET_BUILDERS.items():
            with self.subTest(model=name):
                model = builder(4).eval()
                sample = np.random.default_rng(42).standard_normal((1, 3, 64, 64)).astype(np.float32)
                with tempfile.TemporaryDirectory() as directory:
                    fp32 = Path(directory) / "fp32.onnx"
                    int8 = Path(directory) / "int8.onnx"
                    torch.onnx.export(
                        model,
                        torch.from_numpy(sample),
                        str(fp32),
                        input_names=["input"],
                        output_names=["logits"],
                        opset_version=13,
                    )
                    quantize_static(
                        str(fp32),
                        str(int8),
                        Reader([sample, sample * 0.5]),
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=True,
                    )
                    validate_onnx_file(int8)
                    from src.deployment.onnx_utils import create_session

                    output = create_session(int8).run(None, {"input": sample})[0]
                    self.assertEqual(tuple(output.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
