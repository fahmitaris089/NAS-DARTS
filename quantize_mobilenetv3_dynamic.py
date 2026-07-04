#!/usr/bin/env python3
"""Try dynamic INT8 quantization for MobileNetV3Small as fallback."""
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

fp32 = Path('Teacher/training_results/MobileNetV3Small/model_benchmark.onnx')
int8_dyn = fp32.with_name('model_benchmark_int8_dynamic.onnx')

print(f"Quantizing {fp32.name} with dynamic INT8...")
quantize_dynamic(
    model_input=str(fp32),
    model_output=str(int8_dyn),
    weight_type=QuantType.QInt8,
)
print(f"✓ Dynamic INT8 saved: {int8_dyn.name}")
print(f"  Size: {int8_dyn.stat().st_size / 1e6:.3f} MB")
