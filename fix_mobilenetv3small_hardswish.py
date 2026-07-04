#!/usr/bin/env python3
"""
Fix MobileNetV3Small INT8 by forcing Hardswish decomposition during ONNX export.

The issue: PyTorch exports MobileNetV3Small with fused HardSwish op (single node),
but exports MobileNetV3Large with decomposed HardSigmoid+Mul. ONNX Runtime INT8
quantization fails on fused HardSwish but works on decomposed version.

Solution: Replace Hardswish with manual decomposition before export.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "Teacher"))
from model_factory import create_model

class DecomposedHardswish(nn.Module):
    """Hardswish with explicit decomposition: x * hardsigmoid(x)"""
    def forward(self, x):
        # hardsigmoid(x) = clip(x + 3, 0, 6) / 6
        # hardswish(x) = x * hardsigmoid(x)
        return x * torch.nn.functional.hardsigmoid(x)

def replace_hardswish_recursive(module):
    """Replace all Hardswish modules with decomposed version."""
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, DecomposedHardswish())
        else:
            replace_hardswish_recursive(child)

def main():
    # Load trained model
    model, _ = create_model('MobileNetV3Small', num_classes=834)
    weights = torch.load('Teacher/training_results/MobileNetV3Small/best_model.pth', 
                        map_location='cpu')
    model.load_state_dict(weights, strict=True)
    model.eval()
    
    # Replace Hardswish with decomposed version
    print("Replacing Hardswish with decomposed HardSigmoid+Mul...")
    replace_hardswish_recursive(model)
    
    # Export to ONNX
    dummy = torch.randn(1, 3, 224, 224)
    output_path = Path('Teacher/training_results/MobileNetV3Small/model_benchmark_decomposed.onnx')
    
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        do_constant_folding=True,
    )
    
    # Verify decomposition
    import onnx
    m = onnx.load(str(output_path))
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    
    print(f"\n✓ Exported: {output_path.name}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.3f} MB")
    print(f"  HardSwish ops: {ops.get('HardSwish', 0)} (should be 0)")
    print(f"  HardSigmoid ops: {ops.get('HardSigmoid', 0)} (should be >0)")
    print(f"  Mul ops: {ops.get('Mul', 0)} (should match HardSigmoid)")
    
    if ops.get('HardSwish', 0) > 0:
        print("\n⚠️  WARNING: Still has fused HardSwish ops!")
        return False
    
    if ops.get('HardSigmoid', 0) == 0:
        print("\n⚠️  WARNING: No HardSigmoid found (decomposition failed)!")
        return False
    
    print("\n✓ Hardswish successfully decomposed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
