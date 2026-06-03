"""
DARTS Operations — Candidate Operations for Search Space
=========================================================
All ops follow:  ReLU → Conv → BN  (pre-activation, DARTS convention)
BN uses affine=False during search (more stable), affine=True for retrain.

Operations are quantization-friendly:
  - ReLU (not Swish/GELU)
  - No Squeeze-and-Excitation
  - Standard Conv / Depthwise-Separable Conv / Dilated Conv / Pooling
"""

import torch
import torch.nn as nn


# ─── Helpers ──────────────────────────────────────────────────────────────────

OPS = {
    'none':         lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: (
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine)
    ),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    # Experiment 2: MobileNet-style inverted residuals (XNNPACK-optimized on ARM)
    'mbconv3_3x3':  lambda C, stride, affine: MBConv(C, stride, expand_ratio=3, affine=affine),
    'mbconv6_3x3':  lambda C, stride, affine: MBConv(C, stride, expand_ratio=6, affine=affine),
}


# ─── Zero (no connection) ────────────────────────────────────────────────────

class Zero(nn.Module):
    """Output zeros — represents no connection (pruned edge)."""

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        # stride > 1: reduce spatial dims
        return x[:, :, ::self.stride, ::self.stride].mul(0.0)


# ─── Identity ────────────────────────────────────────────────────────────────

class Identity(nn.Module):
    """Skip connection (identity mapping)."""

    def forward(self, x):
        return x


# ─── Pooling + BN ────────────────────────────────────────────────────────────

class PoolBN(nn.Module):
    """Pooling → BatchNorm (learnable spatial reduction)."""

    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=False):
        super().__init__()
        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding,
                                     count_include_pad=False)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        return self.bn(self.pool(x))


# ─── Depthwise-Separable Convolution ─────────────────────────────────────────

class DilConv(nn.Module):
    """
    Dilated depthwise-separable convolution:
      ReLU → DepthwiseConv (dilated) → PointwiseConv 1×1 → BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=False):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """
    Stacked depthwise-separable convolution (2×):
      ReLU → DW → PW → BN → ReLU → DW → PW → BN
    Double-stack captures richer features with minimal param overhead.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.op = nn.Sequential(
            # First DW-PW
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            # Second DW-PW
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=1, padding=padding,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


# ─── Factorized Reduce ───────────────────────────────────────────────────────

class FactorizedReduce(nn.Module):
    """
    Reduce spatial dimension by 2× while keeping channel count.
    Used for skip_connect in reduction cells.
    Split input into two halves (offset by 1 pixel), each processed by
    Conv 1×1 stride 2, then concatenated → BN.
    """

    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        assert C_out % 2 == 0, f"C_out must be even, got {C_out}"
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        # Two offset views for richer representation
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)


# ─── Drop Path (Stochastic Depth) ────────────────────────────────────────────

class DropPath(nn.Module):
    """
    Stochastic depth: randomly drop entire path during training.
    Used during retrain (not search) for regularisation.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample random mask (batch dimension preserved)
        mask = torch.zeros(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob


# ─── MBConv (Inverted Residual — MobileNetV2 style) ────────────────────────────────────────────

class MBConv(nn.Module):
    """
    Inverted Residual Block (MobileNetV2-style):
      ReLU → PW expand → BN → ReLU → DW 3×3 → BN → PW project → BN

    expand_ratio=3: lighter (fewer FLOPs, XNNPACK-friendly depthwise on ARM)
    expand_ratio=6: richer capacity, same efficient structure

    Skip connection applied only when stride=1 (same spatial dims).
    Unlike SepConv (2× stacked), this is a single forward pass.
    """

    def __init__(self, C, stride, expand_ratio=3, affine=False):
        super().__init__()
        C_mid = C * expand_ratio
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C_mid, 1, bias=False),            # PW expand
            nn.BatchNorm2d(C_mid, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_mid, C_mid, 3, stride=stride,      # DW 3×3
                      padding=1, groups=C_mid, bias=False),
            nn.BatchNorm2d(C_mid, affine=affine),
            nn.Conv2d(C_mid, C, 1, bias=False),            # PW project
            nn.BatchNorm2d(C, affine=affine),
        )
        self.use_skip = (stride == 1)

    def forward(self, x):
        return x + self.op(x) if self.use_skip else self.op(x)
