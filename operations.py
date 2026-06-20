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
import torch.nn.functional as F


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
    # Experiment 5: re-parameterizable conv (RepVGG/DBB-style) — multi-branch at
    # train time, fuses to a SINGLE dense 3x3 conv at inference. Kills operator
    # fragmentation on edge CPUs (one high-arithmetic-intensity kernel, no extra
    # Add/BN ops after fusion). Variants differ only by kernel size of main branch.
    'rep_conv_3x3': lambda C, stride, affine: RepConvBN(C, kernel_size=3, stride=stride, affine=affine),
    'rep_conv_5x5': lambda C, stride, affine: RepConvBN(C, kernel_size=5, stride=stride, affine=affine),
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


# ─── RepConvBN (Re-parameterizable Conv — RepVGG/DBB style) ────────────────────

class RepConvBN(nn.Module):
    """
    Re-parameterizable convolution block.

    TRAIN TIME (multi-branch, richer gradients):
        out = ReLU( BN(conv_kxk(x)) + BN(conv_1x1(x)) + [BN(x) if stride==1] )

    INFERENCE TIME (after .switch_to_deploy()):
        out = ReLU( fused_conv_kxk(x) )           # ONE dense conv + bias

    Why this fixes edge latency:
      - The three parallel branches share the SAME input and have NO non-linearity
        before the sum, so they fuse EXACTLY into a single k×k conv (RepVGG trick).
      - A single dense conv has high arithmetic intensity → maps to one efficient
        ARM/NEON kernel, unlike depthwise-separable MBConv (PW→DW→PW = 3 convs that
        cannot fuse because of internal ReLUs).
      - Eliminates the per-edge Add/BN plumbing at inference.

    IMPORTANT: ReLU is applied AFTER the (fused) conv — opposite of the pre-activation
    (ReLU-first) convention used by the other ops in this file. This is mandatory:
    the branch sum must stay linear for fusion to be exact.

    Args:
        C:           channels (C_in == C_out)
        kernel_size: main branch kernel (3 or 5). 1x1 + identity branches always added.
        stride:      1 (normal cell) or 2 (reduction cell). Identity branch only at stride 1.
        affine:      BN affine (True for retrain/deploy; False during search).
    """

    def __init__(self, C, kernel_size=3, stride=1, affine=True):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.C = C
        self.kernel_size = kernel_size
        self.stride = stride
        self.affine = affine
        self.padding = kernel_size // 2
        self.deploy = False

        # Main k×k branch
        self.conv_k = nn.Conv2d(C, C, kernel_size, stride=stride,
                                padding=self.padding, bias=False)
        self.bn_k = nn.BatchNorm2d(C, affine=affine)

        # 1×1 branch
        self.conv_1 = nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False)
        self.bn_1 = nn.BatchNorm2d(C, affine=affine)

        # Identity branch (only when spatial dims preserved)
        self.has_identity = (stride == 1)
        self.bn_id = nn.BatchNorm2d(C, affine=affine) if self.has_identity else None

        self.relu = nn.ReLU(inplace=False)
        self.fused_conv = None  # populated by switch_to_deploy()

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(self, x):
        if self.deploy and self.fused_conv is not None:
            return self.relu(self.fused_conv(x))
        out = self.bn_k(self.conv_k(x)) + self.bn_1(self.conv_1(x))
        if self.has_identity:
            out = out + self.bn_id(x)
        return self.relu(out)

    # ── Fusion math ──────────────────────────────────────────────────────────
    def _fuse_conv_bn(self, conv_weight, bn):
        """Fold a (conv, BN) pair into an equivalent (weight, bias)."""
        device = conv_weight.device
        mean = bn.running_mean
        var = bn.running_var
        std = torch.sqrt(var + bn.eps)
        if bn.affine:
            gamma = bn.weight
            beta = bn.bias
        else:
            gamma = torch.ones(bn.num_features, device=device)
            beta = torch.zeros(bn.num_features, device=device)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_w = conv_weight * t
        fused_b = beta - mean * gamma / std
        return fused_w, fused_b

    def _pad_to_k(self, weight):
        """Center-pad a smaller kernel (1×1 or k') up to the main kernel size."""
        if weight.shape[-1] == self.kernel_size:
            return weight
        pad = (self.kernel_size - weight.shape[-1]) // 2
        return F.pad(weight, [pad, pad, pad, pad])

    def _identity_weight(self, device):
        """Identity branch expressed as a 1×1 conv with an identity kernel."""
        w = torch.zeros(self.C, self.C, 1, 1, device=device)
        for i in range(self.C):
            w[i, i, 0, 0] = 1.0
        return w

    def get_equivalent_kernel_bias(self):
        """Return the single fused (weight, bias) for a k×k conv with bias."""
        device = self.conv_k.weight.device

        wk, bk = self._fuse_conv_bn(self.conv_k.weight, self.bn_k)

        w1, b1 = self._fuse_conv_bn(self.conv_1.weight, self.bn_1)
        w1 = self._pad_to_k(w1)

        w = wk + w1
        b = bk + b1

        if self.has_identity:
            wid, bid = self._fuse_conv_bn(self._identity_weight(device), self.bn_id)
            w = w + self._pad_to_k(wid)
            b = b + bid

        return w, b

    @torch.no_grad()
    def switch_to_deploy(self):
        """Collapse all branches into a single conv for inference/export."""
        if self.deploy:
            return
        w, b = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            self.C, self.C, self.kernel_size, stride=self.stride,
            padding=self.padding, bias=True,
        ).to(w.device)
        self.fused_conv.weight.data.copy_(w)
        self.fused_conv.bias.data.copy_(b)
        # Drop training-time branches to free memory / avoid double export
        for attr in ("conv_k", "bn_k", "conv_1", "bn_1", "bn_id"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.has_identity = False
        self.deploy = True


def fuse_reparam_model(model):
    """Walk a model and switch every RepConvBN into deploy (fused) mode.

    Call this on the EvalNetwork after loading trained weights, before ONNX
    export or latency benchmarking, so the multi-branch blocks collapse to
    single convs. Returns the (in-place modified) model for convenience.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, RepConvBN) and not m.deploy:
            m.switch_to_deploy()
            n += 1
    return model, n
