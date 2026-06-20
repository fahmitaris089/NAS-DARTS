"""
Verifikasi: RepConvBN multi-branch (train) == fused single conv (deploy).

Fusi re-parameterization HARUS exact secara matematis (dalam toleransi float),
kalau tidak akurasi model berubah saat deploy. Test ini mengacak parameter BN
(gamma, beta, running_mean, running_var) supaya benar-benar menguji matematika
fusi, bukan kasus trivial BN-identity.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from operations import RepConvBN, fuse_reparam_model


def _randomize_bn(module):
    """Isi BN dengan statistik & affine acak realistis (bukan identity)."""
    for m in module.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.5)
            m.running_var.uniform_(0.5, 1.5)   # harus positif
            if m.affine:
                m.weight.data.uniform_(0.5, 1.5)
                m.bias.data.normal_(0, 0.3)


def _check(C, kernel_size, stride, affine, tol=1e-4):
    torch.manual_seed(0)
    block = RepConvBN(C, kernel_size=kernel_size, stride=stride, affine=affine)
    # Acak bobot conv juga
    for m in block.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.1)
    _randomize_bn(block)
    block.eval()

    x = torch.randn(2, C, 16, 16)

    with torch.no_grad():
        y_multi = block(x)          # jalur multi-branch
        block.switch_to_deploy()    # collapse → single conv
        y_fused = block(x)          # jalur fused

    max_abs = (y_multi - y_fused).abs().max().item()
    assert y_multi.shape == y_fused.shape, f"shape mismatch {y_multi.shape} vs {y_fused.shape}"
    ok = max_abs < tol
    print(f"  C={C:>3} k={kernel_size} stride={stride} affine={str(affine):>5} "
          f"| out={tuple(y_fused.shape)} | max|Δ|={max_abs:.2e} | {'OK' if ok else 'FAIL'}")
    assert ok, f"Fusi tidak exact: max|Δ|={max_abs:.2e} >= {tol}"


def main():
    print("=" * 70)
    print("Test fusi RepConvBN (multi-branch == fused single conv)")
    print("=" * 70)
    cases = []
    for C in [4, 8, 16]:
        for k in [3, 5]:
            for stride in [1, 2]:
                for affine in [True, False]:
                    cases.append((C, k, stride, affine))
    for c in cases:
        _check(*c)

    # Test integrasi: fuse_reparam_model pada modul bersarang
    print("\nTest fuse_reparam_model (nested):")
    torch.manual_seed(1)
    net = torch.nn.Sequential(
        RepConvBN(8, kernel_size=3, stride=1, affine=True),
        RepConvBN(8, kernel_size=3, stride=2, affine=True),
    )
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.1)
    _randomize_bn(net)
    net.eval()
    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        y_before = net(x)
        _, n = fuse_reparam_model(net)
        y_after = net(x)
    d = (y_before - y_after).abs().max().item()
    print(f"  fused {n} blok | max|Δ|={d:.2e} | {'OK' if d < 1e-4 else 'FAIL'}")
    assert n == 2 and d < 1e-4

    print("\nSEMUA TEST LULUS — fusi exact, aman untuk deploy.")


if __name__ == "__main__":
    main()
