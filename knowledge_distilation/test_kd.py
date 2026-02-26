"""
Quick smoke test untuk modul KD.
Jalankan dari folder Student/:
    python knowledge_distilation/test_kd.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# ─── 1. kd_config ─────────────────────────────────────────────────────────────
print("=" * 50)
print("TEST 1: kd_config")
print("=" * 50)
from knowledge_distilation.kd_config import KD_CFG, print_config
print_config(KD_CFG)
print("  kd_config OK\n")

# ─── 2. kd_loss ───────────────────────────────────────────────────────────────
print("=" * 50)
print("TEST 2: kd_loss — forward pass dengan dummy tensor")
print("=" * 50)
from knowledge_distilation.kd_loss import HintonKDLoss, SoftCEKDLoss, get_kd_loss

logits_s = torch.randn(4, 834, requires_grad=True)
logits_t = torch.randn(4, 834)
targets  = torch.randint(0, 834, (4,))

# Hinton KD
hinton = get_kd_loss("hinton", temperature=4.0, alpha=0.3)
loss, bd = hinton(logits_s, logits_t, targets)
assert loss.item() > 0, "Loss harus positif"
print(f"  HintonKDLoss  → total={bd['loss_total']:.4f}  ce={bd['loss_ce']:.4f}  kd={bd['loss_kd']:.4f}")

# Soft CE
softce = get_kd_loss("soft_ce", alpha=0.3)
loss2, bd2 = softce(logits_s, logits_t, targets)
assert loss2.item() > 0
print(f"  SoftCEKDLoss  → total={bd2['loss_total']:.4f}  ce={bd2['loss_ce']:.4f}")

# Backward test
loss.backward()
print("  backward() OK (gradien mengalir)")
print("  kd_loss OK\n")

# ─── 3. kd_train imports ──────────────────────────────────────────────────────
print("=" * 50)
print("TEST 3: kd_train — import semua dependency")
print("=" * 50)
from genotypes import dict_to_genotype, Genotype
from model_eval import EvalNetwork
from palm_vein_dataset import create_retrain_dataloaders
print("  genotypes, model_eval, palm_vein_dataset OK")

# Build student dari genotype dummy untuk test arsitektur
dummy_genotype = Genotype(
    normal=[
        ("dil_conv_3x3", 1), ("dil_conv_3x3", 0),
        ("dil_conv_3x3", 1), ("dil_conv_3x3", 2),
        ("dil_conv_3x3", 2), ("dil_conv_3x3", 1),
        ("dil_conv_3x3", 4), ("dil_conv_3x3", 1),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ("dil_conv_3x3", 1), ("dil_conv_3x3", 0),
        ("skip_connect", 1), ("dil_conv_3x3", 2),
        ("dil_conv_3x3", 0), ("skip_connect", 1),
        ("dil_conv_3x3", 3), ("dil_conv_3x3", 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)

student = EvalNetwork(
    genotype=dummy_genotype, C_init=8, num_cells=8,
    num_classes=834, auxiliary=False, dropout=0.3,
)
n_params = sum(p.numel() for p in student.parameters()) / 1e3
print(f"  EvalNetwork built OK  |  {n_params:.1f}K params")

# Forward pass dummy
student.eval()
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    out = student(x)
assert out.shape == (2, 834), f"Shape salah: {out.shape}"
print(f"  EvalNetwork forward OK  |  output shape: {out.shape}")
print("  kd_train imports OK\n")

# ─── 4. KD loss dengan output student ────────────────────────────────────────
print("=" * 50)
print("TEST 4: KD loss end-to-end dengan student output")
print("=" * 50)
teacher_logits = torch.randn(2, 834)
labels         = torch.randint(0, 834, (2,))
criterion      = HintonKDLoss(temperature=4.0, alpha=0.3)
loss, bd       = criterion(out, teacher_logits, labels)
print(f"  Loss OK  total={bd['loss_total']:.4f}  ce={bd['loss_ce']:.4f}  kd={bd['loss_kd']:.4f}")
print("  End-to-end OK\n")

print("=" * 50)
print("  SEMUA TEST PASSED")
print("=" * 50)
