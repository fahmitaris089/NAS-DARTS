"""
McNemar's Test — Statistical significance between two models on the SAME test set.

Membandingkan dua model (arsitektur student SAMA, weights berbeda) untuk menentukan
apakah perbedaan akurasi signifikan secara statistik atau hanya noise.

McNemar menguji apakah dua model berbeda pada sampel yang sama:
  b = jumlah sampel: model A BENAR, model B SALAH
  c = jumlah sampel: model A SALAH, model B BENAR
  H0: kedua model punya error rate sama (b ≈ c)

Untuk b+c kecil (<25) pakai exact binomial test (lebih akurat dari chi-square).

Contoh:
  python3 mcnemar_test.py \
      --genotype nas_results/retrain_mobile_v2_C4_834cls/config.json \
      --c-init 4 \
      --weights-a knowledge_distilation/kd_results/run_v2C4_KD_t8_a0.3_e500/best_model.pth \
      --label-a "KD" \
      --weights-b knowledge_distilation/kd_results/run_v2C4_NOKD_e150/best_model.pth \
      --label-b "no-KD"
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nas_config import NUM_CLASSES, RETRAIN_CFG
from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from palm_vein_dataset import create_retrain_dataloaders
from utils import get_device


def load_genotype(path: Path):
    """Load genotype from a config.json (with 'genotype' key) or a genotype json."""
    with open(path) as f:
        data = json.load(f)
    if "genotype" in data:
        return dict_to_genotype(data["genotype"]), data.get("C_init")
    return dict_to_genotype(data), None


@torch.no_grad()
def get_predictions(model, loader, device):
    """Return array of predicted labels and true labels for the whole test set."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def build_and_load(genotype, c_init, num_cells, weights_path, device):
    model = EvalNetwork(
        genotype=genotype, C_init=c_init, num_cells=num_cells,
        num_classes=NUM_CLASSES, auxiliary=False, dropout=RETRAIN_CFG["dropout"],
    ).to(device)
    state = torch.load(weights_path, map_location=device)
    state = {k: v for k, v in state.items() if not k.startswith("_auxiliary_head")}
    model.load_state_dict(state, strict=False)
    return model


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) under Binomial(n, 0.5), two-sided
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def main():
    ap = argparse.ArgumentParser(description="McNemar's test between two models")
    ap.add_argument("--genotype", required=True, help="config.json (with genotype) or genotype json")
    ap.add_argument("--c-init", type=int, default=None)
    ap.add_argument("--num-cells", type=int, default=RETRAIN_CFG["num_cells"])
    ap.add_argument("--weights-a", required=True)
    ap.add_argument("--weights-b", required=True)
    ap.add_argument("--label-a", default="Model A")
    ap.add_argument("--label-b", default="Model B")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split-path", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = get_device()
    genotype, cfg_c_init = load_genotype(Path(args.genotype))
    c_init = args.c_init or cfg_c_init or RETRAIN_CFG["C_init"]
    print(f"Architecture: C_init={c_init}, num_cells={args.num_cells}")

    # Test loader (same split for both models)
    _, _, test_loader, info = create_retrain_dataloaders(
        data_dir=args.data_dir, split_path=args.split_path,
        batch_size=args.batch_size, num_workers=2, use_augmentation=False,
    )

    # Model A
    model_a = build_and_load(genotype, c_init, args.num_cells, args.weights_a, device)
    preds_a, labels = get_predictions(model_a, test_loader, device)

    # Model B (rebuild fresh, same architecture)
    model_b = build_and_load(genotype, c_init, args.num_cells, args.weights_b, device)
    preds_b, labels_b = get_predictions(model_b, test_loader, device)

    assert np.array_equal(labels, labels_b), "Test order mismatch between models!"

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    n = len(labels)
    acc_a = correct_a.mean()
    acc_b = correct_b.mean()

    # Contingency
    both_correct = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))   # b in McNemar
    b_only = int(np.sum(~correct_a & correct_b))   # c in McNemar
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    p_exact = exact_mcnemar_pvalue(a_only, b_only)

    print("\n" + "=" * 60)
    print("  McNemar's Test")
    print("=" * 60)
    print(f"  Test samples       : {n}")
    print(f"  {args.label_a:<18}: {acc_a*100:.2f}% ({int(correct_a.sum())}/{n})")
    print(f"  {args.label_b:<18}: {acc_b*100:.2f}% ({int(correct_b.sum())}/{n})")
    print(f"\n  Contingency table:")
    print(f"    both correct                 : {both_correct}")
    print(f"    {args.label_a} correct, {args.label_b} wrong : {a_only}  (b)")
    print(f"    {args.label_a} wrong, {args.label_b} correct : {b_only}  (c)")
    print(f"    both wrong                   : {both_wrong}")
    print(f"\n  Discordant pairs (b+c): {a_only + b_only}")
    print(f"  Exact McNemar p-value : {p_exact:.4f}")
    alpha = 0.05
    if p_exact < alpha:
        print(f"  Result: SIGNIFICANT at α={alpha} → perbedaan nyata, bukan noise.")
    else:
        print(f"  Result: NOT significant at α={alpha} → perbedaan dalam noise.")
    print("=" * 60)

    # Save
    out = {
        "label_a": args.label_a, "label_b": args.label_b,
        "n": n, "acc_a": float(acc_a), "acc_b": float(acc_b),
        "both_correct": both_correct, "a_only_b": a_only, "b_only_c": b_only,
        "both_wrong": both_wrong, "discordant": a_only + b_only,
        "p_value_exact": p_exact, "significant_at_0.05": bool(p_exact < alpha),
    }
    out_path = Path("mcnemar_result.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
