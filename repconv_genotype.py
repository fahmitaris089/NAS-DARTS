"""
Utility: ubah genotype berbasis MBConv menjadi berbasis RepConv.

Dipakai untuk Fase-1 (probe murah, tanpa search ulang): pertahankan topologi
sel hasil search yang sudah ada, tapi ganti operator MBConv (3 conv + ReLU,
tak bisa fuse) dengan RepConv (multi-branch saat train, fuse jadi 1 conv saat
inference). Lalu retrain.py melatih ulang dari awal dengan operator baru.

Pemetaan operator:
    mbconv3_3x3 -> rep_conv_3x3
    mbconv6_3x3 -> rep_conv_3x3   (RepConv tidak punya expand ratio; 3x3 default)
    sep_conv_5x5 / dil_conv_5x5 -> rep_conv_5x5  (pertahankan receptive field)
    sep_conv_3x3 / dil_conv_3x3 -> rep_conv_3x3
    skip_connect / none / pooling -> dibiarkan (struktur DAG tetap)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REP_MAP = {
    "mbconv3_3x3": "rep_conv_3x3",
    "mbconv6_3x3": "rep_conv_3x3",
    "sep_conv_3x3": "rep_conv_3x3",
    "dil_conv_3x3": "rep_conv_3x3",
    "sep_conv_5x5": "rep_conv_5x5",
    "dil_conv_5x5": "rep_conv_5x5",
}


def convert_edges(edges):
    out = []
    for op_name, src in edges:
        out.append([REP_MAP.get(op_name, op_name), src])
    return out


def convert_genotype(geno: dict) -> dict:
    return {
        "normal": convert_edges(geno["normal"]),
        "normal_concat": list(geno["normal_concat"]),
        "reduce": convert_edges(geno["reduce"]),
        "reduce_concat": list(geno["reduce_concat"]),
    }


def load_genotype_from(path: Path) -> dict:
    """Terima file genotype langsung ATAU config.json yang memuat key 'genotype'."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "genotype" in data:
        return data["genotype"]
    if "normal" in data:
        return data
    raise ValueError(f"Tidak menemukan genotype di {path}")


def main():
    ap = argparse.ArgumentParser(description="Convert MBConv genotype -> RepConv genotype")
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="genotype.json atau config.json sumber")
    ap.add_argument("--out", dest="out", type=Path, required=True,
                    help="path genotype RepConv hasil konversi")
    args = ap.parse_args()

    geno = load_genotype_from(args.inp)
    rep = convert_genotype(geno)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    # Ringkasan perubahan
    from collections import Counter
    before = Counter(op for op, _ in geno["normal"] + geno["reduce"])
    after = Counter(op for op, _ in rep["normal"] + rep["reduce"])
    print(f"Genotype RepConv ditulis ke: {args.out}")
    print(f"  Operator sebelum: {dict(before)}")
    print(f"  Operator sesudah: {dict(after)}")


if __name__ == "__main__":
    main()
