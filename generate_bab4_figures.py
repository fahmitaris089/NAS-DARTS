from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import fill

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/matplotlib-bab4")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures_bab4"
OUT_DIR.mkdir(exist_ok=True)

RAW_PREPROCESS_SAMPLE = Path("/Users/fahmitaris/Downloads/palm vein dataset/SCUT_PV_V1_raw10/1/1_1.bmp")
PREPROCESS_SAMPLE = Path("/Users/fahmitaris/Downloads/NAS-DARTS-TEMP/preprocessed_results/1/1_1.bmp")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


COLORS = {
    "blue": "#2F6DB3",
    "light_blue": "#DCEAF7",
    "green": "#2E8B57",
    "light_green": "#DFF1E6",
    "orange": "#D9822B",
    "light_orange": "#F8E6D2",
    "red": "#B23B3B",
    "light_red": "#F4DADA",
    "gray": "#5E6673",
    "light_gray": "#F1F3F5",
    "purple": "#6F5CC2",
    "light_purple": "#E7E2F8",
    "ink": "#1F2933",
}


def save_figure(fig: plt.Figure, name: str) -> None:
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, w, h, text, fc, ec=None, fontsize=10, weight="normal"):
    ec = ec or COLORS["ink"]
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        weight=weight,
    )
    return patch


def arrow(ax, start, end, color=None, lw=1.8, rad=0.0, text=None, text_offset=(0, 0)):
    color = color or COLORS["gray"]
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(
            mx,
            my,
            text,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.86),
            color=COLORS["ink"],
        )


def contrast_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def estimate_roi_box(arr: np.ndarray, crop_size: int = 384) -> tuple[int, int, int, int]:
    threshold = max(18.0, float(np.percentile(arr, 35)))
    mask = arr > threshold
    ys, xs = np.nonzero(mask)
    height, width = arr.shape
    side = min(crop_size, height, width)
    if len(xs) == 0:
        return (width - side) // 2, (height - side) // 2, side, side

    weights = np.maximum(arr[ys, xs].astype(np.float32) - float(arr.min()), 1.0)
    cx = float(np.average(xs, weights=weights))
    cy = float(np.average(ys, weights=weights))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x1 = max(0, min(x1, width - side))
    y1 = max(0, min(y1, height - side))
    return x1, y1, side, side


def figure_4_2_preprocessing_example():
    if not RAW_PREPROCESS_SAMPLE.exists() or not PREPROCESS_SAMPLE.exists():
        print("Skip gambar_4_2: sample raw/preprocessed tidak ditemukan.")
        return

    raw = np.array(Image.open(RAW_PREPROCESS_SAMPLE).convert("L"))
    preprocessed = np.array(Image.open(PREPROCESS_SAMPLE).convert("L"))
    mask = raw > max(25.0, float(np.percentile(raw, 45)))
    x1, y1, w, h = estimate_roi_box(raw)
    crop = raw[y1 : y1 + h, x1 : x1 + w]

    fig = plt.figure(figsize=(14, 4.9))
    flow_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    flow_ax.set_xlim(0, 14)
    flow_ax.set_ylim(0, 5)
    flow_ax.axis("off")

    labels = [
        "Citra NIR\nasli",
        "Mask biner\nforeground",
        "Bounding box\nROI",
        "Crop ROI",
        "Input model\n224 x 224",
    ]
    xs = [0.45, 3.15, 5.85, 8.55, 11.25]
    for i, (x, label) in enumerate(zip(xs, labels)):
        box(flow_ax, (x, 3.95), 2.05, 0.62, label, "#F8FAFC", COLORS["ink"], 9, "bold")
        if i < len(xs) - 1:
            arrow(flow_ax, (x + 2.05, 4.26), (xs[i + 1], 4.26), COLORS["gray"], lw=1.5)

    image_lefts = [0.035, 0.228, 0.421, 0.614, 0.807]
    image_w = 0.16
    image_y = 0.23
    image_h = 0.53
    axes = [fig.add_axes([left, image_y, image_w, image_h]) for left in image_lefts]

    axes[0].imshow(contrast_for_display(raw), cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(contrast_for_display(raw), cmap="gray", vmin=0, vmax=1)
    axes[2].add_patch(Rectangle((x1, y1), w, h, fill=False, edgecolor="#00D084", linewidth=2.2))
    axes[2].scatter([x1 + w / 2], [y1 + h / 2], s=14, color="#246BFE")
    axes[3].imshow(contrast_for_display(crop), cmap="gray", vmin=0, vmax=1)
    axes[4].imshow(contrast_for_display(preprocessed), cmap="gray", vmin=0, vmax=1)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#CBD2D9")

    flow_ax.text(
        7.0,
        0.35,
        (
            "Contoh menggunakan pasangan file kelas 1: raw 1_1.bmp dan hasil preprocessing 1_1.bmp. "
            "Mask, bounding box, dan crop divisualisasikan dari citra raw untuk menjelaskan alur; "
            "panel akhir memakai file preprocessing aktual."
        ),
        ha="center",
        va="center",
        fontsize=8.7,
        color=COLORS["gray"],
    )
    flow_ax.set_title("Contoh Hasil Preprocessing Citra Palm Vein", weight="bold", pad=12)
    save_figure(fig, "gambar_4_2_contoh_preprocessing_palm_vein")


def load_raspi_models():
    with open(ROOT / "raspi_benchmark_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["model"]: item for item in data["models"]}


def retrain_accuracy(model_dir: str) -> float:
    with open(ROOT / "nas_results" / model_dir / "test_results.json", "r", encoding="utf-8") as f:
        return json.load(f)["accuracy"] * 100.0


def figure_4_3_lut_objective():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    box(ax, (0.4, 4.5), 2.3, 1.1, "Search space\noperator kandidat", COLORS["light_blue"], COLORS["blue"], 10, "bold")
    box(ax, (0.4, 1.5), 2.3, 1.35, "Raspberry Pi\nINT8 LUT\ncost(op)", COLORS["light_orange"], COLORS["orange"], 10, "bold")
    box(ax, (3.3, 4.5), 2.2, 1.1, "Probabilitas\noperator\nsoftmax(alpha)", COLORS["light_purple"], COLORS["purple"], 10, "bold")
    box(ax, (3.3, 1.5), 2.2, 1.35, "Expected latency\nsum p(op) x cost(op)", COLORS["light_green"], COLORS["green"], 10, "bold")
    box(ax, (6.2, 4.5), 2.1, 1.1, "Classification\nloss\nL_CE", COLORS["light_gray"], COLORS["gray"], 10, "bold")
    box(ax, (6.2, 1.45), 2.1, 1.45, "Objective NAS\nL = L_CE +\nlambda x latency", "#FFF3CD", "#B08900", 10, "bold")
    box(ax, (9.1, 3.1), 2.45, 1.25, "Update alpha\npilih genotype\nhardware-aware", "#E8F5F1", COLORS["green"], 10, "bold")

    arrow(ax, (2.7, 5.05), (3.3, 5.05), COLORS["blue"])
    arrow(ax, (4.4, 4.5), (4.4, 2.85), COLORS["purple"], rad=0.0)
    arrow(ax, (2.7, 2.2), (3.3, 2.2), COLORS["orange"])
    arrow(ax, (5.5, 2.2), (6.2, 2.2), COLORS["green"])
    arrow(ax, (7.25, 4.5), (7.25, 2.9), COLORS["gray"])
    arrow(ax, (8.3, 2.2), (9.1, 3.7), COLORS["green"], rad=-0.15)

    ax.text(
        0.45,
        0.45,
        "LUT yang dipakai: latency_lut_pi_int8_corrected.json. Penalti latency aktif untuk lambda > 0.",
        fontsize=9,
        color=COLORS["gray"],
    )
    ax.set_title("Skema Integrasi Latency Lookup Table ke Objective NAS", weight="bold", pad=14)
    save_figure(fig, "gambar_4_3_lut_objective_nas")


def figure_4_4_lut_cost_bar():
    with open(ROOT / "latency_lut_pi_int8_corrected.json", "r", encoding="utf-8") as f:
        lut = json.load(f)["cost"]

    items = sorted(lut.items(), key=lambda item: item[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    colors = [COLORS["red"] if v == max(values) else COLORS["blue"] for v in values]
    bars = ax.barh(labels, values, color=colors, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Cost operator INT8 Raspberry Pi (ms)")
    ax.set_title("Cost Operator berdasarkan LUT INT8 Raspberry Pi", weight="bold")
    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.5f}", va="center", fontsize=8.5)
    ax.set_xlim(0, max(values) * 1.22 if max(values) > 0 else 1)
    ax.text(
        0.0,
        -0.85,
        "Sumber: latency_lut_pi_int8_corrected.json; QDQ-boundary floor sudah dikoreksi.",
        fontsize=8.5,
        color=COLORS["gray"],
    )
    save_figure(fig, "gambar_4_4_bar_cost_operator_lut_int8")


def plot_genotype_cell(ax, cell, title):
    coords = {
        0: (0.5, 3.4),
        1: (0.5, 1.6),
        2: (2.6, 3.7),
        3: (4.2, 2.85),
        4: (5.8, 2.0),
        5: (7.4, 1.15),
        "out": (9.3, 2.45),
    }
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0.2, 4.6)
    ax.axis("off")
    ax.set_title(title, weight="bold", pad=8)

    node_fc = "#F8FAFC"
    for n in [0, 1, 2, 3, 4, 5]:
        label = f"node_{n}"
        if n == 0:
            label += "\n(c_k-2)"
        elif n == 1:
            label += "\n(c_k-1)"
        box(ax, (coords[n][0] - 0.42, coords[n][1] - 0.28), 0.84, 0.56, label, node_fc, COLORS["ink"], 8)

    box(ax, (coords["out"][0] - 0.6, coords["out"][1] - 0.3), 1.2, 0.6, "concat\n2,3,4,5", COLORS["light_blue"], COLORS["blue"], 8, "bold")

    op_short = {
        "skip_connect": "skip",
        "rep_conv_3x3": "rep3",
        "rep_conv_5x5": "rep5",
    }
    op_color = {
        "skip_connect": COLORS["blue"],
        "rep_conv_3x3": COLORS["green"],
        "rep_conv_5x5": COLORS["orange"],
    }

    for idx, (op, src) in enumerate(cell):
        target = 2 + idx // 2
        rad = 0.12 if idx % 2 == 0 else -0.12
        sx, sy = coords[src]
        tx, ty = coords[target]
        label_offsets = {
            0: (0.00, 0.25),
            1: (0.00, -0.25),
            2: (0.00, 0.25),
            3: (0.00, -0.25),
            4: (-0.28, -0.34),
            5: (-0.38, 0.34),
            6: (0.00, -0.28),
            7: (0.00, 0.25),
        }
        label_offset = label_offsets[idx]
        if target == 5 and src == 3:
            label_offset = (0.55, -0.65)
        arrow(
            ax,
            (sx + 0.42, sy),
            (tx - 0.42, ty),
            color=op_color.get(op, COLORS["gray"]),
            lw=1.7,
            rad=rad,
            text=op_short.get(op, op),
            text_offset=label_offset,
        )

    for n in [2, 3, 4, 5]:
        sx, sy = coords[n]
        ox, oy = coords["out"]
        arrow(ax, (sx + 0.42, sy), (ox - 0.6, oy), color="#9AA5B1", lw=1.1, rad=0.04)


def figure_4_5_genotype():
    with open(ROOT / "nas_results/search_hwint8_l0.20/genotype_final.json", "r", encoding="utf-8") as f:
        genotype = json.load(f)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.8))
    plot_genotype_cell(axes[0], genotype["normal"], "Normal Cell, lambda=0.20")
    plot_genotype_cell(axes[1], genotype["reduce"], "Reduction Cell, lambda=0.20")

    legend_ax = axes[1]
    legend_ax.text(
        0.2,
        0.33,
        "Label edge: skip = skip_connect, rep3 = rep_conv_3x3, rep5 = rep_conv_5x5",
        fontsize=9,
        color=COLORS["gray"],
    )
    fig.suptitle("Visual Genotype Arsitektur Terpilih (lambda=0.20)", fontsize=14, weight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.35)
    save_figure(fig, "gambar_4_5_genotype_lambda_0_20")


def figure_4_6_genotype_operator_distribution():
    from collections import Counter

    lambdas = ["0.0", "0.05", "0.10", "0.20"]
    counts_by_lambda = {}
    all_ops = set()
    for lam in lambdas:
        with open(ROOT / f"nas_results/search_hwint8_l{lam}/genotype_final.json", "r", encoding="utf-8") as f:
            genotype = json.load(f)
        ops = [op for op, _ in genotype["normal"]] + [op for op, _ in genotype["reduce"]]
        counts = Counter(ops)
        counts_by_lambda[lam] = counts
        all_ops.update(counts)

    preferred = ["sep_conv_3x3", "rep_conv_3x3", "rep_conv_5x5", "dil_conv_3x3", "skip_connect"]
    ops_order = [op for op in preferred if op in all_ops] + sorted(all_ops - set(preferred))
    palette = {
        "sep_conv_3x3": COLORS["blue"],
        "rep_conv_3x3": COLORS["green"],
        "rep_conv_5x5": COLORS["orange"],
        "dil_conv_3x3": COLORS["purple"],
        "skip_connect": COLORS["gray"],
    }

    x = np.arange(len(lambdas))
    width = 0.14
    offsets = (np.arange(len(ops_order)) - (len(ops_order) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for offset, op in zip(offsets, ops_order):
        vals = np.array([counts_by_lambda[lam].get(op, 0) for lam in lambdas])
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            label=op,
            color=palette.get(op, "#9AA5B1"),
            alpha=0.92,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.15,
                    str(int(val)),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )

    ax.set_xticks(x, [f"lambda={lam}" for lam in lambdas])
    ax.set_ylabel("Jumlah edge terpilih")
    ax.set_title("Distribusi Operator Genotype pada Setiap Lambda", weight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    ax.set_ylim(0, 12.5)
    ax.text(
        0.0,
        -0.14,
        "Setiap kelompok lambda menunjukkan jumlah kemunculan operator pada 16 edge genotype.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["gray"],
    )
    save_figure(fig, "gambar_4_6_distribusi_operator_genotype")


def figure_4_7_tradeoff_nas():
    raspi = load_raspi_models()
    rows = [
        ("L0.05 C6", "retrain_hwNAS_L0.05_C6_stemds8_834cls", "0.05"),
        ("L0.05 C8", "retrain_hwNAS_L0.05_C8_stemds8_834cls", "0.05"),
        ("L0.05 C10", "retrain_hwNAS_L0.05_C10_stemds8_834cls", "0.05"),
        ("L0.20 C6", "retrain_hwNAS_L0.20_C6_stemds8_834cls", "0.20"),
        ("L0.20 C8", "retrain_hwNAS_L0.20_C8_stemds8_834cls", "0.20"),
        ("L0.20 C10", "retrain_hwNAS_L0.20_C10_stemds8_834cls", "0.20"),
    ]

    fig, ax = plt.subplots(figsize=(8.7, 6.2))
    styles = {"0.05": ("o", COLORS["blue"]), "0.20": ("s", COLORS["green"])}
    for label, model_dir, lam in rows:
        acc = retrain_accuracy(model_dir)
        lat = raspi[model_dir]["int8"]["latency_ms"]["mean"]
        size = raspi[model_dir]["int8"]["size_mb"]
        marker, color = styles[lam]
        edge = COLORS["red"] if label == "L0.20 C8" else "white"
        lw = 2.2 if label == "L0.20 C8" else 0.8
        ax.scatter(lat, acc, s=260 + size * 260, marker=marker, color=color, edgecolor=edge, linewidth=lw, alpha=0.9)
        ax.annotate(label, (lat, acc), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Latency Raspberry Pi INT8 mean (ms)")
    ax.set_ylabel("Test accuracy retrain (%)")
    ax.set_title("Trade-off Accuracy vs Latency Kandidat NAS", weight="bold")
    ax.set_xlim(1.85, 3.35)
    ax.set_ylim(96.0, 99.6)
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="w", markerfacecolor=COLORS["blue"], markersize=9, label="lambda=0.05"),
            plt.Line2D([], [], marker="s", color="w", markerfacecolor=COLORS["green"], markersize=9, label="lambda=0.20"),
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="white", markeredgecolor=COLORS["red"], markersize=9, label="Baseline student"),
        ],
        loc="lower right",
        frameon=True,
    )
    ax.text(1.87, 96.15, "Ukuran marker mengikuti ukuran ONNX INT8.", fontsize=8.5, color=COLORS["gray"])
    save_figure(fig, "gambar_4_7_tradeoff_accuracy_latency_nas")


def figure_4_8_tradeoff_accuracy_model_size_nas():
    rows = [
        ("L0.05 C6", "retrain_hwNAS_L0.05_C6_stemds8_834cls", "0.05"),
        ("L0.05 C8", "retrain_hwNAS_L0.05_C8_stemds8_834cls", "0.05"),
        ("L0.05 C10", "retrain_hwNAS_L0.05_C10_stemds8_834cls", "0.05"),
        ("L0.20 C6", "retrain_hwNAS_L0.20_C6_stemds8_834cls", "0.20"),
        ("L0.20 C8", "retrain_hwNAS_L0.20_C8_stemds8_834cls", "0.20"),
        ("L0.20 C10", "retrain_hwNAS_L0.20_C10_stemds8_834cls", "0.20"),
    ]

    fig, ax = plt.subplots(figsize=(8.7, 6.2))
    styles = {"0.05": ("o", COLORS["blue"]), "0.20": ("s", COLORS["green"])}
    for label, model_dir, lam in rows:
        with open(ROOT / "nas_results" / model_dir / "test_results.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        acc = d["accuracy"] * 100.0
        size = d["model_size_mb"]
        marker, color = styles[lam]
        edge = COLORS["red"] if label == "L0.20 C8" else "white"
        lw = 2.2 if label == "L0.20 C8" else 0.8
        ax.scatter(size, acc, s=190, marker=marker, color=color, edgecolor=edge, linewidth=lw, alpha=0.92)
        ax.annotate(label, (size, acc), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Estimasi ukuran bobot FP32 (MB)")
    ax.set_ylabel("Test accuracy retrain (%)")
    ax.set_title("Trade-off Accuracy vs Model Size Kandidat NAS", weight="bold")
    ax.set_xlim(1.05, 2.92)
    ax.set_ylim(96.0, 99.6)
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="w", markerfacecolor=COLORS["blue"], markersize=9, label="lambda=0.05"),
            plt.Line2D([], [], marker="s", color="w", markerfacecolor=COLORS["green"], markersize=9, label="lambda=0.20"),
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="white", markeredgecolor=COLORS["red"], markersize=9, label="Baseline student"),
        ],
        loc="lower right",
        frameon=True,
    )
    ax.text(1.07, 96.15, "Size mengikuti Tabel 4.6: estimasi bobot FP32 PyTorch, bukan file ONNX.", fontsize=8.5, color=COLORS["gray"])
    save_figure(fig, "gambar_4_8_tradeoff_accuracy_model_size_nas")


def load_kd_results():
    paths = [
        ROOT / "knowledge_distilation/kd_results/finetune_hwNAS_L0.20_C8_t3_a1.0_ls0_nomix_lr1e4_fixed/test_results.json",
        ROOT / "knowledge_distilation/kd_results/kd_hwNAS_L0.20_C8_t2_a0.8_ls0_nomix_lr1e4_fixed/test_results.json",
        ROOT / "knowledge_distilation/kd_results/kd_hwNAS_L0.20_C8_t3_a0.7_ls0_nomix_lr1e4_fixed/test_results.json",
        ROOT / "knowledge_distilation/kd_results/kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed/test_results.json",
        ROOT / "knowledge_distilation/kd_results/kd_hwNAS_L0.20_C8_t3_a0.9_ls0_nomix_lr1e4_fixed/test_results.json",
        ROOT / "knowledge_distilation/kd_results/kd_hwNAS_L0.20_C8_t4_a0.8_ls0_nomix_lr1e4_fixed/test_results.json",
    ]
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cfg = d["kd_config"]
        rows.append(
            {
                "temperature": float(cfg["temperature"]),
                "alpha": float(cfg["alpha"]),
                "test_acc": d["test_acc"] * 100.0,
                "test_loss": d["test_loss"],
                "best_val_acc": d["best_val_acc"] * 100.0,
            }
        )
    return rows


def figure_4_9_temperature_effect():
    rows = [r for r in load_kd_results() if abs(r["alpha"] - 0.8) < 1e-9]
    rows = sorted(rows, key=lambda r: r["temperature"])
    xs = [r["temperature"] for r in rows]
    ys = [r["test_acc"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.7, 5.6))
    ax.plot(xs, ys, marker="o", linewidth=2.2, color=COLORS["blue"], markersize=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}%", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(xs, [f"T={int(x)}" for x in xs])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Temperature")
    ax.set_ylim(98.75, 99.15)
    ax.set_title("Pengaruh Temperature terhadap Test Accuracy", weight="bold")
    ax.text(0.02, 0.04, "Konfigurasi alpha tetap 0.8.", transform=ax.transAxes, fontsize=8.5, color=COLORS["gray"])
    save_figure(fig, "gambar_4_9_pengaruh_temperature_test_accuracy")


def figure_4_10_alpha_effect():
    rows = [r for r in load_kd_results() if abs(r["temperature"] - 3.0) < 1e-9]
    rows = sorted(rows, key=lambda r: r["alpha"])
    xs = [r["alpha"] for r in rows]
    ys = [r["test_acc"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.7, 5.6))
    ax.plot(xs, ys, marker="o", linewidth=2.2, color=COLORS["green"], markersize=8)
    final_acc = next(r["test_acc"] for r in rows if abs(r["alpha"] - 0.8) < 1e-9)
    ax.scatter([0.8], [final_acc], s=180, color=COLORS["red"], marker="*", zorder=4, label="KD final")
    for r in rows:
        offsets = {
            0.7: (32, 12),
            0.8: (0, 15),
            0.9: (0, 12),
            1.0: (-38, 12),
        }
        ax.annotate(
            f"{r['test_acc']:.2f}%\nloss {r['test_loss']:.4f}",
            (r["alpha"], r["test_acc"]),
            xytext=offsets.get(round(r["alpha"], 1), (0, 10)),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )
    ax.set_xticks(xs, [f"{x:.1f}" for x in xs])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Alpha")
    ax.set_xlim(0.66, 1.03)
    ax.set_ylim(98.10, 99.22)
    ax.set_title("Pengaruh Alpha terhadap Test Accuracy", weight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.03,
        0.04,
        "Temperature tetap T=3; alpha=1.0 merepresentasikan fine-tuning tanpa KD.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["gray"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
    )
    save_figure(fig, "gambar_4_10_pengaruh_alpha_test_accuracy")


def figure_4_11_onnx_ptq_flow():
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    steps = [
        ("Checkpoint\nPyTorch", COLORS["light_gray"], COLORS["gray"]),
        ("Load model\n+ config", COLORS["light_blue"], COLORS["blue"]),
        ("Export\nONNX FP32\nopset 13", COLORS["light_green"], COLORS["green"]),
        ("Graph\npreprocess", "#FFF3CD", "#B08900"),
        ("Static PTQ\nQDQ INT8", COLORS["light_orange"], COLORS["orange"]),
        ("ONNX INT8\nmodel", COLORS["light_purple"], COLORS["purple"]),
        ("Benchmark\nRaspberry Pi\nORT CPU", "#E8F5F1", COLORS["green"]),
    ]
    x = 0.35
    y = 2.2
    w = 1.55
    h = 1.05
    centers = []
    for i, (txt, fc, ec) in enumerate(steps):
        box(ax, (x, y), w, h, txt, fc, ec, 9.2, "bold")
        centers.append((x + w / 2, y + h / 2))
        if i < len(steps) - 1:
            arrow(ax, (x + w, y + h / 2), (x + w + 0.38, y + h / 2), COLORS["gray"])
        x += 1.92

    box(ax, (6.65, 0.45), 2.0, 0.9, "Calibration\nimages", "#F8FAFC", COLORS["ink"], 9, "bold")
    arrow(ax, (7.65, 1.35), (8.05, 2.2), COLORS["orange"], rad=-0.12)

    ax.text(
        0.35,
        4.35,
        "Static PTQ: activation=QUInt8, weight=QInt8, per-channel weight quantization",
        fontsize=10,
        color=COLORS["gray"],
    )
    ax.set_title("Alur Export ONNX dan Post-Training Quantization INT8", weight="bold", pad=12)
    save_figure(fig, "gambar_4_11_alur_export_onnx_ptq_int8")


def ptq_rows():
    raspi = load_raspi_models()
    mapping = [
        ("EffNetV2M", "EfficientNetV2M"),
        ("MobileNetV3\nLarge", "MobileNetV3Large"),
        ("MobileNetV3\nSmall", "MobileNetV3Small"),
        ("ShuffleNetV2", "ShuffleNetV2_x1_0"),
        ("EffNetLite0", "EfficientNetLite0"),
        ("NAS\nbaseline", "retrain_hwNAS_L0.20_C8_stemds8_834cls"),
        ("NAS+KD\nfinal", "kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed"),
    ]
    return [(label, raspi[key]["fp32"], raspi[key]["int8"]) for label, key in mapping]


def grouped_bar_fp32_int8(metric, ylabel, title, filename, logy=True):
    rows = ptq_rows()
    labels = [r[0] for r in rows]
    fp32 = [r[1][metric] if metric != "latency" else r[1]["latency_ms"]["mean"] for r in rows]
    int8 = [r[2][metric] if metric != "latency" else r[2]["latency_ms"]["mean"] for r in rows]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.bar(x - width / 2, fp32, width, label="FP32", color=COLORS["blue"], alpha=0.88)
    ax.bar(x + width / 2, int8, width, label="INT8", color=COLORS["orange"], alpha=0.88)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight="bold")
    if logy:
        ax.set_yscale("log")
        ax.text(
            0.01,
            -0.16,
            "Sumbu-y log scale karena rentang nilai antar model sangat lebar.",
            transform=ax.transAxes,
            fontsize=8.5,
            color=COLORS["gray"],
        )
    ax.legend(frameon=True)
    save_figure(fig, filename)


def figure_4_12_model_size_after_int8():
    grouped_bar_fp32_int8(
        "size_mb",
        "Model size (MB, log scale)",
        "Perubahan Ukuran Model Setelah INT8",
        "gambar_4_12_perubahan_ukuran_model_setelah_int8",
        logy=True,
    )


def figure_4_13_latency_after_int8():
    grouped_bar_fp32_int8(
        "latency",
        "Raspberry Pi mean latency (ms, log scale)",
        "Perubahan Latency Setelah INT8",
        "gambar_4_13_perubahan_latency_setelah_int8",
        logy=True,
    )


def selected_deployment_rows():
    raspi = load_raspi_models()
    mapping = [
        ("Teacher\nEffNetV2M", "EfficientNetV2M"),
        ("MobileNetV3\nLarge", "MobileNetV3Large"),
        ("MobileNetV3\nSmall", "MobileNetV3Small"),
        ("ShuffleNetV2", "ShuffleNetV2_x1_0"),
        ("EffNetLite0", "EfficientNetLite0"),
        ("NAS baseline", "retrain_hwNAS_L0.20_C8_stemds8_834cls"),
        ("NAS+KD final", "kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed"),
    ]
    rows = []
    for label, key in mapping:
        item = raspi[key]["int8"]
        rows.append(
            {
                "label": label,
                "key": key,
                "acc": item["accuracy_pct"],
                "lat": item["latency_ms"]["mean"],
                "size": item["size_mb"],
            }
        )
    return rows


def scatter_deployment(x_key, x_label, title, filename, logx=True):
    rows = selected_deployment_rows()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for r in rows:
        is_final = r["label"] == "NAS+KD final"
        is_teacher = r["label"].startswith("Teacher")
        color = COLORS["red"] if is_final else (COLORS["purple"] if is_teacher else COLORS["blue"])
        marker = "*" if is_final else ("D" if is_teacher else "o")
        size = 330 if is_final else (170 if is_teacher else 120)
        ax.scatter(r[x_key], r["acc"], s=size, marker=marker, color=color, edgecolor="white", linewidth=1.0, alpha=0.92, zorder=3)
        offsets = {
            "Teacher\nEffNetV2M": (-48, -2),
            "MobileNetV3\nLarge": (8, -13),
            "MobileNetV3\nSmall": (8, 5),
            "ShuffleNetV2": (8, -12),
            "EffNetLite0": (8, 5),
            "NAS baseline": (8, -13),
            "NAS+KD final": (8, 8),
        }
        ax.annotate(r["label"], (r[x_key], r["acc"]), xytext=offsets.get(r["label"], (6, 6)), textcoords="offset points", fontsize=8.8)

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("INT8 accuracy (%)")
    ax.set_title(title, weight="bold")
    ax.set_ylim(97.5, 100.15)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(
        0.02,
        0.03,
        "Sumbu-x log scale untuk menampilkan teacher dan model ringan dalam satu grafik.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["gray"],
    )
    save_figure(fig, filename)


def figure_4_14_accuracy_latency():
    scatter_deployment(
        "lat",
        "Raspberry Pi INT8 mean latency (ms, log scale)",
        "Accuracy vs Latency pada Raspberry Pi",
        "gambar_4_14_accuracy_vs_latency_raspberry_pi",
        logx=True,
    )


def figure_4_15_accuracy_size():
    scatter_deployment(
        "size",
        "INT8 model size (MB, log scale)",
        "INT8 Accuracy vs INT8 Model Size",
        "gambar_4_15_int8_accuracy_vs_model_size",
        logx=True,
    )


def final_complexity_rows():
    return [
        {"label": "EffNetV2M", "params": 53926710, "flops": 5446.3, "group": "Teacher"},
        {"label": "MobileNetV3\nLarge", "params": 5270386, "flops": 234.6, "group": "Lightweight"},
        {"label": "MobileNetV3\nSmall", "params": 2372706, "flops": 62.3, "group": "Lightweight"},
        {"label": "ShuffleNetV2", "params": 2108454, "flops": 152.5, "group": "Lightweight"},
        {"label": "EffNetLite0", "params": 4439362, "flops": 367.4, "group": "Lightweight"},
        {"label": "NAS\nbaseline", "params": 522332, "flops": 41.1, "group": "NAS"},
        {"label": "NAS+KD\nfinal", "params": 522332, "flops": 41.1, "group": "NAS"},
    ]


def complexity_bar(metric, ylabel, title, filename):
    rows = final_complexity_rows()
    labels = [r["label"] for r in rows]
    vals = [r[metric] for r in rows]
    color_map = {"Teacher": COLORS["purple"], "Lightweight": COLORS["blue"], "NAS": COLORS["green"]}
    colors = [color_map[r["group"]] for r in rows]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    bars = ax.bar(labels, vals, color=colors, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight="bold")
    for bar, val in zip(bars, vals):
        label = f"{val / 1e6:.2f}M" if metric == "params" else f"{val:.1f}M"
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, label, ha="center", va="bottom", fontsize=8.4)
    handles = [
        plt.Line2D([], [], marker="s", color="w", markerfacecolor=COLORS["purple"], markersize=9, label="Teacher"),
        plt.Line2D([], [], marker="s", color="w", markerfacecolor=COLORS["blue"], markersize=9, label="Lightweight CNN"),
        plt.Line2D([], [], marker="s", color="w", markerfacecolor=COLORS["green"], markersize=9, label="NAS"),
    ]
    ax.legend(handles=handles, frameon=True)
    ax.text(
        0.01,
        -0.16,
        "Sumbu-y log scale untuk menjaga keterbacaan model besar dan model ringan.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["gray"],
    )
    save_figure(fig, filename)


def figure_4_16_parameter_comparison():
    complexity_bar(
        "params",
        "Jumlah parameter (log scale)",
        "Perbandingan Parameter Seluruh Model",
        "gambar_4_16_perbandingan_parameter_seluruh_model",
    )


def figure_4_17_flops_comparison():
    complexity_bar(
        "flops",
        "FLOPs/MMACs (M, log scale)",
        "Perbandingan FLOPs Seluruh Model",
        "gambar_4_17_perbandingan_flops_seluruh_model",
    )


def figure_4_18_final_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    steps = [
        ("Dataset +\npreprocessing", "SCUT_PV_v1\n834 kelas\n224 x 224", COLORS["light_gray"], COLORS["gray"]),
        ("Hardware-\naware NAS", "P-DARTS\nLUT INT8 Pi\nlambda=0.20", COLORS["light_blue"], COLORS["blue"]),
        ("Retrain\nstudent", "L0.20 C8\n98.44%\n2.22 ms INT8", COLORS["light_green"], COLORS["green"]),
        ("Knowledge\nDistillation", "Teacher: EffNetV2M\nT=3, alpha=0.8\n99.04% FP32", COLORS["light_purple"], COLORS["purple"]),
        ("PTQ INT8", "QDQ static PTQ\n0.596 MB\n98.80%", COLORS["light_orange"], COLORS["orange"]),
        ("Raspberry Pi\ndeployment", "2.27 ms mean\n2.20 ms median\n2.51 ms p95", "#E8F5F1", COLORS["green"]),
    ]

    x = 0.35
    y = 2.6
    w = 1.85
    h = 1.15
    for i, (title, detail, fc, ec) in enumerate(steps):
        box(ax, (x, y), w, h, title, fc, ec, 9.3, "bold")
        ax.text(x + w / 2, y - 0.22, detail, ha="center", va="top", fontsize=8.4, color=COLORS["gray"])
        if i < len(steps) - 1:
            arrow(ax, (x + w, y + h / 2), (x + w + 0.35, y + h / 2), COLORS["gray"])
        x += 2.22

    box(
        ax,
        (1.55, 0.45),
        10.9,
        0.9,
        "Kontribusi akhir: hardware-aware NAS + Knowledge Distillation + PTQ INT8\nuntuk model palm vein ringan, cepat, dan akurat pada Raspberry Pi",
        "#FFFDF5",
        "#B08900",
        9,
        "bold",
    )
    ax.set_title("Ringkasan Pipeline dan Hasil Akhir Model Final", weight="bold", pad=14)
    save_figure(fig, "gambar_4_18_ringkasan_pipeline_model_final")


def main():
    figure_4_2_preprocessing_example()
    figure_4_3_lut_objective()
    figure_4_4_lut_cost_bar()
    figure_4_5_genotype()
    figure_4_6_genotype_operator_distribution()
    figure_4_7_tradeoff_nas()
    figure_4_8_tradeoff_accuracy_model_size_nas()
    figure_4_9_temperature_effect()
    figure_4_10_alpha_effect()
    figure_4_11_onnx_ptq_flow()
    figure_4_12_model_size_after_int8()
    figure_4_13_latency_after_int8()
    figure_4_14_accuracy_latency()
    figure_4_15_accuracy_size()
    figure_4_16_parameter_comparison()
    figure_4_17_flops_comparison()
    figure_4_18_final_pipeline()
    print(f"Generated figures in: {OUT_DIR}")
    for path in sorted(OUT_DIR.glob("gambar_4_*.*")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
