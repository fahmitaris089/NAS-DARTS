"""
Palm Vein Recognition — Compare All 9 Models
=============================================
Run after all models are trained:
    python3 evaluate_all.py

Reads test_results.json from each model folder and generates:
- comparison_table.csv
- accuracy_comparison.png
- roc_comparison.png
- training_curves_comparison.png
- comparison_summary.txt
"""

import json
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("training_results")

MODEL_NAMES = [
    "InceptionV3", "ResNet50", "VGG16", "DenseNet121",
    "EfficientNetB4", "EfficientNetV2M", "MobileNetV3Large",
    "MobileNetV3Small", "ShuffleNetV2_x1_0", "EfficientNetLite0",
    "ConvNeXtBase", "RegNetY16GF",
]


def load_results():
    """Load test_results.json from each model folder."""
    results = {}
    for name in MODEL_NAMES:
        path = RESULTS_DIR / name / "test_results.json"
        if path.exists():
            with open(path, "r") as f:
                results[name] = json.load(f)
        else:
            print(f"  [SKIP] {name} — not found at {path}")
    return results


def load_last_results():
    """Load last_model_results.json from each model folder."""
    results = {}
    for name in MODEL_NAMES:
        path = RESULTS_DIR / name / "last_model_results.json"
        if path.exists():
            with open(path, "r") as f:
                results[name] = json.load(f)
    return results


def generate_comparison_table(results):
    """Generate CSV comparison table and print to console."""
    csv_path = RESULTS_DIR / "comparison_table.csv"

    headers = [
        "Model", "Accuracy(%)", "Precision", "Recall", "F1",
        "AUC", "EER(%)", "Params(M)", "BestEpoch",
        "InferenceTime(ms)", "TrainingTime(min)",
    ]

    rows = []
    for name in MODEL_NAMES:
        if name not in results:
            continue
        r = results[name]
        rows.append([
            name,
            f"{r['accuracy']*100:.2f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1_score']:.4f}",
            f"{r.get('auc', 'N/A') or 'N/A'}",
            f"{r.get('eer', 0)*100:.2f}" if r.get("eer") else "N/A",
            f"{r.get('total_params', 0)/1e6:.1f}",
            str(r.get("best_epoch", "N/A")),
            f"{r.get('inference_time_per_batch_sec', 0)*1000:.1f}",
            f"{r.get('training_time_min', 0):.1f}",
        ])

    # Sort by accuracy descending
    rows.sort(key=lambda x: float(x[1]), reverse=True)

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # Print table
    print(f"\n{'='*120}")
    print("  MODEL COMPARISON RESULTS")
    print(f"{'='*120}")
    header_fmt = f"{'Model':<20} {'Acc(%)':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} {'EER(%)':>8} {'Params(M)':>10} {'BestEp':>7} {'Infer(ms)':>10} {'Train(min)':>11}"
    print(header_fmt)
    print("-" * 120)
    for row in rows:
        print(f"{row[0]:<20} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8} "
              f"{row[5]:>8} {row[6]:>8} {row[7]:>10} {row[8]:>7} {row[9]:>10} {row[10]:>11}")
    print(f"{'='*120}")
    print(f"Saved → {csv_path}")

    return rows


def plot_accuracy_comparison(results):
    """Bar chart comparing accuracy and F1 across models."""
    names = []
    accs  = []
    f1s   = []
    params = []

    for name in MODEL_NAMES:
        if name not in results:
            continue
        names.append(name)
        accs.append(results[name]["accuracy"] * 100)
        f1s.append(results[name]["f1_score"] * 100)
        params.append(results[name].get("total_params", 0) / 1e6)

    # Sort by accuracy
    idx = np.argsort(accs)[::-1]
    names  = [names[i] for i in idx]
    accs   = [accs[i] for i in idx]
    f1s    = [f1s[i] for i in idx]
    params = [params[i] for i in idx]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Accuracy & F1 bars
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy (%)", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1 Score (%)", color="tomato", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Model Comparison — Accuracy & F1 Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    # Accuracy vs Params scatter
    ax2 = axes[1]
    ax2.scatter(params, accs, s=100, c="steelblue", edgecolors="navy", zorder=3)
    for i, name in enumerate(names):
        ax2.annotate(name, (params[i], accs[i]), fontsize=7,
                     xytext=(5, 5), textcoords="offset points")
    ax2.set_xlabel("Parameters (M)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy vs Model Size")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = RESULTS_DIR / "accuracy_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_training_curves_comparison(results):
    """Overlay val_loss and val_acc curves for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name in MODEL_NAMES:
        log_path = RESULTS_DIR / name / "training_log.csv"
        if not log_path.exists():
            continue

        epochs, val_losses, val_accs = [], [], []
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                val_losses.append(float(row["val_loss"]))
                val_accs.append(float(row["val_acc"]))

        ax1.plot(epochs, val_losses, label=name, linewidth=1.2, alpha=0.8)
        ax2.plot(epochs, val_accs,   label=name, linewidth=1.2, alpha=0.8)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Loss")
    ax1.set_title("Validation Loss — All Models")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy")
    ax2.set_title("Validation Accuracy — All Models")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = RESULTS_DIR / "training_curves_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_best_vs_last(best_results, last_results):
    """Compare best model vs last model accuracy for each architecture."""
    names = []
    best_accs = []
    last_accs = []

    for name in MODEL_NAMES:
        if name in best_results and name in last_results:
            names.append(name)
            best_accs.append(best_results[name]["accuracy"] * 100)
            last_accs.append(last_results[name]["accuracy"] * 100)

    if not names:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    w = 0.35

    ax.bar(x - w/2, best_accs, w, label="Best Model (min val_loss)", color="steelblue")
    ax.bar(x + w/2, last_accs, w, label="Last Model (epoch 300)", color="lightcoral")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Best Model vs Last Model — Overfitting Analysis")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add difference annotation
    for i in range(len(names)):
        diff = best_accs[i] - last_accs[i]
        color = "green" if diff > 0 else "red"
        ax.annotate(f"Δ={diff:+.1f}%", (x[i], max(best_accs[i], last_accs[i]) + 0.5),
                    ha="center", fontsize=7, color=color, fontweight="bold")

    plt.tight_layout()
    save_path = RESULTS_DIR / "best_vs_last.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def generate_summary(results):
    """Generate text summary."""
    summary_path = RESULTS_DIR / "comparison_summary.txt"

    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )

    lines = []
    lines.append("=" * 60)
    lines.append("  PALM VEIN RECOGNITION — MODEL COMPARISON SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Ranking by Test Accuracy:")
    for rank, (name, r) in enumerate(sorted_models, 1):
        lines.append(
            f"  #{rank}  {name:<20}  "
            f"Acc={r['accuracy']*100:.2f}%  "
            f"F1={r['f1_score']:.4f}  "
            f"Params={r.get('total_params', 0)/1e6:.1f}M  "
            f"BestEp={r.get('best_epoch', 'N/A')}"
        )

    lines.append("")
    if sorted_models:
        best_name, best_r = sorted_models[0]
        lines.append(f"BEST MODEL: {best_name}")
        lines.append(f"  Accuracy  : {best_r['accuracy']*100:.2f}%")
        lines.append(f"  F1 Score  : {best_r['f1_score']:.4f}")
        lines.append(f"  AUC       : {best_r.get('auc', 'N/A')}")
        lines.append(f"  EER       : {best_r.get('eer', 'N/A')}")
        lines.append(f"  Parameters: {best_r.get('total_params', 0):,}")

        # Smallest model with >90% accuracy (if any)
        efficient = [(n, r) for n, r in sorted_models
                     if r["accuracy"] > 0.9 and r.get("total_params", 0) > 0]
        if efficient:
            efficient.sort(key=lambda x: x[1].get("total_params", float("inf")))
            eff_name, eff_r = efficient[0]
            lines.append("")
            lines.append(f"MOST EFFICIENT (>90% acc): {eff_name}")
            lines.append(f"  Accuracy  : {eff_r['accuracy']*100:.2f}%")
            lines.append(f"  Parameters: {eff_r.get('total_params', 0):,}")

    lines.append("")
    lines.append("=" * 60)

    text = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(text)

    print(text)
    print(f"\nSaved → {summary_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading results from trained models...")
    results      = load_results()
    last_results = load_last_results()

    if not results:
        print("No trained models found! Run train_model.py first.")
        return

    print(f"Found {len(results)}/{len(MODEL_NAMES)} models\n")

    # 1. Comparison table
    generate_comparison_table(results)

    # 2. Accuracy bar chart + scatter
    plot_accuracy_comparison(results)

    # 3. Training curves overlay
    plot_training_curves_comparison(results)

    # 4. Best vs Last analysis
    plot_best_vs_last(results, last_results)

    # 5. Text summary
    generate_summary(results)

    print(f"\nAll outputs saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
