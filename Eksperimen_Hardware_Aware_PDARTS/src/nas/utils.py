"""
Utility Functions for NAS
==========================
Logging, visualisation, reproducibility, metrics helpers.
"""

import os
import sys
import json
import random
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Device ──────────────────────────────────────────────────────────────────

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)} (CUDA)")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("Device: Apple MPS")
    else:
        dev = torch.device("cpu")
        print("Device: CPU")
    return dev


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logger(name, log_file, level=logging.INFO):
    """Create a logger that writes to both file and console."""
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ─── Average Meter ───────────────────────────────────────────────────────────

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name=""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ─── Timer ───────────────────────────────────────────────────────────────────

class Timer:
    """Simple timer context manager."""

    def __init__(self):
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


# ─── Model Size & FLOPs ─────────────────────────────────────────────────────

def model_size_mb(model):
    """Compute model size in MB (FP32 weights)."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def estimate_flops(model, input_size=(1, 3, 224, 224), device="cpu"):
    """
    Estimate FLOPs using thop if available, else manual estimate.
    Returns (flops, params) or (None, params).
    """
    try:
        from thop import profile
        dummy = torch.randn(*input_size).to(device)
        model_copy = model.to(device)
        model_copy.eval()
        flops, params = profile(model_copy, inputs=(dummy,), verbose=False)
        return flops, params
    except ImportError:
        params = sum(p.numel() for p in model.parameters())
        return None, params
    except Exception:
        params = sum(p.numel() for p in model.parameters())
        return None, params


def measure_latency(model, input_size=(1, 3, 224, 224), device="cuda",
                    warmup=10, repeats=50):
    """
    Measure inference latency (ms) on given device.
    Returns: (mean_ms, std_ms)
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device == "cuda" or str(device).startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(dummy)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)

    return np.mean(times), np.std(times)


# ─── CutOut Augmentation ────────────────────────────────────────────────────

class Cutout:
    """
    Randomly masks out a square patch in the image tensor.
    Applied after ToTensor and normalisation.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones(h, w, dtype=img.dtype)

        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)

        mask[y1:y2, x1:x2] = 0.0
        mask = mask.unsqueeze(0)  # (1, H, W)
        return img * mask


# ─── Visualisation ───────────────────────────────────────────────────────────

def plot_alpha_evolution(alpha_log, primitives, save_path, cell_type="normal"):
    """
    Plot how alpha weights evolve during search.

    Args:
        alpha_log: list of alpha tensors (one per epoch)
        primitives: list of op names
        save_path: output PNG path
        cell_type: "normal" or "reduce"
    """
    if len(alpha_log) == 0:
        return

    num_epochs = len(alpha_log)
    num_edges = alpha_log[0].shape[0]
    num_ops = alpha_log[0].shape[1]

    # Average across edges for each op
    avg_weights = np.zeros((num_epochs, num_ops))
    for t, alpha in enumerate(alpha_log):
        w = torch.softmax(torch.tensor(alpha), dim=-1).numpy()
        avg_weights[t] = w.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, prim in enumerate(primitives):
        ax.plot(range(num_epochs), avg_weights[:, i], label=prim, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Softmax Weight")
    ax.set_title(f"Alpha Evolution — {cell_type.capitalize()} Cell")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(log_path, save_dir):
    """Plot training & validation loss/accuracy curves from CSV log."""
    import csv

    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch_val = row.get("epoch", "").strip()
                if not epoch_val:
                    continue
                epochs.append(int(epoch_val))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                train_acc.append(float(row["train_acc"]))
                val_acc.append(float(row["val_acc"]))
            except (ValueError, KeyError):
                continue

    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, val_loss, label="Val Loss", linewidth=1.5)
    best_epoch = epochs[np.argmin(val_loss)]
    ax1.axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
                label=f"Best (epoch {best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train Acc", linewidth=1.5)
    ax2.plot(epochs, val_acc, label="Val Acc", linewidth=1.5)
    ax2.axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
                label=f"Best (epoch {best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def visualize_genotype(genotype, save_path, num_nodes=4):
    """
    Create a text-based / matplotlib visualisation of the cell DAG.
    Falls back to text if graphviz not available.
    """
    lines = []
    for cell_type, ops, concat in [
        ("Normal", genotype.normal, genotype.normal_concat),
        ("Reduce", genotype.reduce, genotype.reduce_concat),
    ]:
        lines.append(f"\n{'='*50}")
        lines.append(f"  {cell_type} Cell")
        lines.append(f"{'='*50}")
        lines.append(f"  Inputs: node_0 (c_{{k-2}}), node_1 (c_{{k-1}})")

        for i in range(num_nodes):
            node_id = i + 2
            edges = []
            for j in range(2):  # TOP_K_EDGES = 2
                idx = i * 2 + j
                op_name, src = ops[idx]
                edges.append(f"{op_name}(node_{src})")
            lines.append(f"  node_{node_id} = {' + '.join(edges)}")

        concat_str = ", ".join(f"node_{c}" for c in concat)
        lines.append(f"  Output = concat({concat_str})")

    text = "\n".join(lines)

    # Save text version
    txt_path = Path(save_path).with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write(text)

    # Try graphviz visualisation
    try:
        _visualize_graphviz(genotype, save_path, num_nodes)
    except Exception:
        # Fallback: matplotlib text rendering
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax_idx, (cell_type, ops, concat) in enumerate([
            ("Normal", genotype.normal, genotype.normal_concat),
            ("Reduce", genotype.reduce, genotype.reduce_concat),
        ]):
            ax = axes[ax_idx]
            ax.set_xlim(0, 10)
            ax.set_ylim(0, num_nodes + 3)
            ax.set_title(f"{cell_type} Cell", fontsize=14, fontweight='bold')
            ax.axis('off')

            # Draw nodes
            node_positions = {}
            # Input nodes
            for n_idx, label in enumerate(["c_{k-2}", "c_{k-1}"]):
                y = num_nodes + 2 - n_idx * 0.8
                node_positions[n_idx] = (1.5, y)
                ax.text(1.5, y, label, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue'),
                        fontsize=9)

            # Intermediate nodes
            for i in range(num_nodes):
                node_id = i + 2
                y = num_nodes + 2 - (i + 2) * 0.8
                node_positions[node_id] = (8, y)
                ax.text(8, y, f"node_{node_id}", ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'),
                        fontsize=9)

                # Draw edges
                for j in range(2):
                    idx = i * 2 + j
                    op_name, src = ops[idx]
                    src_pos = node_positions[src]
                    dst_pos = node_positions[node_id]
                    mid_x = (src_pos[0] + dst_pos[0]) / 2
                    mid_y = (src_pos[1] + dst_pos[1]) / 2 + 0.15 * (j * 2 - 1)
                    ax.annotate('', xy=dst_pos, xytext=src_pos,
                                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
                    ax.text(mid_x, mid_y, op_name, ha='center', va='center',
                            fontsize=7, color='darkred')

            # Output node
            y = num_nodes + 2 - (num_nodes + 2) * 0.8
            ax.text(5, y - 0.3, f"output = concat({','.join(str(c) for c in concat)})",
                    ha='center', fontsize=8, style='italic')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return text


def _visualize_graphviz(genotype, save_path, num_nodes):
    """Visualise cell using graphviz (optional dependency)."""
    import graphviz

    for cell_type, ops, concat in [
        ("normal", genotype.normal, genotype.normal_concat),
        ("reduce", genotype.reduce, genotype.reduce_concat),
    ]:
        g = graphviz.Digraph(
            name=cell_type,
            format='png',
            graph_attr={'rankdir': 'LR', 'dpi': '150'},
            node_attr={'shape': 'rectangle', 'style': 'filled', 'fontsize': '10'},
        )

        g.node("c_{k-2}", fillcolor='lightblue')
        g.node("c_{k-1}", fillcolor='lightblue')

        for i in range(num_nodes):
            g.node(f"node_{i+2}", fillcolor='lightyellow')
            for j in range(2):
                idx = i * 2 + j
                op_name, src = ops[idx]
                src_label = f"c_{{k-{2-src}}}" if src < 2 else f"node_{src}"
                g.edge(src_label, f"node_{i+2}", label=op_name)

        g.node("output", fillcolor='lightgreen')
        for c in concat:
            g.edge(f"node_{c}", "output")

        out_path = str(Path(save_path).with_suffix('')) + f"_{cell_type}"
        g.render(out_path, cleanup=True)
