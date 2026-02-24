"""
Evaluate NAS Model — Comprehensive Metrics & Teacher Comparison
================================================================
Standalone evaluation script for the retrained NAS model.
Computes all metrics identical to teacher evaluation for fair comparison.

Usage:
    python evaluate.py --model_path nas_results/retrain/best_model.pth \
                       --genotype nas_results/search/genotype_final.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nas_config import NUM_CLASSES, RETRAIN_CFG, RETRAIN_DIR, SEARCH_DIR
from genotypes import dict_to_genotype
from model_eval import EvalNetwork, count_parameters, param_breakdown
from palm_vein_dataset import create_retrain_dataloaders
from utils import get_device, model_size_mb, estimate_flops, measure_latency


def main():
    parser = argparse.ArgumentParser(description="Evaluate NAS model")
    parser.add_argument("--model_path", type=str,
                        default=str(RETRAIN_DIR / "best_model.pth"),
                        help="Path to model weights")
    parser.add_argument("--genotype", type=str,
                        default=str(SEARCH_DIR / "genotype_final.json"),
                        help="Path to genotype JSON")
    parser.add_argument("--C_init", type=int, default=None,
                        help="C_init (auto-detect from config if not set)")
    parser.add_argument("--num_cells", type=int, default=RETRAIN_CFG["num_cells"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = get_device()

    # Try to load config for C_init
    C_init = args.C_init
    retrain_dir = Path(args.model_path).parent
    config_path = retrain_dir / "config.json"
    if C_init is None and config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        C_init = cfg.get("C_init", RETRAIN_CFG["C_init"])
        print(f"Loaded C_init={C_init} from config")
    elif C_init is None:
        C_init = RETRAIN_CFG["C_init"]

    output_dir = Path(args.output_dir) if args.output_dir else retrain_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load genotype
    with open(args.genotype) as f:
        genotype = dict_to_genotype(json.load(f))

    # Build model
    model = EvalNetwork(
        genotype=genotype,
        C_init=C_init,
        num_cells=args.num_cells,
        num_classes=NUM_CLASSES,
        auxiliary=False,  # no aux for eval
        dropout=RETRAIN_CFG["dropout"],
    ).to(device)

    # Load weights
    state_dict = torch.load(args.model_path, map_location=device)
    # Handle auxiliary head mismatch (trained with aux, eval without)
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("_auxiliary_head")}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    total_params = count_parameters(model)
    print(f"\nModel: NAS-PDARTS (C_init={C_init}, cells={args.num_cells})")
    print(f"Parameters: {total_params:,}")
    print(param_breakdown(model))

    # Data
    _, _, test_loader, data_info = create_retrain_dataloaders(
        data_dir=args.data_dir,
        split_path=args.split_path,
        batch_size=args.batch_size,
        num_workers=2,
        use_augmentation=False,
    )
    num_classes = data_info["num_classes"]

    # Full evaluation
    from retrain import evaluate_test
    results, cm, cls_report, all_labels, all_preds, all_probs = \
        evaluate_test(model, test_loader, device, num_classes)

    # Efficiency metrics
    results["total_params"] = total_params
    results["model_size_mb"] = model_size_mb(model)

    flops, _ = estimate_flops(model, device="cpu")
    if flops:
        results["flops"] = flops
        results["flops_M"] = flops / 1e6

    try:
        lat_gpu, lat_std = measure_latency(model, device=str(device))
        results["latency_gpu_ms"] = lat_gpu
    except Exception:
        pass

    try:
        lat_cpu, _ = measure_latency(model.cpu(), device="cpu")
        results["latency_cpu_ms"] = lat_cpu
        model.to(device)
    except Exception:
        pass

    # Print
    print(f"\n{'='*60}")
    print(f"  NAS-PDARTS Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy  : {results['accuracy']*100:.2f}%")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1        : {results['f1_score']:.4f}")
    print(f"  AUC       : {results.get('auc', 'N/A')}")
    print(f"  EER       : {results.get('eer', 'N/A')}")
    print(f"  Params    : {total_params:,}")
    print(f"  Size      : {results['model_size_mb']:.2f} MB")
    if flops:
        print(f"  FLOPs     : {flops/1e6:.1f} M")
    if "latency_gpu_ms" in results:
        print(f"  Latency   : {results['latency_gpu_ms']:.1f} ms (GPU)")
    if "latency_cpu_ms" in results:
        print(f"  Latency   : {results['latency_cpu_ms']:.1f} ms (CPU)")

    # Teacher comparison
    teacher_csv = Path(__file__).resolve().parent.parent / "Teacher" / "training_results" / "comparison_table.csv"
    if teacher_csv.exists():
        import csv
        print(f"\n{'='*60}")
        print(f"  Comparison with Teacher Models")
        print(f"{'='*60}")
        print(f"  {'Model':<25} {'Acc':>8} {'Params':>12} {'Compression':>12}")
        print(f"  {'-'*57}")

        with open(teacher_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_name = row.get("model", row.get("Model", "?"))
                t_acc = row.get("test_accuracy", row.get("Test Accuracy", "?"))
                t_params = row.get("total_params", row.get("Total Params", "?"))
                try:
                    t_p = int(str(t_params).replace(",", ""))
                    ratio = f"{t_p / total_params:.0f}x"
                except (ValueError, TypeError):
                    ratio = "?"
                print(f"  {t_name:<25} {t_acc:>8} {t_params:>12} {ratio:>12}")

        print(f"  {'-'*57}")
        print(f"  {'NAS-PDARTS':<25} {results['accuracy']*100:.2f}% {total_params:>12,} {'1x':>12}")

    # Save
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(output_dir / "eval_classification_report.txt", "w") as f:
        f.write(f"NAS-PDARTS Evaluation\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Accuracy: {results['accuracy']*100:.2f}%\n\n")
        f.write(cls_report)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
