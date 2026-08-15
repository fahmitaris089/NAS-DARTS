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
    parser.add_argument("--num_cells", type=int, default=None,
                        help="Number of cells (default: read from checkpoint config)")
    parser.add_argument("--stem_downsample", type=int, default=None,
                        help="Stem downsampling factor (default: read from config)")
    parser.add_argument("--reduction_indices", type=str, default=None,
                        help="Comma-separated reduction cell indices (default: read from config)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = get_device()

    # Load the effective training configuration stored beside the checkpoint.
    # Do not use the nested ``retrain_cfg`` snapshot here: it records project
    # defaults, which may have been overridden by the original CLI invocation.
    retrain_dir = Path(args.model_path).parent
    config_path = retrain_dir / "config.json"
    run_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            run_cfg = json.load(f)

    C_init = args.C_init if args.C_init is not None else int(
        run_cfg.get("C_init", RETRAIN_CFG["C_init"])
    )
    num_cells = args.num_cells if args.num_cells is not None else int(
        run_cfg.get("num_cells", RETRAIN_CFG["num_cells"])
    )
    stem_downsample = args.stem_downsample if args.stem_downsample is not None else int(
        run_cfg.get("stem_downsample", 2)
    )
    reduction_value = (
        args.reduction_indices
        if args.reduction_indices is not None
        else run_cfg.get("reduction_indices")
    )
    if isinstance(reduction_value, str):
        reduction_indices = [int(x) for x in reduction_value.split(",") if x.strip()]
    elif reduction_value is None:
        reduction_indices = None
    else:
        reduction_indices = [int(x) for x in reduction_value]

    data_dir = args.data_dir or run_cfg.get("data_dir")
    split_path = args.split_path or run_cfg.get("split_path")
    print(
        "Effective architecture: "
        f"C_init={C_init}, cells={num_cells}, stem_downsample={stem_downsample}, "
        f"reduction_indices={reduction_indices}"
    )

    output_dir = Path(args.output_dir) if args.output_dir else retrain_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load genotype
    with open(args.genotype) as f:
        genotype = dict_to_genotype(json.load(f))

    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "student" in state_dict:
        state_dict = state_dict["student"]
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("_auxiliary_head")}

    # The selected PK-CE checkpoint was trained without an auxiliary head.
    # Reconstruct the exact inference architecture, including its stem and
    # explicit reduction locations, and require an exact state-dict match.
    model = EvalNetwork(
        genotype=genotype,
        C_init=C_init,
        num_cells=num_cells,
        num_classes=NUM_CLASSES,
        auxiliary=False,
        dropout=RETRAIN_CFG["dropout"],
        stem_downsample=stem_downsample,
        reduction_indices=reduction_indices,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    total_params = count_parameters(model)
    config_params = run_cfg.get("total_params")
    if config_params is not None and int(config_params) != total_params:
        raise ValueError(
            f"Parameter mismatch: reconstructed={total_params}, config={config_params}"
        )

    print(f"\nModel: NAS-PDARTS (C_init={C_init}, cells={num_cells})")
    print(f"Parameters: {total_params:,}")
    print(param_breakdown(model))

    # Data
    _, _, test_loader, data_info = create_retrain_dataloaders(
        data_dir=data_dir,
        split_path=split_path,
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
