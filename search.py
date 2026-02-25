"""
P-DARTS Search — Progressive Differentiable Architecture Search
================================================================
3-stage progressive search:
  Stage 1: 5 cells, 8 ops, 50 epochs  → explore all operations
  Stage 2: 8 cells, 5 ops, 50 epochs  → prune 3 weakest ops
  Stage 3: 11 cells, 3 ops, 50 epochs → prune 2 more, final genotype

Key P-DARTS features:
  - Progressive depth increase (bridges search/eval depth gap)
  - Operation pruning between stages
  - Alpha warmup (15 epochs weight-only training per stage)
  - Skip-connect dropout regularisation
  - First-order bilevel optimisation

Usage:
    python search.py
    python search.py --batch_size 32 --epochs_per_stage 15  # quick test
"""

import argparse
import csv
import json
import sys
import time
import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from nas_config import (
    PRIMITIVES, PDARTS_STAGES, SEARCH_CFG, SEARCH_DIR, NUM_CLASSES,
    INPUT_SIZE, SEARCH_INPUT_SIZE, SEED, NUM_NODES, NUM_INPUT_NODES,
)
from model_search import SearchNetwork
from architect import Architect
from genotypes import Genotype, genotype_to_dict
from palm_vein_dataset import create_search_dataloaders
from utils import (
    set_seed, get_device, setup_logger, AverageMeter, Timer,
    plot_alpha_evolution, visualize_genotype,
)


# ─── Search One Epoch ────────────────────────────────────────────────────────

def search_epoch(model, train_loader, val_loader, criterion,
                 w_optimizer, architect, device,
                 skip_dropout_mask, grad_clip, epoch, logger,
                 update_alpha=True):
    """
    One epoch of bilevel search:
      For each train batch:
        1. Update architecture α on val batch (if update_alpha=True)
        2. Update weights w on train batch

    Args:
        update_alpha: if False, skip alpha updates (warmup phase)
    """
    model.train()
    loss_w = AverageMeter("w_loss")
    loss_a = AverageMeter("a_loss")
    top1 = AverageMeter("acc")
    num_batches = len(train_loader)

    val_iter = iter(val_loader)

    for step, (input_train, target_train) in enumerate(train_loader):
        input_train = input_train.to(device, non_blocking=True)
        target_train = target_train.to(device, non_blocking=True)

        # ── Step 1: Update architecture parameters (α) ──
        if update_alpha:
            try:
                input_val, target_val = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                input_val, target_val = next(val_iter)

            input_val = input_val.to(device, non_blocking=True)
            target_val = target_val.to(device, non_blocking=True)

            a_loss = architect.step(input_val, target_val, criterion, skip_dropout_mask)
            loss_a.update(a_loss, input_val.size(0))

        # ── Step 2: Update network weights (w) ──
        w_optimizer.zero_grad()
        logits = model(input_train, skip_dropout_mask=skip_dropout_mask)
        w_loss = criterion(logits, target_train)
        w_loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.weight_parameters(), grad_clip)

        w_optimizer.step()

        # Accuracy
        _, pred = logits.max(1)
        acc = pred.eq(target_train).float().mean().item()

        loss_w.update(w_loss.item(), input_train.size(0))
        top1.update(acc, input_train.size(0))

        # Batch-level progress logging
        if step % 10 == 0 or step == num_batches - 1:
            logger.info(
                f"    batch {step+1:>4}/{num_batches} │ "
                f"w={loss_w.avg:.4f} a={loss_a.avg:.4f} acc={top1.avg:.4f}"
            )

    return loss_w.avg, loss_a.avg, top1.avg


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate on validation set (no alpha update)."""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        _, pred = logits.max(1)
        acc = pred.eq(labels).float().mean().item()

        losses.update(loss.item(), images.size(0))
        top1.update(acc, images.size(0))

    return losses.avg, top1.avg


# ─── Operation Pruning (P-DARTS) ─────────────────────────────────────────────

def prune_operations(model, primitives, num_ops_to_keep):
    """
    Prune weak operations from search space.

    Strategy: compute average softmax weight across all edges for each op,
    keep top-K operations.

    Args:
        model:            SearchNetwork
        primitives:       current list of op names
        num_ops_to_keep:  how many ops to retain

    Returns: new_primitives (list of kept op names)
    """
    if num_ops_to_keep >= len(primitives):
        return primitives

    # Compute average weight per op across normal + reduce alphas
    all_alphas = torch.cat([
        F.softmax(model.alpha_normal, dim=-1).detach().cpu(),
        F.softmax(model.alpha_reduce, dim=-1).detach().cpu(),
    ], dim=0)  # (2*num_edges, num_ops)

    avg_weights = all_alphas.mean(dim=0)  # (num_ops,)

    # Rank operations
    op_scores = [(primitives[i], avg_weights[i].item()) for i in range(len(primitives))]
    op_scores_sorted = sorted(op_scores, key=lambda x: -x[1])

    # Structural ops that are always kept
    STRUCTURAL_OPS = {'none', 'skip_connect'}
    # Convolution ops — guarantee at least one survives pruning
    CONV_OPS = {'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'}

    kept = []
    remaining = []
    for name, score in op_scores_sorted:
        if name in STRUCTURAL_OPS and len(kept) < num_ops_to_keep:
            kept.append(name)
        else:
            remaining.append((name, score))

    # Guarantee at least one conv op survives (anti-collapse safeguard)
    has_conv = any(name in CONV_OPS for name in kept)
    if not has_conv and len(kept) < num_ops_to_keep:
        # Find the best-scoring conv op and add it
        for name, score in remaining:
            if name in CONV_OPS:
                kept.append(name)
                remaining = [(n, s) for n, s in remaining if n != name]
                logging.info(f"  Anti-collapse: forced conv op '{name}' (score={score:.4f}) into kept set")
                break

    # Fill remaining slots with highest-scoring ops
    for name, score in remaining:
        if len(kept) >= num_ops_to_keep:
            break
        kept.append(name)

    # Sort to maintain consistent ordering
    kept_ordered = [p for p in PRIMITIVES if p in kept]

    return kept_ordered


# ─── Transfer Alphas Between Stages ──────────────────────────────────────────

def transfer_alphas(old_model, new_model, old_primitives, new_primitives):
    """
    Transfer alpha values from old search model to new one
    when operations are pruned between stages.
    """
    old_prims = old_primitives
    new_prims = new_primitives

    # Map: new_op_idx → old_op_idx
    mapping = []
    for new_idx, prim in enumerate(new_prims):
        old_idx = old_prims.index(prim)
        mapping.append(old_idx)

    with torch.no_grad():
        for alpha_name in ['alpha_normal', 'alpha_reduce']:
            old_alpha = getattr(old_model, alpha_name).data
            new_alpha = getattr(new_model, alpha_name).data

            for edge in range(new_alpha.shape[0]):
                for new_idx, old_idx in enumerate(mapping):
                    if edge < old_alpha.shape[0]:
                        new_alpha[edge, new_idx] = old_alpha[edge, old_idx]


# ─── Main Search ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P-DARTS Architecture Search")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset path")
    parser.add_argument("--split_path", type=str, default=None, help="Split JSON path")
    parser.add_argument("--output_dir", type=str, default=str(SEARCH_DIR))
    parser.add_argument("--batch_size", type=int, default=SEARCH_CFG["batch_size"])
    parser.add_argument("--epochs_per_stage", type=int, default=None,
                        help="Override epochs per stage (for quick test)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=SEARCH_CFG["num_workers"])
    parser.add_argument("--search_input_size", type=int, default=SEARCH_INPUT_SIZE,
                        help="Input image size during search (default: 96, smaller=faster)")
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("search", save_dir / "search.log")
    logger.info(f"P-DARTS Search Started at {datetime.now().isoformat()}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Search input size: {args.search_input_size}")

    # Data
    search_train_loader, search_val_loader, val_loader, test_loader, data_info = \
        create_search_dataloaders(
            data_dir=args.data_dir,
            split_path=args.split_path,
            batch_size=args.batch_size,
            input_size=args.search_input_size,
            num_workers=args.num_workers,
        )
    num_classes = data_info["num_classes"]
    logger.info(f"Classes: {num_classes}")
    logger.info(f"Search train: {data_info['search_train_size']}, "
                f"Search val: {data_info['search_val_size']}")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=SEARCH_CFG["label_smoothing"]).to(device)

    # ─── Progressive Search ───────────────────────────────────────────────
    current_primitives = list(PRIMITIVES)
    search_start = time.time()
    all_genotypes = []
    global_epoch = 0

    # CSV log
    log_path = save_dir / "search_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "global_epoch", "stage", "epoch", "w_loss", "a_loss", "train_acc",
        "val_loss", "val_acc", "lr", "num_ops", "epoch_time",
    ])

    for stage_idx, stage_cfg in enumerate(PDARTS_STAGES):
        num_cells = stage_cfg["cells"]
        num_ops_target = stage_cfg["num_ops"]
        epochs = args.epochs_per_stage or stage_cfg["epochs"]

        # Prune operations (except stage 1)
        if stage_idx > 0:
            new_primitives = prune_operations(model, current_primitives, num_ops_target)
            logger.info(f"\nPruned ops: {current_primitives} → {new_primitives}")
            old_primitives = current_primitives
            old_model = model
            current_primitives = new_primitives
        else:
            old_model = None
            old_primitives = None

        logger.info(f"\n{'='*60}")
        logger.info(f"  Stage {stage_idx + 1}/{len(PDARTS_STAGES)}")
        logger.info(f"  Cells: {num_cells}  |  Ops: {len(current_primitives)}  |  Epochs: {epochs}")
        logger.info(f"  Primitives: {current_primitives}")
        logger.info(f"{'='*60}")

        # Build search network for this stage
        model = SearchNetwork(
            C_init=SEARCH_CFG["C_search"],
            num_cells=num_cells,
            num_classes=num_classes,
            primitives=current_primitives,
        ).to(device)

        # Transfer alphas from previous stage
        if old_model is not None and old_primitives is not None:
            transfer_alphas(old_model, model, old_primitives, current_primitives)
            logger.info("Transferred alphas from previous stage")
            del old_model
            torch.cuda.empty_cache()

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Search model params: {total_params:,}")

        # Optimizers
        w_optimizer = torch.optim.SGD(
            model.weight_parameters(),
            lr=SEARCH_CFG["w_lr"],
            momentum=SEARCH_CFG["w_momentum"],
            weight_decay=SEARCH_CFG["w_weight_decay"],
        )
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            w_optimizer, T_max=epochs, eta_min=SEARCH_CFG["w_lr_min"]
        )

        architect = Architect(model, SEARCH_CFG)

        # Alpha logging
        alpha_log_normal = []
        alpha_log_reduce = []

        # ── Epoch loop for this stage ──
        for epoch in range(1, epochs + 1):
            global_epoch += 1

            # Skip-connect dropout (linearly increases across epochs in stage)
            skip_drop_progress = epoch / epochs
            skip_dropout_prob = (
                SEARCH_CFG["skip_dropout_start"] +
                (SEARCH_CFG["skip_dropout_end"] - SEARCH_CFG["skip_dropout_start"]) * skip_drop_progress
            )
            # During training, we apply dropout as a multiplier
            skip_dropout_mask = 1.0 - skip_dropout_prob if model.training else 1.0

            epoch_start = time.time()

            # Alpha warmup: only train weights for first N epochs
            alpha_warmup = SEARCH_CFG.get("alpha_warmup_epochs", 0)
            do_alpha_update = (epoch > alpha_warmup)
            if epoch == alpha_warmup + 1:
                logger.info(f"  ── Alpha warmup complete ({alpha_warmup} epochs). Starting alpha updates. ──")

            # Search epoch
            w_loss, a_loss, train_acc = search_epoch(
                model, search_train_loader, search_val_loader,
                criterion, w_optimizer, architect, device,
                skip_dropout_mask, SEARCH_CFG["grad_clip"],
                epoch, logger, update_alpha=do_alpha_update,
            )

            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            w_scheduler.step()
            current_lr = w_optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            # Log alphas
            alpha_log_normal.append(model.alpha_normal.detach().cpu().numpy().copy())
            alpha_log_reduce.append(model.alpha_reduce.detach().cpu().numpy().copy())

            # CSV log
            log_writer.writerow([
                global_epoch, stage_idx + 1, epoch,
                f"{w_loss:.6f}", f"{a_loss:.6f}", f"{train_acc:.4f}",
                f"{val_loss:.6f}", f"{val_acc:.4f}",
                f"{current_lr:.6f}", len(current_primitives), f"{epoch_time:.1f}",
            ])
            log_file.flush()

            # Print progress
            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                logger.info(
                    f"  S{stage_idx+1} E{epoch:>3}/{epochs} │ "
                    f"w_loss={w_loss:.4f}  a_loss={a_loss:.4f}  "
                    f"train_acc={train_acc:.4f} │ "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} │ "
                    f"lr={current_lr:.5f}  skip_drop={skip_dropout_prob:.2f}  "
                    f"{epoch_time:.1f}s"
                )

        # ── End of stage: derive genotype ──
        genotype = model.genotype()
        all_genotypes.append(genotype)

        logger.info(f"\nStage {stage_idx + 1} Genotype:")
        logger.info(f"  Normal: {genotype.normal}")
        logger.info(f"  Reduce: {genotype.reduce}")
        logger.info(model.alphas_summary())

        # Save stage results
        stage_dir = save_dir / f"stage_{stage_idx + 1}"
        stage_dir.mkdir(exist_ok=True)

        with open(stage_dir / "genotype.json", "w") as f:
            json.dump(genotype_to_dict(genotype), f, indent=2)

        # Alpha evolution plots
        plot_alpha_evolution(alpha_log_normal, current_primitives,
                            stage_dir / "alpha_normal.png", "normal")
        plot_alpha_evolution(alpha_log_reduce, current_primitives,
                            stage_dir / "alpha_reduce.png", "reduce")

        # Save alpha arrays
        np.save(stage_dir / "alpha_normal_log.npy", np.array(alpha_log_normal))
        np.save(stage_dir / "alpha_reduce_log.npy", np.array(alpha_log_reduce))

    log_file.close()

    # ─── Final Genotype ──────────────────────────────────────────────────
    final_genotype = all_genotypes[-1]

    # Enforce max skip-connect constraint (use ops from final search space)
    final_genotype = enforce_skip_limit(
        final_genotype, SEARCH_CFG["max_skip_connect"],
        final_primitives=current_primitives,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"  FINAL GENOTYPE (after skip-connect limit)")
    logger.info(f"{'='*60}")
    logger.info(f"Normal: {final_genotype.normal}")
    logger.info(f"Normal concat: {final_genotype.normal_concat}")
    logger.info(f"Reduce: {final_genotype.reduce}")
    logger.info(f"Reduce concat: {final_genotype.reduce_concat}")

    # Save final genotype
    with open(save_dir / "genotype_final.json", "w") as f:
        json.dump(genotype_to_dict(final_genotype), f, indent=2)

    # Visualise
    viz_text = visualize_genotype(final_genotype, save_dir / "genotype_final.png")
    logger.info(viz_text)

    # Estimate retrain params
    from model_eval import EvalNetwork, count_parameters, find_optimal_C_init, param_breakdown

    logger.info(f"\n{'='*60}")
    logger.info(f"  Parameter Estimates for Retrain")
    logger.info(f"{'='*60}")

    for C in [16, 20, 24, 28, 32]:
        test_model = EvalNetwork(final_genotype, C, 8, num_classes,
                                 auxiliary=False, dropout=0.3)
        n_params = count_parameters(test_model)
        logger.info(f"  C_init={C:>3}, cells=8  →  {n_params:>10,} params")
        del test_model

    # Find optimal C_init for target range
    best_C, best_params = find_optimal_C_init(
        final_genotype, 8, num_classes,
        target_min=250_000, target_max=400_000,
        auxiliary=True, dropout=0.3,
    )
    if best_C:
        logger.info(f"\n  Recommended: C_init={best_C} → {best_params:,} params")
        eval_model = EvalNetwork(final_genotype, best_C, 8, num_classes,
                                 auxiliary=True, dropout=0.3)
        logger.info(param_breakdown(eval_model))
        del eval_model

    total_time = time.time() - search_start
    logger.info(f"\nTotal search time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time_min": total_time / 60,
        "seed": args.seed,
        "stages": [
            {
                "cells": s["cells"],
                "num_ops": s["num_ops"],
                "epochs": args.epochs_per_stage or s["epochs"],
            }
            for s in PDARTS_STAGES
        ],
        "final_genotype": genotype_to_dict(final_genotype),
        "recommended_C_init": best_C,
        "estimated_params": best_params,
        "search_config": {k: str(v) for k, v in SEARCH_CFG.items()},
    }
    with open(save_dir / "search_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSearch complete! Results saved to {save_dir}")
    logger.info(f"Next step: python retrain.py --genotype {save_dir / 'genotype_final.json'}")


# ─── Skip-Connect Limiter ───────────────────────────────────────────────────

def enforce_skip_limit(genotype, max_skip=2, final_primitives=None):
    """
    Enforce maximum number of skip-connects per cell.
    Replace excess skip_connects with the strongest non-skip, non-none op
    from the final search space (not hardcoded).
    """
    # Determine replacement op from the actual search space
    if final_primitives is not None:
        candidates = [p for p in final_primitives if p not in ('none', 'skip_connect')]
    else:
        candidates = []
    # Fallback: use sep_conv_3x3 if no candidates found
    replacement_op = candidates[0] if candidates else 'sep_conv_3x3'

    def _limit(ops, max_skip):
        skip_count = sum(1 for op, _ in ops if op == 'skip_connect')
        if skip_count <= max_skip:
            return ops

        # Find which skips to keep (first max_skip)
        new_ops = list(ops)
        skip_seen = 0
        for i, (op, src) in enumerate(new_ops):
            if op == 'skip_connect':
                skip_seen += 1
                if skip_seen > max_skip:
                    new_ops[i] = (replacement_op, src)
        return new_ops

    return Genotype(
        normal=_limit(genotype.normal, max_skip),
        normal_concat=genotype.normal_concat,
        reduce=_limit(genotype.reduce, max_skip),
        reduce_concat=genotype.reduce_concat,
    )


if __name__ == "__main__":
    main()
