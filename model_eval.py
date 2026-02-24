"""
Evaluation Network — Derived Architecture from Genotype
========================================================
Builds a discrete (non-mixed) network from the discovered Genotype.
Used for retraining from scratch and final evaluation.

Key differences from SearchNetwork:
  - Each edge has exactly ONE operation (not MixedOp)
  - BN uses affine=True
  - DropPath for regularisation
  - Optional auxiliary head at 2/3 depth
  - Configurable C_init and num_cells to hit param target
"""

import torch
import torch.nn as nn

from operations import OPS, DropPath, FactorizedReduce, Identity
from nas_config import NUM_NODES, NUM_INPUT_NODES, TOP_K_EDGES


# ─── Discrete Cell ───────────────────────────────────────────────────────────

class EvalCell(nn.Module):
    """
    A single cell built from a discrete genotype.

    Each intermediate node has exactly TOP_K_EDGES incoming edges,
    each with a single selected operation.
    """

    def __init__(self, genotype_ops, C_pp, C_p, C, reduction, reduction_prev):
        """
        Args:
            genotype_ops: list of (op_name, input_node_idx) for this cell type
            C_pp, C_p, C: channel counts
            reduction:     True if reduction cell
            reduction_prev: True if previous cell was reduction
        """
        super().__init__()
        self.reduction = reduction
        self.num_nodes = NUM_NODES

        # Preprocessing
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_pp, C, affine=True)
        else:
            self.preprocess0 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_pp, C, 1, bias=False),
                nn.BatchNorm2d(C, affine=True),
            )
        self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_p, C, 1, bias=False),
            nn.BatchNorm2d(C, affine=True),
        )

        # Build discrete ops from genotype
        self._ops = nn.ModuleList()
        self._indices = []  # source node index for each op

        for i in range(NUM_NODES):
            for j in range(TOP_K_EDGES):
                idx = i * TOP_K_EDGES + j
                op_name, src_node = genotype_ops[idx]
                stride = 2 if reduction and src_node < NUM_INPUT_NODES else 1
                op = OPS[op_name](C, stride, affine=True)
                self._ops.append(op)
                self._indices.append(src_node)

        self.drop_path = DropPath()

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]

        for i in range(self.num_nodes):
            # Sum the TOP_K_EDGES inputs for this node
            node_inputs = []
            for j in range(TOP_K_EDGES):
                op_idx = i * TOP_K_EDGES + j
                src = self._indices[op_idx]
                h = self._ops[op_idx](states[src])
                # Apply drop path to non-identity ops
                if not isinstance(self._ops[op_idx], Identity):
                    h = self.drop_path(h)
                node_inputs.append(h)
            states.append(sum(node_inputs))

        # Concat intermediate nodes
        return torch.cat(states[NUM_INPUT_NODES:], dim=1)

    def set_drop_path_prob(self, prob):
        self.drop_path.drop_prob = prob


# ─── Auxiliary Head ──────────────────────────────────────────────────────────

class AuxiliaryHead(nn.Module):
    """
    Auxiliary classifier at 2/3 depth for training stability.
    Small overhead, helps gradient flow in deep networks.
    """

    def __init__(self, C, num_classes, input_size=224):
        super().__init__()
        # Estimate feature map size at 2/3 depth
        # After 2 reductions from 224: 224→112→56, at 2/3 we're at 56
        # But with stem not reducing, it depends on architecture
        # Use adaptive pooling to handle any size
        self.features = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.AvgPool2d(5, stride=3, padding=0),  # reduce spatial
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─── Evaluation Network ─────────────────────────────────────────────────────

class EvalNetwork(nn.Module):
    """
    Full discrete network built from a Genotype.

    Architecture:
      Stem → N Cells (with reduction at 1/3, 2/3) → GAP → Dropout → Linear

    Configurable C_init and num_cells to hit parameter budget.
    """

    def __init__(self, genotype, C_init, num_cells, num_classes,
                 auxiliary=False, dropout=0.3):
        """
        Args:
            genotype:    Genotype namedtuple
            C_init:      initial channels (controls total params)
            num_cells:   number of cells
            num_classes: output classes (834)
            auxiliary:   use auxiliary head at 2/3 depth
            dropout:     classifier dropout rate
        """
        super().__init__()
        self.num_cells = num_cells
        self.auxiliary = auxiliary
        self._auxiliary_head = None
        self.drop_path_prob = 0.0

        C_curr = C_init * 3  # stem output

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )

        # Build cells
        C_pp, C_p = C_curr, C_curr
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(num_cells):
            reduction = (i in [num_cells // 3, 2 * num_cells // 3])

            if i < num_cells // 3:
                C = C_init
            elif i < 2 * num_cells // 3:
                C = C_init * 2
            else:
                C = C_init * 4

            if reduction:
                cell_ops = genotype.reduce
            else:
                cell_ops = genotype.normal

            cell = EvalCell(cell_ops, C_pp, C_p, C, reduction, reduction_prev)
            self.cells.append(cell)

            reduction_prev = reduction
            C_pp = C_p
            C_p = C * NUM_NODES  # concat

            # Auxiliary head at 2/3 depth
            if auxiliary and i == 2 * num_cells // 3:
                self._auxiliary_head = AuxiliaryHead(C_p, num_classes)

        # Classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(C_p, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        logits_aux = None

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if self.auxiliary and self.training and i == 2 * self.num_cells // 3:
                if self._auxiliary_head is not None:
                    logits_aux = self._auxiliary_head(s1)

        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        logits = self.classifier(out)

        if self.training and logits_aux is not None:
            return logits, logits_aux
        return logits

    def set_drop_path_prob(self, prob):
        """Set drop path probability for all cells (scheduled during training)."""
        self.drop_path_prob = prob
        for cell in self.cells:
            cell.set_drop_path_prob(prob)


# ─── Param Budget Utilities ─────────────────────────────────────────────────

def count_parameters(model, trainable_only=True):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def find_optimal_C_init(genotype, num_cells, num_classes,
                        target_min=250_000, target_max=400_000,
                        auxiliary=False, dropout=0.3):
    """
    Binary search for C_init that puts param count in target range.

    Returns: (best_C_init, param_count)
    """
    best = None
    best_params = 0

    for C in range(8, 64, 2):
        model = EvalNetwork(genotype, C, num_cells, num_classes,
                            auxiliary=auxiliary, dropout=dropout)
        n_params = count_parameters(model)

        if target_min <= n_params <= target_max:
            if best is None or abs(n_params - (target_min + target_max) // 2) < \
               abs(best_params - (target_min + target_max) // 2):
                best = C
                best_params = n_params

        if n_params > target_max * 1.5:
            break

    if best is None:
        # Find closest below target_max
        for C in range(8, 64, 2):
            model = EvalNetwork(genotype, C, num_cells, num_classes,
                                auxiliary=auxiliary, dropout=dropout)
            n_params = count_parameters(model)
            if n_params <= target_max:
                best = C
                best_params = n_params

    return best, best_params


def param_breakdown(model):
    """Print parameter breakdown by module."""
    lines = [f"\n{'Module':<40} {'Params':>12} {'%':>6}"]
    lines.append("-" * 60)

    total = count_parameters(model, trainable_only=False)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        pct = 100.0 * n / total if total > 0 else 0
        lines.append(f"  {name:<38} {n:>12,} {pct:>5.1f}%")

    lines.append("-" * 60)
    lines.append(f"  {'TOTAL':<38} {total:>12,} 100.0%")
    return "\n".join(lines)
