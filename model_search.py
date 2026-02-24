"""
Search Network (Supernet) — P-DARTS
=====================================
One-shot supernet where every edge is a MixedOp (weighted sum of all
candidate operations). Architecture parameters (alphas) are optimised
jointly with network weights via bilevel optimisation.

Key components:
  MixedOp  — single edge with all ops weighted by softmax(alpha)
  Cell     — DAG of MixedOp edges with 4 intermediate nodes
  SearchNetwork — full supernet: stem → cells → classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from operations import OPS, FactorizedReduce
from nas_config import PRIMITIVES, NUM_NODES, NUM_INPUT_NODES, TOP_K_EDGES


# ─── MixedOp ─────────────────────────────────────────────────────────────────

class MixedOp(nn.Module):
    """
    A single edge containing ALL candidate operations.
    During forward: output = sum( softmax(alpha_i) * op_i(x) )

    For P-DARTS, ops can be pruned between stages (via active_ops mask).
    """

    def __init__(self, C, stride, primitives=None):
        super().__init__()
        self.primitives = primitives or PRIMITIVES
        self._ops = nn.ModuleList()
        for prim in self.primitives:
            op = OPS[prim](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights, skip_dropout=None):
        """
        Args:
            x:      input tensor
            weights: softmax(alpha) for this edge, shape (num_ops,)
            skip_dropout: optional dropout applied to skip_connect weight
        """
        result = 0.0
        for i, (w, op) in enumerate(zip(weights, self._ops)):
            # Apply skip-connect dropout (P-DARTS regularisation)
            if skip_dropout is not None and self.primitives[i] == 'skip_connect':
                w = w * skip_dropout
            result = result + w * op(x)
        return result


# ─── Cell ─────────────────────────────────────────────────────────────────────

class Cell(nn.Module):
    """
    A single cell (normal or reduction) in the search network.

    DAG structure with NUM_NODES intermediate nodes:
      - Node 0, 1: inputs (c_{k-2}, c_{k-1}) after preprocessing
      - Node 2..5: intermediate, each receives edges from all prior nodes
      - Output: concat(intermediate_nodes) → 1×1 conv to adjust channels

    For reduction cell, stride=2 on edges from input nodes.
    """

    def __init__(self, num_nodes, C_pp, C_p, C, reduction, reduction_prev, primitives=None):
        """
        Args:
            num_nodes:      number of intermediate nodes (default 4)
            C_pp:           channels of c_{k-2}
            C_p:            channels of c_{k-1}
            C:              cell output channels
            reduction:      True if this is a reduction cell (stride=2)
            reduction_prev: True if previous cell was reduction
            primitives:     list of op names (can be pruned between stages)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.reduction = reduction
        self.primitives = primitives or PRIMITIVES

        # Preprocessing: adjust input channels to C
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preprocess0 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_pp, C, 1, bias=False),
                nn.BatchNorm2d(C, affine=False),
            )
        self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_p, C, 1, bias=False),
            nn.BatchNorm2d(C, affine=False),
        )

        # Build edges: for each intermediate node, edges from all prior nodes
        self._ops = nn.ModuleList()
        self._edge_indices = []  # (from_node, to_node) for each edge group

        for i in range(num_nodes):  # intermediate nodes 0..3
            node_ops = nn.ModuleList()
            for j in range(NUM_INPUT_NODES + i):  # all prior nodes
                stride = 2 if reduction and j < NUM_INPUT_NODES else 1
                op = MixedOp(C, stride, primitives=self.primitives)
                node_ops.append(op)
            self._ops.append(node_ops)
            self._edge_indices.append(list(range(NUM_INPUT_NODES + i)))

    def forward(self, s0, s1, alphas, skip_dropout_mask=None):
        """
        Args:
            s0, s1:      outputs from cell k-2 and k-1
            alphas:      architecture weights, shape (num_edges, num_ops)
            skip_dropout_mask: scalar dropout factor for skip-connect
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]

        offset = 0
        for i in range(self.num_nodes):
            # Sum edges from all prior nodes to this node
            edge_outputs = []
            for j, src_idx in enumerate(self._edge_indices[i]):
                edge_weights = F.softmax(alphas[offset + j], dim=-1)
                edge_out = self._ops[i][j](states[src_idx], edge_weights, skip_dropout_mask)
                edge_outputs.append(edge_out)
            node_out = sum(edge_outputs)
            states.append(node_out)
            offset += len(self._edge_indices[i])

        # Concat all intermediate nodes
        return torch.cat(states[NUM_INPUT_NODES:], dim=1)


# ─── Search Network ──────────────────────────────────────────────────────────

class SearchNetwork(nn.Module):
    """
    Full P-DARTS supernet.

    Architecture:
      Stem (3→C_init*3) → N Cells → GAP → Dropout → Linear(834)

    Reduction cells at 1/3 and 2/3 positions (channel doubling).
    Architecture parameters (alpha_normal, alpha_reduce) are separate
    nn.Parameters optimised by the Architect.
    """

    def __init__(self, C_init, num_cells, num_classes, primitives=None, num_nodes=NUM_NODES):
        """
        Args:
            C_init:      initial number of channels
            num_cells:   total number of cells
            num_classes: classification output (834)
            primitives:  list of op names (pruned across stages)
            num_nodes:   intermediate nodes per cell
        """
        super().__init__()
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.primitives = primitives or PRIMITIVES
        self.C_init = C_init

        C_curr = C_init * 3  # stem output channels (DARTS convention)

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
            # Reduction at 1/3 and 2/3
            reduction = (i in [num_cells // 3, 2 * num_cells // 3])
            C = C_init * (2 ** sum(1 for r in range(num_cells)
                                   if r in [num_cells // 3, 2 * num_cells // 3]
                                   and r <= i))

            # Simpler channel calculation
            if i < num_cells // 3:
                C = C_init
            elif i < 2 * num_cells // 3:
                C = C_init * 2
            else:
                C = C_init * 4

            cell = Cell(num_nodes, C_pp, C_p, C, reduction, reduction_prev,
                        primitives=self.primitives)
            self.cells.append(cell)

            reduction_prev = reduction
            C_pp = C_p
            C_p = C * num_nodes  # concat of all intermediate nodes

        # Classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_p, num_classes)

        # ── Architecture parameters ──
        self._init_alphas()

    def _init_alphas(self):
        """Initialise architecture parameters (alpha) for normal & reduce cells."""
        num_ops = len(self.primitives)
        # Total edges: for 4 intermediate nodes = 2+3+4+5 = 14
        num_edges = sum(NUM_INPUT_NODES + i for i in range(self.num_nodes))

        self.alpha_normal = nn.Parameter(
            1e-3 * torch.randn(num_edges, num_ops))
        self.alpha_reduce = nn.Parameter(
            1e-3 * torch.randn(num_edges, num_ops))

    def arch_parameters(self):
        """Return architecture parameters (for architect optimizer)."""
        return [self.alpha_normal, self.alpha_reduce]

    def weight_parameters(self):
        """Return network weight parameters (exclude alphas)."""
        arch_param_ids = {id(p) for p in self.arch_parameters()}
        return [p for p in self.parameters() if id(p) not in arch_param_ids]

    def forward(self, x, skip_dropout_mask=None):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            alphas = self.alpha_reduce if cell.reduction else self.alpha_normal
            s0, s1 = s1, cell(s0, s1, alphas, skip_dropout_mask)

        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits

    def genotype(self):
        """
        Derive discrete architecture (Genotype) from continuous alphas.

        For each intermediate node, keep TOP_K_EDGES strongest edges
        (by max softmax weight, excluding 'none').
        """
        from genotypes import Genotype

        def _parse(alphas):
            gene = []
            offset = 0
            for i in range(self.num_nodes):
                num_edges_for_node = NUM_INPUT_NODES + i
                edges = []
                for j in range(num_edges_for_node):
                    W = F.softmax(alphas[offset + j], dim=-1).detach().cpu()
                    # Score each op (exclude 'none')
                    for k, prim in enumerate(self.primitives):
                        if prim != 'none':
                            edges.append((W[k].item(), prim, j))
                offset += num_edges_for_node

                # Keep top-K edges for this node (by weight)
                edges_sorted = sorted(edges, key=lambda x: -x[0])

                # Select top-K with unique source nodes preferred
                selected = []
                selected_sources = set()
                for w_val, op_name, src_node in edges_sorted:
                    if len(selected) >= TOP_K_EDGES:
                        break
                    if src_node not in selected_sources:
                        selected.append((op_name, src_node))
                        selected_sources.add(src_node)

                # If not enough unique sources, fill from remaining
                if len(selected) < TOP_K_EDGES:
                    for w_val, op_name, src_node in edges_sorted:
                        if len(selected) >= TOP_K_EDGES:
                            break
                        if (op_name, src_node) not in selected:
                            selected.append((op_name, src_node))

                gene.extend(selected)

            return gene

        gene_normal = _parse(self.alpha_normal)
        gene_reduce = _parse(self.alpha_reduce)

        concat = list(range(NUM_INPUT_NODES, NUM_INPUT_NODES + self.num_nodes))

        return Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat,
        )

    def alphas_summary(self):
        """Print human-readable alpha weights for debugging."""
        lines = []
        for name, alphas in [("Normal", self.alpha_normal),
                              ("Reduce", self.alpha_reduce)]:
            lines.append(f"\n{'='*50}")
            lines.append(f"  {name} Cell Alphas")
            lines.append(f"{'='*50}")
            W = F.softmax(alphas, dim=-1).detach().cpu()
            offset = 0
            for i in range(self.num_nodes):
                lines.append(f"\n  Node {i + NUM_INPUT_NODES}:")
                for j in range(NUM_INPUT_NODES + i):
                    weights = W[offset + j]
                    ops_str = "  ".join(
                        f"{self.primitives[k][:10]:>10}={weights[k]:.3f}"
                        for k in range(len(self.primitives))
                    )
                    lines.append(f"    edge({j}→{i+NUM_INPUT_NODES}): {ops_str}")
                offset += NUM_INPUT_NODES + i
        return "\n".join(lines)
