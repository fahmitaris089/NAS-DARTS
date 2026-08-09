"""
Architect — Bilevel Optimisation for Architecture Parameters
=============================================================
First-order approximation (P-DARTS default):
  1. Update weights w on train batch  (SGD)
  2. Update alphas α on val batch     (Adam)

We use ∂L_val/∂α directly (no Hessian), which is more stable for
progressive search and sufficient for finding good architectures.
"""

import torch


class Architect:
    """
    Handles architecture parameter (alpha) updates using a separate
    Adam optimizer on the validation split.

    Optional hardware-aware latency penalty:
        L = L_CE + oplat_lambda * Σ_edge Σ_op softmax(α)[op] * cost[op]
    where ``cost`` is either an op-count proxy (Tier-1) or device-measured
    latency from a Raspberry Pi LUT (Tier-2). The penalty is differentiable
    w.r.t. α, so it biases the search toward low-latency operators.
    """

    def __init__(self, model, cfg, primitives=None, op_cost=None,
                 oplat_lambda=0.0, device=None):
        """
        Args:
            model:        SearchNetwork instance
            cfg:          dict with keys a_lr, a_betas, a_weight_decay
            primitives:   list of op names in the current search space (order
                          must match the alpha columns). Required if oplat_lambda>0.
            op_cost:      dict {op_name: cost}. Required if oplat_lambda>0.
            oplat_lambda: penalty weight (0 disables the penalty entirely).
            device:       torch device for the cost vector.
        """
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.arch_parameters(),
            lr=cfg["a_lr"],
            betas=cfg["a_betas"],
            weight_decay=cfg["a_weight_decay"],
        )

        self.oplat_lambda = float(oplat_lambda)
        self.cost_vec = None
        if self.oplat_lambda > 0.0:
            if primitives is None or op_cost is None:
                raise ValueError("primitives and op_cost are required when oplat_lambda > 0")
            costs = [float(op_cost.get(p, 0.0)) for p in primitives]
            cost_vec = torch.tensor(costs, dtype=torch.float32)
            # Normalize by max cost so lambda is comparable across cost sources
            max_c = cost_vec.max()
            if max_c > 0:
                cost_vec = cost_vec / max_c
            if device is not None:
                cost_vec = cost_vec.to(device)
            self.cost_vec = cost_vec

    def latency_penalty(self):
        """Differentiable expected-cost penalty over all edges (normal+reduce)."""
        if self.cost_vec is None:
            return None
        pen = 0.0
        for alphas in (self.model.alpha_normal, self.model.alpha_reduce):
            w = torch.softmax(alphas, dim=-1)            # (num_edges, num_ops)
            pen = pen + (w * self.cost_vec).sum(dim=-1).mean()
        return pen / 2.0

    def step(self, input_val, target_val, criterion, skip_dropout_mask=None):
        """
        One step of architecture parameter update.

        Forward pass on validation data → compute loss → backprop to alphas.
        Returns (total_loss, ce_loss, penalty_value).
        """
        self.optimizer.zero_grad()
        logits = self.model(input_val, skip_dropout_mask=skip_dropout_mask)
        ce_loss = criterion(logits, target_val)

        penalty = self.latency_penalty()
        if penalty is not None:
            loss = ce_loss + self.oplat_lambda * penalty
            pen_val = float(penalty.item())
        else:
            loss = ce_loss
            pen_val = 0.0

        loss.backward()
        self.optimizer.step()
        return loss.item(), float(ce_loss.item()), pen_val
