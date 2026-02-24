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
    """

    def __init__(self, model, cfg):
        """
        Args:
            model: SearchNetwork instance
            cfg:   dict with keys a_lr, a_betas, a_weight_decay
        """
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.arch_parameters(),
            lr=cfg["a_lr"],
            betas=cfg["a_betas"],
            weight_decay=cfg["a_weight_decay"],
        )

    def step(self, input_val, target_val, criterion, skip_dropout_mask=None):
        """
        One step of architecture parameter update.

        Forward pass on validation data → compute loss → backprop to alphas.
        """
        self.optimizer.zero_grad()
        logits = self.model(input_val, skip_dropout_mask=skip_dropout_mask)
        loss = criterion(logits, target_val)
        loss.backward()
        self.optimizer.step()
        return loss.item()
