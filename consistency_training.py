"""Shared loss components for deterministic consistency training."""

from __future__ import annotations

import torch.nn.functional as F


def js_consistency_loss(logits_a, logits_b, temperature: float = 4.0):
    """Return temperature-scaled Jensen-Shannon divergence between logits."""
    if temperature <= 0:
        raise ValueError("consistency temperature must be positive")
    log_p = F.log_softmax(logits_a / temperature, dim=1)
    log_q = F.log_softmax(logits_b / temperature, dim=1)
    p, q = log_p.exp(), log_q.exp()
    mean = 0.5 * (p + q)
    log_mean = mean.clamp_min(1e-12).log()
    return 0.5 * (
        F.kl_div(log_mean, p, reduction="batchmean")
        + F.kl_div(log_mean, q, reduction="batchmean")
    ) * (temperature ** 2)
