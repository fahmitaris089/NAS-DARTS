"""AdaFace classification head for palm-vein identification.

The adaptive margin is used only when labels are supplied during training.
Inference returns scaled cosine logits and therefore needs no labels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFaceHead(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        classnum: int,
        *,
        m: float = 0.4,
        h: float = 0.333,
        s: float = 64.0,
        t_alpha: float = 0.01,
    ) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.classnum = int(classnum)
        self.m = float(m)
        self.h = float(h)
        self.s = float(s)
        self.t_alpha = float(t_alpha)
        self.weight = nn.Parameter(torch.empty(self.classnum, self.embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))

    def cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
        weight = F.normalize(self.weight, p=2, dim=1, eps=1e-12)
        return F.linear(normalized, weight) * self.s

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        norms = torch.norm(embeddings, 2, dim=1, keepdim=True).clamp(1e-3, 100.0)
        normalized = embeddings / norms
        cosine = F.linear(normalized, F.normalize(self.weight, p=2, dim=1, eps=1e-12))
        if labels is None:
            return cosine * self.s

        safe_norms = norms.detach()
        with torch.no_grad():
            mean = safe_norms.mean()
            std = safe_norms.std(unbiased=False).clamp_min(1e-6)
            self.batch_mean.mul_(1.0 - self.t_alpha).add_(mean * self.t_alpha)
            self.batch_std.mul_(1.0 - self.t_alpha).add_(std * self.t_alpha)
            margin_scaler = ((safe_norms - self.batch_mean) / (self.batch_std + 1e-3))
            margin_scaler = (margin_scaler * self.h).clamp(-1.0, 1.0)

        target = cosine.gather(1, labels.view(-1, 1))
        g_angle = -self.m * margin_scaler
        theta = torch.acos(target.clamp(-1.0 + 1e-4, 1.0 - 1e-4))
        target_margin = torch.cos(theta + g_angle)
        g_add = self.m + self.m * margin_scaler
        target_margin = target_margin - g_add
        output = cosine.clone()
        output.scatter_(1, labels.view(-1, 1), target_margin)
        return output * self.s


def replace_linear_with_adaface(
    model: nn.Module,
    *,
    num_classes: int,
    m: float = 0.4,
    h: float = 0.333,
    s: float = 64.0,
    t_alpha: float = 0.01,
) -> AdaFaceHead:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise TypeError("AdaFace replacement requires model.classifier to be nn.Linear")
    head = AdaFaceHead(classifier.in_features, num_classes, m=m, h=h, s=s, t_alpha=t_alpha)
    model.classifier = head
    return head


class ArcFaceHead(nn.Module):
    """ArcFace classifier with optional K sub-centers per identity.

    For K>1 the maximum cosine across sub-centers is used both to select the
    target center during training and to produce one inference logit per class.
    """

    def __init__(self, embedding_size: int, classnum: int, *, m: float = 0.5,
                 s: float = 64.0, num_subcenters: int = 1) -> None:
        super().__init__()
        if num_subcenters < 1:
            raise ValueError("num_subcenters must be positive")
        self.embedding_size = int(embedding_size)
        self.classnum = int(classnum)
        self.m = float(m)
        self.s = float(s)
        self.num_subcenters = int(num_subcenters)
        self.weight = nn.Parameter(torch.empty(
            self.classnum * self.num_subcenters, self.embedding_size
        ))
        nn.init.xavier_uniform_(self.weight)

    def cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
        weight = F.normalize(self.weight, p=2, dim=1, eps=1e-12)
        cosine = F.linear(embeddings, weight).view(
            embeddings.shape[0], self.classnum, self.num_subcenters
        )
        return cosine.max(dim=2).values * self.s

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = self.cosine_logits(embeddings) / self.s
        if labels is None:
            return cosine * self.s
        target = cosine.gather(1, labels.view(-1, 1))
        theta = torch.acos(target.clamp(-1.0 + 1e-4, 1.0 - 1e-4))
        target_margin = torch.cos(theta + self.m)
        output = cosine.clone()
        output.scatter_(1, labels.view(-1, 1), target_margin)
        return output * self.s


def replace_linear_with_arcface(
    model: nn.Module, *, num_classes: int, m: float = 0.5,
    s: float = 64.0, num_subcenters: int = 1,
) -> ArcFaceHead:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise TypeError("ArcFace replacement requires model.classifier to be nn.Linear")
    head = ArcFaceHead(
        classifier.in_features, num_classes, m=m, s=s,
        num_subcenters=num_subcenters,
    )
    model.classifier = head
    return head
