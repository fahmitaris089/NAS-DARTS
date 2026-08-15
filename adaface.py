"""AdaFace head shared by NAS retraining, distillation, and export."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFaceHead(nn.Module):
    def __init__(self, embedding_size: int, classnum: int, *, m: float = 0.4,
                 h: float = 0.333, s: float = 64.0, t_alpha: float = 0.01):
        super().__init__()
        self.embedding_size, self.classnum = int(embedding_size), int(classnum)
        self.m, self.h, self.s, self.t_alpha = float(m), float(h), float(s), float(t_alpha)
        self.weight = nn.Parameter(torch.empty(self.classnum, self.embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))

    def cosine_logits(self, embeddings):
        return F.linear(F.normalize(embeddings, dim=1, eps=1e-12),
                        F.normalize(self.weight, dim=1, eps=1e-12)) * self.s

    def forward(self, embeddings, labels=None):
        norms = torch.norm(embeddings, 2, dim=1, keepdim=True).clamp(1e-3, 100.0)
        cosine = F.linear(embeddings / norms, F.normalize(self.weight, dim=1, eps=1e-12))
        if labels is None:
            return cosine * self.s
        with torch.no_grad():
            mean, std = norms.detach().mean(), norms.detach().std(unbiased=False).clamp_min(1e-6)
            self.batch_mean.mul_(1 - self.t_alpha).add_(mean * self.t_alpha)
            self.batch_std.mul_(1 - self.t_alpha).add_(std * self.t_alpha)
            scaler = (((norms.detach() - self.batch_mean) / (self.batch_std + 1e-3)) * self.h).clamp(-1, 1)
        target = cosine.gather(1, labels[:, None])
        target = torch.cos(torch.acos(target.clamp(-1 + 1e-4, 1 - 1e-4)) - self.m * scaler)
        target = target - (self.m + self.m * scaler)
        output = cosine.clone()
        output.scatter_(1, labels[:, None], target)
        return output * self.s


def replace_linear_with_adaface(model, *, num_classes, m=0.4, h=0.333, s=64.0, t_alpha=0.01):
    if not isinstance(model.classifier, nn.Linear):
        raise TypeError("AdaFace replacement requires model.classifier to be nn.Linear")
    model.classifier = AdaFaceHead(model.classifier.in_features, num_classes, m=m, h=h, s=s, t_alpha=t_alpha)
    return model.classifier


class ArcFaceHead(nn.Module):
    """ArcFace classifier with optional K sub-centers per identity."""

    def __init__(self, embedding_size: int, classnum: int, *, m: float = 0.5,
                 s: float = 64.0, num_subcenters: int = 1):
        super().__init__()
        if num_subcenters < 1:
            raise ValueError("num_subcenters must be positive")
        self.embedding_size, self.classnum = int(embedding_size), int(classnum)
        self.m, self.s, self.num_subcenters = float(m), float(s), int(num_subcenters)
        self.weight = nn.Parameter(torch.empty(
            self.classnum * self.num_subcenters, self.embedding_size
        ))
        nn.init.xavier_uniform_(self.weight)

    def cosine_logits(self, embeddings):
        embeddings = F.normalize(embeddings, dim=1, eps=1e-12)
        weights = F.normalize(self.weight, dim=1, eps=1e-12)
        cosine = F.linear(embeddings, weights).view(
            embeddings.shape[0], self.classnum, self.num_subcenters
        )
        return cosine.max(dim=2).values * self.s

    def forward(self, embeddings, labels=None):
        cosine = self.cosine_logits(embeddings) / self.s
        if labels is None:
            return cosine * self.s
        target = cosine.gather(1, labels[:, None])
        target_margin = torch.cos(
            torch.acos(target.clamp(-1 + 1e-4, 1 - 1e-4)) + self.m
        )
        output = cosine.clone()
        output.scatter_(1, labels[:, None], target_margin)
        return output * self.s


def replace_linear_with_arcface(model, *, num_classes, m=0.5, s=64.0,
                                num_subcenters=1):
    if not isinstance(model.classifier, nn.Linear):
        raise TypeError("ArcFace replacement requires model.classifier to be nn.Linear")
    model.classifier = ArcFaceHead(
        model.classifier.in_features, num_classes, m=m, s=s,
        num_subcenters=num_subcenters,
    )
    return model.classifier
