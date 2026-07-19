"""Bounded feature adaptation in sufficiently covered class-pair directions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassPairFeatureAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        rank: int,
        max_gate: float,
        gate_init: float,
        epsilon: float,
        min_active_rank: int = 1,
    ):
        super().__init__()
        if feature_dim <= 0 or rank <= 0:
            raise ValueError("Feature dimension and pair-feature rank must be positive")
        if min_active_rank <= 0 or min_active_rank > rank:
            raise ValueError("Minimum active rank must be in [1, rank]")
        if not 0.0 < max_gate <= 1.0:
            raise ValueError("Pair-feature max gate must be in (0, 1]")
        self.min_active_rank = int(min_active_rank)
        self.max_gate = float(max_gate)
        self.epsilon = float(epsilon)
        self.router = nn.Linear(feature_dim, rank, bias=False)
        nn.init.zeros_(self.router.weight)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.register_buffer("directions", torch.zeros(rank, feature_dim))
        self.register_buffer("active_rank", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def set_pairs(self, pairs, classifier_weight):
        if classifier_weight.ndim != 2:
            raise ValueError("Classifier weight must be a class-by-feature matrix")
        if classifier_weight.size(1) != self.directions.size(1):
            raise ValueError("Classifier and adapter feature dimensions differ")
        if len(pairs) > self.directions.size(0):
            raise ValueError("Pair count exceeds pair-feature rank")

        weight = classifier_weight.detach().to(
            device=self.directions.device, dtype=self.directions.dtype
        )
        directions = torch.zeros_like(self.directions)
        for index, (loser, winner) in enumerate(pairs):
            if loser == winner:
                raise ValueError("Class-pair direction must connect distinct classes")
            if min(loser, winner) < 0 or max(loser, winner) >= weight.size(0):
                raise ValueError("Class-pair index is outside classifier classes")
            directions[index] = F.normalize(
                weight[winner] - weight[loser], dim=0, eps=self.epsilon
            )
        self.directions.copy_(directions)
        self.active_rank.fill_(len(pairs))

    def is_effective(self):
        return int(self.active_rank.item()) >= self.min_active_rank

    def effective_gate(self):
        if not self.is_effective():
            return self.gate_logit.detach().new_zeros(())
        return self.max_gate * torch.sigmoid(self.gate_logit.detach())

    def forward(self, features, detach_delta: bool = False):
        if not self.is_effective():
            return features
        coefficient = torch.tanh(self.router(features).float())
        raw_delta = coefficient @ self.directions.float()
        raw_norm = raw_delta.norm(dim=1, keepdim=True)
        bounded_delta = raw_delta / raw_norm.clamp_min(1.0)
        feature_norm = features.detach().float().norm(
            dim=1, keepdim=True
        ).clamp_min(self.epsilon)
        gate = self.max_gate * torch.sigmoid(self.gate_logit.float())
        delta = gate * feature_norm * bounded_delta
        if detach_delta:
            delta = delta.detach()
        return features + delta.to(features.dtype)


def weighted_graph_temporal_kl(logits, target_prob, sample_weight, epsilon):
    """Weighted graph-temporal distillation with a differentiable zero case."""
    if logits.ndim != 2 or target_prob.shape != logits.shape:
        raise ValueError("Pair-feature logits and graph target must have equal 2D shape")
    if sample_weight.ndim != 1 or sample_weight.numel() != logits.size(0):
        raise ValueError("Pair-feature graph weights must have one value per sample")
    weights = sample_weight.detach().float().clamp_min(0.0)
    weight_sum = weights.sum()
    if float(weight_sum.item()) <= 0.0:
        return logits.sum() * 0.0
    per_sample = F.kl_div(
        F.log_softmax(logits.float(), dim=1),
        target_prob.detach().float(),
        reduction="none",
    ).sum(dim=1)
    return (per_sample * weights).sum() / weight_sum.clamp_min(float(epsilon))
