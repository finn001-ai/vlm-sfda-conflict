"""Dataset-level persistent class-pair flow for constrained target adaptation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _empty_flow_state(num_classes: int, rank: int):
    return {
        "counts": torch.zeros(num_classes, num_classes, dtype=torch.float),
        "seen_cycles": torch.zeros(num_classes, num_classes, dtype=torch.long),
        "basis": torch.zeros(rank, num_classes, dtype=torch.float),
        "pairs": [],
        "active_rank": 0,
        "frozen": False,
    }


@torch.no_grad()
def update_class_pair_flow(
    source_label,
    clip_label,
    resolved_label,
    resolved_mask,
    state,
    num_classes: int,
    rank: int,
    min_count: int,
    min_cycles: int,
):
    if rank <= 0 or rank >= num_classes:
        raise ValueError("Pair-flow rank must be in [1, num_classes - 1]")
    if min_count <= 0 or min_cycles <= 0:
        raise ValueError("Pair-flow persistence thresholds must be positive")
    if state is None:
        state = _empty_flow_state(num_classes, rank)
    if state["frozen"]:
        return state

    source_label = source_label.long().cpu()
    clip_label = clip_label.long().cpu()
    resolved_label = resolved_label.long().cpu()
    resolved_mask = resolved_mask.bool().cpu()
    conflict = source_label != clip_label
    candidate_resolved = (
        (resolved_label == source_label) | (resolved_label == clip_label)
    )
    valid = resolved_mask & conflict & candidate_resolved

    cycle_counts = torch.zeros(num_classes * num_classes, dtype=torch.float)
    if valid.any():
        winner = resolved_label[valid]
        loser = torch.where(
            winner == source_label[valid],
            clip_label[valid],
            source_label[valid],
        )
        flat_index = loser * num_classes + winner
        cycle_counts.scatter_add_(
            0, flat_index, torch.ones_like(flat_index, dtype=torch.float)
        )
    cycle_counts = cycle_counts.view(num_classes, num_classes)
    cycle_support = cycle_counts >= int(min_count)
    state["counts"] += cycle_counts
    state["seen_cycles"] += cycle_support.long()

    total_count = state["counts"] + state["counts"].t()
    net_count = state["counts"] - state["counts"].t()
    eligible = (state["seen_cycles"] >= int(min_cycles)) & (net_count > 0)
    eligible.fill_diagonal_(False)
    if not eligible.any():
        return state

    directional_dominance = net_count / total_count.clamp_min(1.0)
    score = directional_dominance * torch.log1p(total_count)
    candidate_score = score.masked_fill(~eligible, float("-inf")).flatten()
    candidate_indices = torch.argsort(candidate_score, descending=True)

    parent = list(range(num_classes))

    def find(class_index):
        while parent[class_index] != class_index:
            parent[class_index] = parent[parent[class_index]]
            class_index = parent[class_index]
        return class_index

    selected_pairs = []
    for flat_index in candidate_indices.tolist():
        if not torch.isfinite(candidate_score[flat_index]).item():
            break
        loser = flat_index // num_classes
        winner = flat_index % num_classes
        loser_root = find(loser)
        winner_root = find(winner)
        if loser_root == winner_root:
            continue
        parent[loser_root] = winner_root
        selected_pairs.append((loser, winner))
        if len(selected_pairs) == rank:
            break

    active_rank = len(selected_pairs)
    if active_rank == 0:
        return state

    basis = torch.zeros(rank, num_classes, dtype=torch.float)
    for basis_index, (loser, winner) in enumerate(selected_pairs):
        basis[basis_index, loser] = -1.0 / math.sqrt(2.0)
        basis[basis_index, winner] = 1.0 / math.sqrt(2.0)
    state["basis"] = basis
    state["pairs"] = selected_pairs
    state["active_rank"] = active_rank
    state["frozen"] = True
    return state


class ClassPairFlowAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        rank: int,
        max_gate: float,
        gate_init: float,
        epsilon: float,
    ):
        super().__init__()
        if rank <= 0 or rank >= num_classes:
            raise ValueError("Pair-flow rank must be in [1, num_classes - 1]")
        if not 0.0 < max_gate <= 1.0:
            raise ValueError("Pair-flow max gate must be in (0, 1]")
        self.max_gate = float(max_gate)
        self.epsilon = float(epsilon)
        self.coefficient = nn.Linear(feature_dim, rank, bias=False)
        nn.init.zeros_(self.coefficient.weight)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.register_buffer("basis", torch.zeros(rank, num_classes))

    @torch.no_grad()
    def set_basis(self, basis):
        if basis.shape != self.basis.shape:
            raise ValueError("Pair-flow basis shape changed")
        self.basis.copy_(basis.to(device=self.basis.device, dtype=self.basis.dtype))

    def effective_gate(self):
        return self.max_gate * torch.sigmoid(self.gate_logit.detach())

    def forward(self, features, source_logits):
        coefficient = torch.tanh(self.coefficient(features).float())
        residual = coefficient @ self.basis.float()
        source_scale = source_logits.detach().float().std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(self.epsilon)
        gate = self.max_gate * torch.sigmoid(self.gate_logit.float())
        delta = gate * source_scale * torch.tanh(residual)
        return source_logits + delta.to(source_logits.dtype)
