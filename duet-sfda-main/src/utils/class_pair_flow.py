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
        "fixed_candidates": False,
        "candidate_source": None,
        "candidate_clip": None,
        "last_valid_count": 0,
        "last_candidate_mass": 0.0,
    }


def _freeze_persistent_flow(state, num_classes: int, rank: int, min_cycles: int):
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
    fixed_candidates: bool = False,
):
    if rank <= 0 or rank >= num_classes:
        raise ValueError("Pair-flow rank must be in [1, num_classes - 1]")
    if min_count <= 0 or min_cycles <= 0:
        raise ValueError("Pair-flow persistence thresholds must be positive")
    source_label = source_label.long().cpu()
    clip_label = clip_label.long().cpu()
    resolved_label = resolved_label.long().cpu()
    resolved_mask = resolved_mask.bool().cpu()
    if state is None:
        state = _empty_flow_state(num_classes, rank)
        state["fixed_candidates"] = bool(fixed_candidates)
        if fixed_candidates:
            state["candidate_source"] = source_label.clone()
            state["candidate_clip"] = clip_label.clone()
    elif state["fixed_candidates"] != bool(fixed_candidates):
        raise ValueError("Pair-flow fixed-candidate mode changed after initialization")
    if state["frozen"]:
        return state

    if fixed_candidates:
        candidate_source = state["candidate_source"]
        candidate_clip = state["candidate_clip"]
        if candidate_source.numel() != resolved_label.numel():
            raise ValueError("Pair-flow sample count changed after fixing candidates")
    else:
        candidate_source = source_label
        candidate_clip = clip_label

    conflict = candidate_source != candidate_clip
    candidate_resolved = (
        (resolved_label == candidate_source) | (resolved_label == candidate_clip)
    )
    valid = resolved_mask & conflict & candidate_resolved
    state["last_valid_count"] = int(valid.sum().item())

    cycle_counts = torch.zeros(num_classes * num_classes, dtype=torch.float)
    if valid.any():
        winner = resolved_label[valid]
        loser = torch.where(
            winner == candidate_source[valid],
            candidate_clip[valid],
            candidate_source[valid],
        )
        flat_index = loser * num_classes + winner
        cycle_counts.scatter_add_(
            0, flat_index, torch.ones_like(flat_index, dtype=torch.float)
        )
    cycle_counts = cycle_counts.view(num_classes, num_classes)
    cycle_support = cycle_counts >= int(min_count)
    state["counts"] += cycle_counts
    state["seen_cycles"] += cycle_support.long()

    return _freeze_persistent_flow(state, num_classes, rank, min_cycles)


@torch.no_grad()
def update_soft_class_pair_flow(
    source_label,
    clip_label,
    resolved_prob,
    state,
    num_classes: int,
    rank: int,
    min_count: float,
    min_cycles: int,
    fixed_candidates: bool = False,
):
    if rank <= 0 or rank >= num_classes:
        raise ValueError("Pair-flow rank must be in [1, num_classes - 1]")
    if min_count <= 0 or min_cycles <= 0:
        raise ValueError("Pair-flow persistence thresholds must be positive")
    if resolved_prob.ndim != 2 or resolved_prob.size(1) != num_classes:
        raise ValueError("Resolved probability must be sample-by-class")

    source_label = source_label.long().cpu()
    clip_label = clip_label.long().cpu()
    resolved_prob = resolved_prob.float().cpu()
    if resolved_prob.size(0) != source_label.numel():
        raise ValueError("Pair-flow labels and probabilities have different samples")
    if state is None:
        state = _empty_flow_state(num_classes, rank)
        state["fixed_candidates"] = bool(fixed_candidates)
        if fixed_candidates:
            state["candidate_source"] = source_label.clone()
            state["candidate_clip"] = clip_label.clone()
    elif state["fixed_candidates"] != bool(fixed_candidates):
        raise ValueError("Pair-flow fixed-candidate mode changed after initialization")

    if fixed_candidates:
        candidate_source = state["candidate_source"]
        candidate_clip = state["candidate_clip"]
        if candidate_source.numel() != resolved_prob.size(0):
            raise ValueError("Pair-flow sample count changed after fixing candidates")
    else:
        candidate_source = source_label
        candidate_clip = clip_label

    conflict = candidate_source != candidate_clip
    rows = torch.nonzero(conflict, as_tuple=False).squeeze(1)
    state["last_valid_count"] = int(rows.numel())
    if rows.numel() > 0:
        source = candidate_source[rows]
        clip = candidate_clip[rows]
        source_mass = resolved_prob[rows, source]
        clip_mass = resolved_prob[rows, clip]
        state["last_candidate_mass"] = float(
            (source_mass + clip_mass).sum().item()
        )
    else:
        source = torch.empty(0, dtype=torch.long)
        clip = torch.empty(0, dtype=torch.long)
        source_mass = torch.empty(0, dtype=torch.float)
        clip_mass = torch.empty(0, dtype=torch.float)
        state["last_candidate_mass"] = 0.0
    if state["frozen"]:
        return state

    cycle_counts = torch.zeros(num_classes * num_classes, dtype=torch.float)
    if rows.numel() > 0:
        cycle_counts.scatter_add_(0, source * num_classes + clip, clip_mass)
        cycle_counts.scatter_add_(0, clip * num_classes + source, source_mass)
    cycle_counts = cycle_counts.view(num_classes, num_classes)
    state["counts"] += cycle_counts
    state["seen_cycles"] += (cycle_counts >= float(min_count)).long()
    return _freeze_persistent_flow(state, num_classes, rank, min_cycles)


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
