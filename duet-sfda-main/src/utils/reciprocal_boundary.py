"""Reciprocal class-pair boundary learning for DCCL.

The module deliberately avoids assigning a dataset-level winner to a conflict
pair.  Each active unordered pair receives a sample-dependent signed residual,
and the residual is antisymmetric in logit space.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_cpu_long(values: torch.Tensor) -> torch.Tensor:
    return values.detach().to(device="cpu", dtype=torch.long)


def _unordered_pair_counts(
    source_label: torch.Tensor,
    clip_label: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    source = _as_cpu_long(source_label)
    clip = _as_cpu_long(clip_label)
    conflict = source != clip
    low = torch.minimum(source[conflict], clip[conflict])
    high = torch.maximum(source[conflict], clip[conflict])
    flat = low * num_classes + high
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def init_reciprocal_boundary_state(
    num_samples: int,
    num_classes: int,
) -> dict:
    return {
        "num_samples": int(num_samples),
        "num_classes": int(num_classes),
        "cycles_seen": 0,
        "anchor_label": torch.full((num_samples,), -1, dtype=torch.long),
        "anchor_streak": torch.zeros(num_samples, dtype=torch.long),
        "pair_cycle_support": torch.zeros(
            (num_classes, num_classes), dtype=torch.long
        ),
        "pair_total_support": torch.zeros(
            (num_classes, num_classes), dtype=torch.long
        ),
        "pairs": [],
        "frozen": False,
        "last_conflicts": 0,
        "last_stable_anchors": 0,
        "last_eligible_pairs": 0,
        "pair_anchor_counts": torch.zeros((0, 2), dtype=torch.long),
        "active_conflict_mask": torch.zeros(num_samples, dtype=torch.bool),
    }


def _update_stable_agreements(
    state: dict,
    source_label: torch.Tensor,
    clip_label: torch.Tensor,
) -> None:
    source = _as_cpu_long(source_label)
    clip = _as_cpu_long(clip_label)
    agreement = source == clip
    same = agreement & (state["anchor_label"] == source)
    state["anchor_streak"] = torch.where(
        same,
        state["anchor_streak"] + 1,
        torch.where(
            agreement,
            torch.ones_like(state["anchor_streak"]),
            torch.zeros_like(state["anchor_streak"]),
        ),
    )
    state["anchor_label"] = torch.where(
        agreement,
        source,
        torch.full_like(state["anchor_label"], -1),
    )


def _select_pairs(
    state: dict,
    *,
    max_pairs: int,
    min_cycles: int,
    min_anchors_per_side: int,
) -> tuple[list[tuple[int, int]], int]:
    stable = state["anchor_streak"] >= min_cycles
    anchor_counts = torch.bincount(
        state["anchor_label"][stable],
        minlength=state["num_classes"],
    )
    candidates = []
    for first in range(state["num_classes"]):
        for second in range(first + 1, state["num_classes"]):
            if (
                state["pair_cycle_support"][first, second] < min_cycles
                or anchor_counts[first] < min_anchors_per_side
                or anchor_counts[second] < min_anchors_per_side
            ):
                continue
            candidates.append(
                (
                    -int(state["pair_total_support"][first, second]),
                    first,
                    second,
                )
            )
    candidates.sort()
    pairs = [(first, second) for _, first, second in candidates[:max_pairs]]
    return pairs, len(candidates)


def _refresh_boundary_masks(
    state: dict,
    source_label: torch.Tensor,
    clip_label: torch.Tensor,
    *,
    stable_cycles: int,
) -> None:
    source = _as_cpu_long(source_label)
    clip = _as_cpu_long(clip_label)
    active_conflict = torch.zeros_like(source, dtype=torch.bool)
    pair_anchor_counts = []
    stable = state["anchor_streak"] >= stable_cycles
    for first, second in state["pairs"]:
        active_conflict |= (
            ((source == first) & (clip == second))
            | ((source == second) & (clip == first))
        )
        pair_anchor_counts.append(
            [
                int((stable & (state["anchor_label"] == first)).sum()),
                int((stable & (state["anchor_label"] == second)).sum()),
            ]
        )
    state["active_conflict_mask"] = active_conflict
    state["pair_anchor_counts"] = torch.tensor(
        pair_anchor_counts,
        dtype=torch.long,
    ).reshape(-1, 2)
    state["last_stable_anchors"] = int(stable.sum())


def update_reciprocal_boundary_state(
    source_label: torch.Tensor,
    clip_label: torch.Tensor,
    state: dict | None,
    *,
    num_classes: int,
    max_pairs: int,
    min_conflicts: int,
    min_cycles: int,
    min_anchors_per_side: int,
) -> dict:
    """Update stable agreement anchors and freeze persistent unordered pairs."""
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if min_conflicts <= 0 or min_cycles <= 0 or min_anchors_per_side <= 0:
        raise ValueError("boundary support thresholds must be positive")
    if source_label.shape != clip_label.shape:
        raise ValueError("source and CLIP labels must have the same shape")
    if state is None:
        state = init_reciprocal_boundary_state(source_label.numel(), num_classes)
    if (
        state["num_samples"] != source_label.numel()
        or state["num_classes"] != num_classes
    ):
        raise ValueError("reciprocal boundary state shape changed")

    _update_stable_agreements(state, source_label, clip_label)
    current_counts = _unordered_pair_counts(
        source_label,
        clip_label,
        num_classes,
    )
    state["cycles_seen"] += 1
    state["last_conflicts"] = int((source_label != clip_label).sum())

    if not state["frozen"]:
        state["pair_total_support"] += current_counts
        state["pair_cycle_support"] += (current_counts >= min_conflicts).long()
        if state["cycles_seen"] >= min_cycles:
            pairs, eligible_count = _select_pairs(
                state,
                max_pairs=max_pairs,
                min_cycles=min_cycles,
                min_anchors_per_side=min_anchors_per_side,
            )
            state["last_eligible_pairs"] = eligible_count
            if pairs:
                state["pairs"] = pairs
                state["frozen"] = True

    _refresh_boundary_masks(
        state,
        source_label,
        clip_label,
        stable_cycles=min_cycles,
    )
    return state


class ReciprocalBoundaryHead(nn.Module):
    """A zero-initialized antisymmetric residual over active class pairs."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int,
        max_pairs: int,
        max_shift: float,
        epsilon: float,
    ):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or num_classes <= 1:
            raise ValueError("invalid reciprocal boundary dimensions")
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        if not 0.0 < max_shift <= 1.0:
            raise ValueError("max_shift must be in (0, 1]")
        self.num_classes = int(num_classes)
        self.max_pairs = int(max_pairs)
        self.max_shift = float(max_shift)
        self.epsilon = float(epsilon)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.coefficient = nn.Linear(hidden_dim, max_pairs, bias=True)
        nn.init.zeros_(self.coefficient.weight)
        nn.init.zeros_(self.coefficient.bias)
        self.register_buffer(
            "pairs",
            torch.full((max_pairs, 2), -1, dtype=torch.long),
        )
        self.register_buffer(
            "class_degree",
            torch.ones(num_classes, dtype=torch.float),
        )
        self.register_buffer("active_count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def set_pairs(self, pairs: Iterable[tuple[int, int]]) -> None:
        pair_list = [(int(first), int(second)) for first, second in pairs]
        if len(pair_list) > self.max_pairs:
            raise ValueError("too many reciprocal boundary pairs")
        seen = set()
        for first, second in pair_list:
            if not 0 <= first < second < self.num_classes:
                raise ValueError("pairs must be unique ordered class indices")
            if (first, second) in seen:
                raise ValueError("duplicate reciprocal boundary pair")
            seen.add((first, second))
        self.pairs.fill_(-1)
        degree = torch.zeros_like(self.class_degree)
        for index, (first, second) in enumerate(pair_list):
            self.pairs[index, 0] = first
            self.pairs[index, 1] = second
            degree[first] += 1.0
            degree[second] += 1.0
        self.class_degree.copy_(degree.clamp_min(1.0))
        self.active_count.fill_(len(pair_list))

    def active_pairs(self) -> torch.Tensor:
        return self.pairs[: int(self.active_count.item())]

    def residual(
        self,
        features: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        active = int(self.active_count.item())
        if active == 0:
            return torch.zeros_like(base_logits)
        raw = self.coefficient(self.project(features))[:, :active]
        scale = base_logits.detach().float().std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(self.epsilon)
        base_probability = F.softmax(base_logits.detach().float(), dim=1)
        residual = torch.zeros_like(base_logits, dtype=torch.float)
        for index, pair in enumerate(self.active_pairs()):
            first = int(pair[0].item())
            second = int(pair[1].item())
            relevance = (
                base_probability[:, first] + base_probability[:, second]
            )
            delta = (
                self.max_shift
                * scale[:, 0]
                * relevance
                * torch.tanh(raw[:, index].float())
            )
            pair_degree = torch.maximum(
                self.class_degree[first],
                self.class_degree[second],
            )
            residual[:, first] += delta / pair_degree
            residual[:, second] -= delta / pair_degree
        return residual.to(dtype=base_logits.dtype)

    def forward(
        self,
        features: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        detach_residual: bool = False,
    ) -> torch.Tensor:
        residual = self.residual(features, base_logits)
        if detach_residual:
            residual = residual.detach()
        return base_logits + residual


def reciprocal_boundary_margin_loss(
    corrected_logits: torch.Tensor,
    base_logits: torch.Tensor,
    anchor_label: torch.Tensor,
    anchor_mask: torch.Tensor,
    pairs: torch.Tensor,
    pair_anchor_counts: torch.Tensor,
    *,
    total_samples: int,
    margin: float,
    epsilon: float,
) -> torch.Tensor:
    """Unbiased minibatch estimate of a two-side-balanced pair margin loss."""
    zero = corrected_logits.sum() * 0.0
    if pairs.numel() == 0:
        return zero
    if pair_anchor_counts.shape != pairs.shape:
        raise ValueError("pair anchor counts must match active pairs")
    scale = base_logits.detach().float().std(
        dim=1,
        unbiased=False,
    ).clamp_min(epsilon)
    residual_logits = corrected_logits - base_logits.detach()
    batch_factor = float(total_samples) / float(max(corrected_logits.size(0), 1))
    total = zero
    active_terms = 0
    for index, pair in enumerate(pairs):
        first = int(pair[0].item())
        second = int(pair[1].item())
        pair_margin = (
            residual_logits[:, first] - residual_logits[:, second]
        ) / scale
        first_mask = anchor_mask & (anchor_label == first)
        second_mask = anchor_mask & (anchor_label == second)
        first_count = int(pair_anchor_counts[index, 0].item())
        second_count = int(pair_anchor_counts[index, 1].item())
        if first_count > 0:
            total = total + (
                0.5
                * batch_factor
                * F.softplus(margin - pair_margin[first_mask]).sum()
                / float(first_count)
            )
            active_terms += 1
        if second_count > 0:
            total = total + (
                0.5
                * batch_factor
                * F.softplus(margin + pair_margin[second_mask]).sum()
                / float(second_count)
            )
            active_terms += 1
    if active_terms == 0:
        return zero
    return total / float(max(pairs.size(0), 1))


def reciprocal_boundary_consistency_loss(
    weak_logits: torch.Tensor,
    strong_logits: torch.Tensor,
    base_weak_logits: torch.Tensor,
    base_strong_logits: torch.Tensor,
    source_label: torch.Tensor,
    clip_label: torch.Tensor,
    pairs: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Stabilize only the relative margin of matching conflict pairs."""
    zero = weak_logits.sum() * 0.0
    if pairs.numel() == 0:
        return zero
    losses = []
    weak_scale = base_weak_logits.detach().float().std(
        dim=1,
        unbiased=False,
    ).clamp_min(epsilon)
    strong_scale = base_strong_logits.detach().float().std(
        dim=1,
        unbiased=False,
    ).clamp_min(epsilon)
    for pair in pairs:
        first = int(pair[0].item())
        second = int(pair[1].item())
        selected = (
            ((source_label == first) & (clip_label == second))
            | ((source_label == second) & (clip_label == first))
        )
        if not selected.any():
            continue
        weak_margin = (
            weak_logits[selected, first] - weak_logits[selected, second]
        ) / weak_scale[selected]
        strong_margin = (
            strong_logits[selected, first] - strong_logits[selected, second]
        ) / strong_scale[selected]
        losses.append(F.smooth_l1_loss(weak_margin, strong_margin))
    return torch.stack(losses).mean() if losses else zero


def reciprocal_boundary_preservation_loss(
    corrected_logits: torch.Tensor,
    base_logits: torch.Tensor,
    preserve_mask: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Keep the boundary residual near zero outside active conflicts."""
    if not preserve_mask.any():
        return corrected_logits.sum() * 0.0
    scale = base_logits.detach().float().std(
        dim=1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(epsilon)
    normalized_residual = (corrected_logits - base_logits) / scale
    return normalized_residual[preserve_mask].pow(2).mean()
