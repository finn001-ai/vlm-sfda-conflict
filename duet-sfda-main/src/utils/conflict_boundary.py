"""Label-free local-boundary signals for task/CLIP conflict diagnostics."""

from __future__ import annotations

import math

import numpy as np
import torch


def pairwise_first_order_boundary(
    logits: torch.Tensor,
    inputs: torch.Tensor,
    own_labels: torch.Tensor,
    other_labels: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return first-order radius, pair margin, and input-gradient norm.

    ``own_labels`` is the model's top-1 class and ``other_labels`` is the other
    model's conflicting top-1 class. Summing the per-sample margins is valid
    here because evaluation-mode networks do not couple samples across a batch.
    No parameter gradient is accumulated.
    """
    if logits.ndim != 2 or inputs.ndim < 2:
        raise ValueError("logits must be 2-D and inputs must include a batch dimension")
    if logits.size(0) != inputs.size(0):
        raise ValueError("logits and inputs must have the same batch size")
    if logits.size(0) == 0:
        raise ValueError("boundary distance requires at least one sample")
    if not inputs.requires_grad:
        raise ValueError("inputs must require gradients")
    if own_labels.shape != other_labels.shape or own_labels.numel() != logits.size(0):
        raise ValueError("label vectors must have one entry per sample")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    row = torch.arange(logits.size(0), device=logits.device)
    pair_margin = logits[row, own_labels] - logits[row, other_labels]
    input_gradient = torch.autograd.grad(
        pair_margin.sum(),
        inputs,
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]
    gradient_norm = input_gradient.flatten(start_dim=1).norm(p=2, dim=1)
    radius = pair_margin.detach().clamp_min(0.0) / gradient_norm.detach().clamp_min(epsilon)
    return radius, pair_margin.detach(), gradient_norm.detach()


def boundary_choice_and_separation(
    task_radius: torch.Tensor,
    clip_radius: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the larger radius and return its absolute log-ratio separation."""
    if task_radius.shape != clip_radius.shape or task_radius.ndim != 1:
        raise ValueError("task_radius and clip_radius must be same-shaped 1-D tensors")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    task_safe = task_radius.clamp_min(epsilon)
    clip_safe = clip_radius.clamp_min(epsilon)
    choose_task = task_safe >= clip_safe
    separation = torch.abs(torch.log(task_safe) - torch.log(clip_safe))
    return choose_task, separation


def fixed_fraction_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """Select a deterministic, label-free top fraction of a 1-D score vector."""
    if scores.ndim != 1:
        raise ValueError("scores must be a 1-D tensor")
    if scores.numel() == 0:
        raise ValueError("scores must not be empty")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, math.ceil(scores.numel() * fraction))
    order = torch.argsort(scores, descending=True, stable=True)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected[order[:count]] = True
    return selected


def route_conflict_probabilities(
    task_prob: torch.Tensor,
    clip_prob: torch.Tensor,
    conflict_mask: torch.Tensor,
    task_radius: torch.Tensor,
    clip_radius: torch.Tensor,
    *,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route a fixed top fraction of conflicts to one complete soft view.

    Non-selected samples retain released DUET's arithmetic fusion. The routing
    signal depends only on model probabilities and boundary radii; no target
    labels enter selection or branch choice.
    """
    if task_prob.shape != clip_prob.shape or task_prob.ndim != 2:
        raise ValueError("task_prob and clip_prob must be same-shaped 2-D tensors")
    sample_count = task_prob.size(0)
    for name, value in (
        ("conflict_mask", conflict_mask),
        ("task_radius", task_radius),
        ("clip_radius", clip_radius),
    ):
        if value.shape != (sample_count,):
            raise ValueError(f"{name} must have shape ({sample_count},)")
    if conflict_mask.dtype != torch.bool:
        raise TypeError("conflict_mask must be boolean")

    fused = (task_prob + clip_prob) / 2.0
    selected = torch.zeros(sample_count, dtype=torch.bool, device=task_prob.device)
    choose_task = torch.zeros_like(selected)
    separation = torch.zeros(sample_count, dtype=task_prob.dtype, device=task_prob.device)
    conflict_index = torch.nonzero(conflict_mask, as_tuple=False).flatten()
    if conflict_index.numel() == 0:
        return fused, selected, choose_task, separation

    local_choice, local_separation = boundary_choice_and_separation(
        task_radius[conflict_index],
        clip_radius[conflict_index],
    )
    local_selected = fixed_fraction_mask(local_separation, fraction)
    selected_index = conflict_index[local_selected]
    selected[selected_index] = True
    choose_task[conflict_index] = local_choice
    separation[conflict_index] = local_separation.to(task_prob.dtype)

    task_selected = selected & choose_task
    clip_selected = selected & ~choose_task
    fused[task_selected] = task_prob[task_selected]
    fused[clip_selected] = clip_prob[clip_selected]
    return fused, selected, choose_task, separation


def paired_accuracy_bootstrap_ci(
    candidate_correct: np.ndarray,
    baseline_correct: np.ndarray,
    *,
    repeats: int = 2_000,
    seed: int = 2_020,
    batch_size: int = 100,
) -> tuple[float, float]:
    """Paired-bootstrap 95% CI for candidate-minus-baseline accuracy in pp."""
    candidate = np.asarray(candidate_correct, dtype=np.float64)
    baseline = np.asarray(baseline_correct, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("correctness arrays must be same-shaped 1-D arrays")
    if candidate.size == 0:
        raise ValueError("correctness arrays must not be empty")
    if repeats <= 0 or batch_size <= 0:
        raise ValueError("repeats and batch_size must be positive")

    difference = candidate - baseline
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats, dtype=np.float64)
    written = 0
    while written < repeats:
        current = min(batch_size, repeats - written)
        sample_indices = rng.integers(0, difference.size, size=(current, difference.size))
        bootstrap[written : written + current] = difference[sample_indices].mean(axis=1) * 100.0
        written += current
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)
