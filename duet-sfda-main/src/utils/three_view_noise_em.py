"""Class-conditional noise EM for source, CLIP, and graph views."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _normalize_probability(probability: torch.Tensor, eps: float) -> torch.Tensor:
    probability = probability.float().clamp_min(eps)
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(eps)


def _validate_inputs(
    views: list[torch.Tensor],
    base_probability: torch.Tensor,
    anchor_mask: torch.Tensor,
    anchor_label: torch.Tensor,
) -> tuple[int, int]:
    if len(views) != 3:
        raise ValueError("three-view EM requires exactly three probability views")
    if base_probability.ndim != 2:
        raise ValueError("base_probability must have shape [samples, classes]")
    num_samples, num_classes = base_probability.shape
    if num_samples == 0 or num_classes < 2:
        raise ValueError("three-view EM requires non-empty multi-class inputs")
    if any(view.shape != base_probability.shape for view in views):
        raise ValueError("all probability views must match base_probability")
    if anchor_mask.shape != (num_samples,) or anchor_label.shape != (num_samples,):
        raise ValueError("anchor tensors must have shape [samples]")
    if anchor_mask.any():
        selected = anchor_label[anchor_mask]
        if int(selected.min()) < 0 or int(selected.max()) >= num_classes:
            raise ValueError("anchor labels are outside the class range")
    return num_samples, num_classes


def _estimate_transition(
    posterior: torch.Tensor,
    view: torch.Tensor,
    dirichlet: float,
    supported_classes: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    num_classes = posterior.size(1)
    identity = torch.eye(num_classes, dtype=posterior.dtype, device=posterior.device)
    counts = posterior.t().matmul(view) + float(dirichlet) * identity
    transition = counts / counts.sum(dim=1, keepdim=True).clamp_min(eps)
    return torch.where(supported_classes.unsqueeze(1), transition, identity)


@torch.no_grad()
def three_view_class_conditional_em(
    source_probability: torch.Tensor,
    clip_probability: torch.Tensor,
    graph_probability: torch.Tensor,
    base_probability: torch.Tensor,
    anchor_mask: torch.Tensor,
    anchor_label: torch.Tensor,
    conflict_mask: torch.Tensor,
    *,
    steps: int = 5,
    dirichlet: float = 5.0,
    min_class_anchors: int = 3,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Infer a soft latent label posterior without target ground-truth labels.

    Stable source/CLIP agreements clamp a subset of latent labels. The three
    view-specific class transition matrices are then estimated jointly with
    the remaining latent posteriors. The returned continuous weight is nonzero
    only on source/CLIP conflicts and is based on posterior entropy.
    """

    views = [source_probability, clip_probability, graph_probability]
    num_samples, num_classes = _validate_inputs(
        views, base_probability, anchor_mask, anchor_label
    )
    if conflict_mask.shape != (num_samples,):
        raise ValueError("conflict_mask must have shape [samples]")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if dirichlet <= 0:
        raise ValueError("dirichlet must be positive")
    if min_class_anchors <= 0:
        raise ValueError("min_class_anchors must be positive")

    device = base_probability.device
    normalized_views = [_normalize_probability(view.to(device), eps) for view in views]
    posterior = _normalize_probability(base_probability.to(device), eps)
    anchor_mask = anchor_mask.to(device=device, dtype=torch.bool)
    anchor_label = anchor_label.to(device=device, dtype=torch.long)
    conflict_mask = conflict_mask.to(device=device, dtype=torch.bool)
    anchor_one_hot = F.one_hot(
        anchor_label.clamp(0, num_classes - 1), num_classes=num_classes
    ).to(dtype=posterior.dtype)
    posterior = torch.where(anchor_mask.unsqueeze(1), anchor_one_hot, posterior)
    anchor_counts = (
        torch.bincount(anchor_label[anchor_mask], minlength=num_classes)
        if anchor_mask.any()
        else torch.zeros(num_classes, dtype=torch.long, device=device)
    )
    supported_classes = anchor_counts >= min_class_anchors

    transition_matrices = []
    for _ in range(steps):
        transition_matrices = [
            _estimate_transition(
                posterior, view, dirichlet, supported_classes, eps
            )
            for view in normalized_views
        ]
        class_prior = (posterior.sum(dim=0) + 1.0) / (num_samples + num_classes)
        log_posterior = class_prior.clamp_min(eps).log().unsqueeze(0)
        for view, transition in zip(normalized_views, transition_matrices):
            log_posterior = log_posterior + view.matmul(
                transition.clamp_min(eps).log().t()
            ) / len(normalized_views)
        posterior = F.softmax(log_posterior, dim=1)
        posterior = torch.where(anchor_mask.unsqueeze(1), anchor_one_hot, posterior)

    entropy = -(posterior * posterior.clamp_min(eps).log()).sum(dim=1)
    confidence = (1.0 - entropy / math.log(num_classes)).clamp(0.0, 1.0)
    weight = confidence * conflict_mask.float()
    active_classes = int(supported_classes.sum().item())
    conflict_count = int(conflict_mask.sum().item())
    weighted_conflicts = int((weight > eps).sum().item())
    base_label = base_probability.to(device).argmax(dim=1)
    diagnostics: dict[str, float | int] = {
        "anchors": int(anchor_mask.sum().item()),
        "active_classes": active_classes,
        "conflicts": conflict_count,
        "weighted_conflicts": weighted_conflicts,
        "mean_conflict_weight": (
            float(weight[conflict_mask].mean().item()) if conflict_count else 0.0
        ),
        "changed_top1": int((posterior.argmax(dim=1) != base_label).sum().item()),
        "source_transition_diagonal": float(
            transition_matrices[0].diagonal().mean().item()
        ),
        "clip_transition_diagonal": float(
            transition_matrices[1].diagonal().mean().item()
        ),
        "graph_transition_diagonal": float(
            transition_matrices[2].diagonal().mean().item()
        ),
        "steps": int(steps),
    }
    return posterior.detach().cpu(), weight.detach().cpu(), diagnostics


def weighted_soft_kl(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted KL used by the head-only consensus branch."""

    if logits.shape != target.shape:
        raise ValueError("logits and target must have the same shape")
    if weight.shape != (logits.size(0),):
        raise ValueError("weight must have shape [samples]")
    per_sample = F.kl_div(
        F.log_softmax(logits, dim=1), target, reduction="none"
    ).sum(dim=1)
    return (per_sample * weight).sum() / weight.sum().clamp_min(eps)
