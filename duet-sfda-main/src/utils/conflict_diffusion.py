"""Agreement-anchored candidate-compatibility diffusion for conflicts."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def select_class_balanced_anchors(
    source_prob: torch.Tensor,
    clip_prob: torch.Tensor,
    source_label: torch.Tensor,
    clip_label: torch.Tensor,
    ratio: float,
    min_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the most confident source/CLIP agreements within every class."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"anchor ratio must be in (0, 1], got {ratio}")

    sample_idx = torch.arange(source_label.numel(), device=source_label.device)
    agreement = source_label == clip_label
    score = torch.sqrt(
        source_prob[sample_idx, source_label].clamp_min(0.0)
        * clip_prob[sample_idx, clip_label].clamp_min(0.0)
    )
    anchors = torch.zeros_like(agreement)

    for class_idx in range(source_prob.size(1)):
        candidates = torch.nonzero(
            agreement & (source_label == class_idx), as_tuple=False
        ).squeeze(1)
        if candidates.numel() == 0:
            continue
        keep = max(min_per_class, int(math.ceil(candidates.numel() * ratio)))
        keep = min(keep, candidates.numel())
        selected = candidates[torch.topk(score[candidates], k=keep).indices]
        anchors[selected] = True

    return anchors, score


@torch.no_grad()
def build_knn_graph(
    features: torch.Tensor,
    k: int,
    temperature: float,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a directed cosine kNN graph without materializing an NxN matrix."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    if features.size(0) < 2:
        raise ValueError("at least two samples are required for graph diffusion")
    if temperature <= 0:
        raise ValueError("graph temperature must be positive")

    features = F.normalize(features.float(), dim=1).to(device)
    k = min(int(k), features.size(0) - 1)
    if k <= 0:
        raise ValueError("graph k must be positive")

    all_indices = []
    all_weights = []
    for start in range(0, features.size(0), chunk_size):
        end = min(start + chunk_size, features.size(0))
        similarity = features[start:end] @ features.t()
        local_rows = torch.arange(end - start, device=device)
        global_rows = torch.arange(start, end, device=device)
        similarity[local_rows, global_rows] = -float("inf")
        values, indices = torch.topk(similarity, k=k, dim=1)
        weights = torch.softmax(values / temperature, dim=1)
        all_indices.append(indices)
        all_weights.append(weights)

    return torch.cat(all_indices, dim=0), torch.cat(all_weights, dim=0)


@torch.no_grad()
def propagate_anchor_labels(
    features: torch.Tensor,
    seed: torch.Tensor,
    k: int,
    temperature: float,
    alpha: float,
    steps: int,
    chunk_size: int = 512,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Diffuse class anchors over a target feature manifold with random restart."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    if steps <= 0:
        raise ValueError("diffusion steps must be positive")
    if seed.ndim != 2 or seed.size(0) != features.size(0):
        raise ValueError("seed and features must have the same sample dimension")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices, weights = build_knn_graph(
        features, k=k, temperature=temperature, chunk_size=chunk_size, device=device
    )
    seed = seed.float().to(device)
    posterior = seed.clone()

    for _ in range(steps):
        neighbor_posterior = posterior[indices]
        propagated = (neighbor_posterior * weights.unsqueeze(2)).sum(dim=1)
        posterior = alpha * propagated + (1.0 - alpha) * seed

    normalizer = posterior.sum(dim=1, keepdim=True)
    uniform = torch.full_like(posterior, 1.0 / posterior.size(1))
    posterior = torch.where(
        normalizer > 0,
        posterior / normalizer.clamp_min(torch.finfo(posterior.dtype).eps),
        uniform,
    )
    return posterior.cpu()


@torch.no_grad()
def dual_space_diffusion(
    task_features: torch.Tensor,
    clip_features: torch.Tensor,
    source_prob: torch.Tensor,
    clip_prob: torch.Tensor,
    source_label: torch.Tensor,
    clip_label: torch.Tensor,
    *,
    anchor_ratio: float,
    anchor_min_per_class: int,
    k: int,
    temperature: float,
    alpha: float,
    steps: int,
    chunk_size: int = 512,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return task, CLIP, and product-of-experts posteriors plus anchor mask."""
    source_prob = source_prob.float().cpu()
    clip_prob = clip_prob.float().cpu()
    source_label = source_label.long().cpu()
    clip_label = clip_label.long().cpu()
    anchors, _ = select_class_balanced_anchors(
        source_prob,
        clip_prob,
        source_label,
        clip_label,
        ratio=anchor_ratio,
        min_per_class=anchor_min_per_class,
    )
    if not anchors.any():
        raise RuntimeError("ACCD found no source/CLIP agreement anchors")

    seed = torch.zeros_like(source_prob)
    seed[anchors, source_label[anchors]] = 1.0
    task_posterior = propagate_anchor_labels(
        task_features,
        seed,
        k=k,
        temperature=temperature,
        alpha=alpha,
        steps=steps,
        chunk_size=chunk_size,
        device=device,
    )
    clip_posterior = propagate_anchor_labels(
        clip_features,
        seed,
        k=k,
        temperature=temperature,
        alpha=alpha,
        steps=steps,
        chunk_size=chunk_size,
        device=device,
    )

    eps = torch.finfo(task_posterior.dtype).eps
    fused = torch.sqrt(task_posterior.clamp_min(eps) * clip_posterior.clamp_min(eps))
    fused = fused / fused.sum(dim=1, keepdim=True).clamp_min(eps)
    return task_posterior, clip_posterior, fused, anchors


@torch.no_grad()
def conflict_diffusion_evidence(
    task_posterior: torch.Tensor,
    clip_posterior: torch.Tensor,
    fused_posterior: torch.Tensor,
    source_label: torch.Tensor,
    clip_label: torch.Tensor,
    candidate_mass_threshold: float,
    candidate_margin_threshold: float,
) -> dict[str, torch.Tensor]:
    """Identify conflicts supported by the same candidate in both manifolds."""
    source_label = source_label.long().cpu()
    clip_label = clip_label.long().cpu()
    sample_idx = torch.arange(source_label.numel())
    conflict = source_label != clip_label

    task_top = task_posterior.argmax(dim=1)
    clip_top = clip_posterior.argmax(dim=1)
    graph_label = fused_posterior.argmax(dim=1)
    cross_space_agreement = (task_top == clip_top) & (task_top == graph_label)
    in_candidate = (graph_label == source_label) | (graph_label == clip_label)

    source_support = fused_posterior[sample_idx, source_label]
    clip_support = fused_posterior[sample_idx, clip_label]
    candidate_mass = source_support + clip_support
    candidate_margin = torch.abs(source_support - clip_support)
    eligible = (
        conflict
        & cross_space_agreement
        & in_candidate
        & (candidate_mass >= candidate_mass_threshold)
        & (candidate_margin >= candidate_margin_threshold)
    )
    outside_candidate = conflict & (candidate_mass < candidate_mass_threshold)

    return {
        "conflict": conflict,
        "eligible": eligible,
        "outside_candidate": outside_candidate,
        "graph_label": graph_label,
        "candidate_mass": candidate_mass,
        "candidate_margin": candidate_margin,
        "cross_space_agreement": cross_space_agreement,
    }
