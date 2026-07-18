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
    anchor_mask: torch.Tensor | None = None,
    anchor_label: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return task, CLIP, and product-of-experts posteriors plus anchor mask."""
    source_prob = source_prob.float().cpu()
    clip_prob = clip_prob.float().cpu()
    source_label = source_label.long().cpu()
    clip_label = clip_label.long().cpu()
    if anchor_mask is None:
        anchors, _ = select_class_balanced_anchors(
            source_prob,
            clip_prob,
            source_label,
            clip_label,
            ratio=anchor_ratio,
            min_per_class=anchor_min_per_class,
        )
        seed_label = source_label
    else:
        anchors = anchor_mask.bool().cpu()
        if anchor_label is None:
            raise ValueError("anchor_label is required with a fixed anchor_mask")
        seed_label = anchor_label.long().cpu()
        if anchors.numel() != source_label.numel() or seed_label.numel() != source_label.numel():
            raise ValueError("fixed anchors must match the number of target samples")
    if not anchors.any():
        raise RuntimeError("ACCD found no source/CLIP agreement anchors")

    seed = torch.zeros_like(source_prob)
    seed[anchors, seed_label[anchors]] = 1.0
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


@torch.no_grad()
def transport_candidate_mass(
    teacher_prob: torch.Tensor,
    graph_prob: torch.Tensor,
    source_label: torch.Tensor,
    clip_label: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Redistribute candidate mass by graph support without changing other classes."""
    if teacher_prob.shape != graph_prob.shape or teacher_prob.ndim != 2:
        raise ValueError("teacher_prob and graph_prob must be matching 2D tensors")
    num_samples = teacher_prob.size(0)
    if any(t.numel() != num_samples for t in (source_label, clip_label, mask)):
        raise ValueError("labels and mask must match the teacher sample dimension")

    graph_prob = graph_prob.to(device=teacher_prob.device, dtype=teacher_prob.dtype)
    source_label = source_label.long().to(teacher_prob.device)
    clip_label = clip_label.long().to(teacher_prob.device)
    active = mask.bool().to(teacher_prob.device) & (source_label != clip_label)
    corrected = teacher_prob.clone()
    shifted_mass = torch.zeros(num_samples, dtype=teacher_prob.dtype, device=teacher_prob.device)
    if not active.any():
        return corrected, shifted_mass

    rows = torch.nonzero(active, as_tuple=False).squeeze(1)
    source = source_label[rows]
    clip = clip_label[rows]
    candidate_mass = teacher_prob[rows, source] + teacher_prob[rows, clip]
    graph_pair = graph_prob[rows, source] + graph_prob[rows, clip]
    source_ratio = graph_prob[rows, source] / graph_pair.clamp_min(
        torch.finfo(graph_prob.dtype).eps
    )
    new_source = candidate_mass * source_ratio
    new_clip = candidate_mass - new_source
    shifted_mass[rows] = (new_source - teacher_prob[rows, source]).abs()
    corrected[rows, source] = new_source
    corrected[rows, clip] = new_clip
    return corrected, shifted_mass


@torch.no_grad()
def update_temporal_resolution(
    pending_label: torch.Tensor,
    pending_count: torch.Tensor,
    resolved_label: torch.Tensor,
    eligible: torch.Tensor,
    proposed_label: torch.Tensor,
    stable_cycles: int,
    memory: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update persistent or reversible conflict labels from temporal evidence."""
    if memory not in {"persistent", "reversible"}:
        raise ValueError(f"Unknown temporal resolution memory: {memory}")
    if stable_cycles <= 0:
        raise ValueError("stable_cycles must be positive")

    previous_resolved = resolved_label.clone()
    already_resolved = previous_resolved >= 0
    trackable = eligible & (~already_resolved) if memory == "persistent" else eligible
    same_label = pending_label == proposed_label
    pending_count = torch.where(
        trackable & same_label,
        pending_count + 1,
        torch.where(trackable, torch.ones_like(pending_count), torch.zeros_like(pending_count)),
    )
    pending_label = torch.where(
        trackable,
        proposed_label,
        torch.full_like(pending_label, -1),
    )
    stable = trackable & (pending_count >= stable_cycles)

    if memory == "persistent":
        resolved_label = torch.where(stable, proposed_label, previous_resolved)
    else:
        resolved_label = torch.where(stable, proposed_label, torch.full_like(previous_resolved, -1))

    resolved_mask = resolved_label >= 0
    newly_resolved = resolved_mask & (
        (previous_resolved < 0) | (previous_resolved != resolved_label)
    )
    demoted = (previous_resolved >= 0) & (~resolved_mask)
    return pending_label, pending_count, resolved_label, newly_resolved, resolved_mask, demoted
