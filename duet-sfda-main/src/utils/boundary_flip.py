"""Label-free boundary-flip proposals for pseudo-label adaptation.

The adjustment in this module is deliberately proposal-only: it does not
replace the DUET teacher.  A sample contributes supervision only after a
class-prior adjustment flips its prediction, the alternative is supported by
one of the two DUET views, and the same transition remains stable over time.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


def init_boundary_flip_state(
    num_samples: int,
    num_classes: int,
    *,
    prior_pseudocount: float = 1.0,
    initial_mean_confidence: float = 0.5,
    device: torch.device | None = None,
) -> TensorDict:
    """Create target-only temporal and class-statistics memory."""
    if num_samples <= 0 or num_classes <= 1:
        raise ValueError("Boundary-flip state requires samples and at least two classes")
    if prior_pseudocount <= 0:
        raise ValueError("prior_pseudocount must be positive")
    if not 0.0 <= initial_mean_confidence <= 1.0:
        raise ValueError("initial_mean_confidence must be in [0, 1]")

    sample_shape = (num_samples,)
    class_count = torch.full(
        (num_classes,), float(prior_pseudocount), dtype=torch.float, device=device
    )
    return {
        "initial_label": torch.full(
            sample_shape, -1, dtype=torch.long, device=device
        ),
        "initial_source_label": torch.full(
            sample_shape, -1, dtype=torch.long, device=device
        ),
        "initial_clip_label": torch.full(
            sample_shape, -1, dtype=torch.long, device=device
        ),
        "pending_label": torch.full(
            sample_shape, -1, dtype=torch.long, device=device
        ),
        "pending_count": torch.zeros(sample_shape, dtype=torch.long, device=device),
        "switch_count": torch.zeros(sample_shape, dtype=torch.long, device=device),
        "accepted_label": torch.full(
            sample_shape, -1, dtype=torch.long, device=device
        ),
        "class_count": class_count,
        "class_confidence_sum": class_count * float(initial_mean_confidence),
    }


def update_class_statistics(
    state: TensorDict,
    labels: torch.Tensor,
    confidence: torch.Tensor,
    anchor_mask: torch.Tensor,
    num_classes: int,
) -> None:
    """Accumulate label-free class frequency and confidence from safe anchors."""
    if labels.ndim != 1 or confidence.shape != labels.shape:
        raise ValueError("labels and confidence must be aligned vectors")
    if anchor_mask.shape != labels.shape or anchor_mask.dtype != torch.bool:
        raise ValueError("anchor_mask must be a boolean vector aligned with labels")
    if not anchor_mask.any():
        return

    anchor_labels = labels[anchor_mask]
    anchor_confidence = confidence[anchor_mask].float().clamp(0.0, 1.0)
    counts = torch.bincount(anchor_labels, minlength=num_classes).float()
    confidence_sum = torch.bincount(
        anchor_labels, weights=anchor_confidence, minlength=num_classes
    ).float()
    state["class_count"].add_(counts)
    state["class_confidence_sum"].add_(confidence_sum)


def dynamic_logit_adjustment(
    base_probability: torch.Tensor,
    class_count: torch.Tensor,
    class_mean_confidence: torch.Tensor,
    *,
    alpha: float,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return adjusted probabilities and the per-row class penalty.

    The probability-space DUET host does not expose a single shared raw logit
    scale.  We therefore apply the DLA idea as an additive penalty to log
    probabilities.  The penalty is larger for overrepresented anchor classes
    and is modulated by how the current confidence differs from that class's
    historical confidence.
    """
    if base_probability.ndim != 2:
        raise ValueError("base_probability must be [samples, classes]")
    if class_count.shape != (base_probability.size(1),):
        raise ValueError("class_count has the wrong number of classes")
    if class_mean_confidence.shape != class_count.shape:
        raise ValueError("class_mean_confidence must match class_count")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    probability = base_probability.float().clamp_min(epsilon)
    probability = probability / probability.sum(dim=1, keepdim=True)
    base_label = probability.argmax(dim=1)
    row_index = torch.arange(probability.size(0), device=probability.device)
    base_confidence = probability[row_index, base_label]
    confidence_deviation = (
        base_confidence - class_mean_confidence[base_label]
    ).clamp(-0.5, 0.5)

    prior = class_count.float() / class_count.sum().clamp_min(epsilon)
    relative_prior = prior * float(class_count.numel())
    modulation = (1.0 - confidence_deviation).unsqueeze(1)
    penalty = float(alpha) * modulation * relative_prior.unsqueeze(0)
    adjusted_probability = F.softmax(probability.log() - penalty, dim=1)
    return adjusted_probability, penalty


def apply_pair_budget(
    mask: torch.Tensor,
    early_label: torch.Tensor,
    late_label: torch.Tensor,
    score: torch.Tensor,
    *,
    num_classes: int,
    max_per_pair: int,
) -> torch.Tensor:
    """Keep the strongest samples independently for each ordered class pair."""
    if max_per_pair <= 0 or not mask.any():
        return mask.clone()
    selected = torch.zeros_like(mask)
    pair_id = early_label * int(num_classes) + late_label
    for current_pair in torch.unique(pair_id[mask]):
        pair_indices = torch.nonzero(
            mask & (pair_id == current_pair), as_tuple=False
        ).squeeze(1)
        keep = min(int(max_per_pair), int(pair_indices.numel()))
        top_positions = torch.topk(score[pair_indices], k=keep).indices
        selected[pair_indices[top_positions]] = True
    return selected


def update_boundary_flip_memory(
    state: TensorDict,
    eligible: torch.Tensor,
    late_label: torch.Tensor,
    *,
    stable_cycles: int,
    max_switches: int,
) -> torch.Tensor:
    """Track one stable transition and reject oscillating flip histories."""
    if stable_cycles <= 0:
        raise ValueError("stable_cycles must be positive")
    if max_switches < 0:
        raise ValueError("max_switches must be non-negative")
    if eligible.dtype != torch.bool or eligible.shape != late_label.shape:
        raise ValueError("eligible and late_label must be aligned vectors")

    pending_label = state["pending_label"]
    pending_count = state["pending_count"]
    had_pending = pending_label >= 0
    same_pending = pending_label == late_label
    changed_candidate = eligible & had_pending & ~same_pending
    interrupted_candidate = ~eligible & had_pending
    state["switch_count"].add_(
        (changed_candidate | interrupted_candidate).long()
    )

    state["pending_count"] = torch.where(
        eligible & same_pending,
        pending_count + 1,
        torch.where(eligible, torch.ones_like(pending_count), torch.zeros_like(pending_count)),
    )
    state["pending_label"] = torch.where(
        eligible, late_label, torch.full_like(pending_label, -1)
    )
    stable = (
        eligible
        & (state["pending_count"] >= int(stable_cycles))
        & (state["switch_count"] <= int(max_switches))
    )
    state["accepted_label"] = torch.where(
        stable, late_label, torch.full_like(late_label, -1)
    )
    return stable


def update_boundary_flip_state(
    model_probability: torch.Tensor,
    clip_probability: torch.Tensor,
    source_label: torch.Tensor,
    clip_label: torch.Tensor,
    anchor_mask: torch.Tensor,
    text_features: torch.Tensor,
    state: TensorDict | None,
    *,
    curr_cycle: int,
    start_cycle: int,
    alpha: float,
    min_adjusted_confidence: float,
    min_margin: float,
    semantic_threshold: float,
    stable_cycles: int,
    max_switches: int,
    max_per_pair: int,
    min_weight: float,
    epsilon: float = 1e-6,
) -> Tuple[TensorDict, TensorDict]:
    """Update class/temporal memory and return current active flip supervision."""
    if model_probability.shape != clip_probability.shape:
        raise ValueError("model and CLIP probabilities must have matching shapes")
    if model_probability.ndim != 2:
        raise ValueError("probabilities must be [samples, classes]")
    num_samples, num_classes = model_probability.shape
    for label in (source_label, clip_label):
        if label.shape != (num_samples,):
            raise ValueError("view labels must align with probability rows")
    if anchor_mask.shape != (num_samples,) or anchor_mask.dtype != torch.bool:
        raise ValueError("anchor_mask must be a boolean sample vector")
    if text_features.ndim != 2 or text_features.size(0) != num_classes:
        raise ValueError("text_features must contain one vector per class")
    if not 0.0 <= min_adjusted_confidence <= 1.0:
        raise ValueError("min_adjusted_confidence must be in [0, 1]")
    if not -1.0 <= semantic_threshold < 1.0:
        raise ValueError("semantic_threshold must be in [-1, 1)")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("min_weight must be in [0, 1]")

    base_probability = (model_probability.float() + clip_probability.float()) / 2
    base_probability = base_probability / base_probability.sum(
        dim=1, keepdim=True
    ).clamp_min(epsilon)
    base_label = base_probability.argmax(dim=1)
    row_index = torch.arange(num_samples, device=base_probability.device)

    if state is None:
        state = init_boundary_flip_state(
            num_samples, num_classes, device=base_probability.device
        )
    if state["initial_label"].numel() != num_samples:
        raise ValueError("boundary-flip state sample count changed")

    uninitialized = state["initial_label"] < 0
    state["initial_label"] = torch.where(
        uninitialized, base_label, state["initial_label"]
    )
    state["initial_source_label"] = torch.where(
        uninitialized, source_label, state["initial_source_label"]
    )
    state["initial_clip_label"] = torch.where(
        uninitialized, clip_label, state["initial_clip_label"]
    )

    view_agreement = source_label == clip_label
    agreement_confidence = torch.sqrt(
        (
            model_probability[row_index, source_label]
            * clip_probability[row_index, source_label]
        ).clamp_min(0.0)
    )
    update_class_statistics(
        state,
        source_label,
        agreement_confidence,
        anchor_mask & view_agreement,
        num_classes,
    )
    class_mean_confidence = state["class_confidence_sum"] / state[
        "class_count"
    ].clamp_min(epsilon)
    adjusted_probability, penalty = dynamic_logit_adjustment(
        base_probability,
        state["class_count"],
        class_mean_confidence,
        alpha=alpha,
        epsilon=epsilon,
    )
    adjusted_label = adjusted_probability.argmax(dim=1)
    initial_label = state["initial_label"]

    normalized_text = F.normalize(text_features.float(), dim=1)
    semantic_similarity = (
        normalized_text[initial_label] * normalized_text[adjusted_label]
    ).sum(dim=1)
    adjusted_confidence = adjusted_probability[row_index, adjusted_label]
    flip_margin = (
        adjusted_probability[row_index, adjusted_label]
        - adjusted_probability[row_index, initial_label]
    )
    supported_by_view = (adjusted_label == source_label) | (
        adjusted_label == clip_label
    )
    eligible = (
        (curr_cycle >= int(start_cycle))
        & (adjusted_label != base_label)
        & (adjusted_label != initial_label)
        & supported_by_view
        & (adjusted_confidence >= float(min_adjusted_confidence))
        & (flip_margin >= float(min_margin))
        & (semantic_similarity >= float(semantic_threshold))
    )
    stable = update_boundary_flip_memory(
        state,
        eligible,
        adjusted_label,
        stable_cycles=stable_cycles,
        max_switches=max_switches,
    )
    active = apply_pair_budget(
        stable,
        initial_label,
        adjusted_label,
        flip_margin,
        num_classes=num_classes,
        max_per_pair=max_per_pair,
    )

    semantic_weight = (
        (semantic_similarity - float(semantic_threshold))
        / max(1.0 - float(semantic_threshold), epsilon)
    ).clamp(0.0, 1.0)
    flip_weight = (adjusted_confidence * semantic_weight).clamp(0.0, 1.0)
    flip_weight = torch.where(
        active,
        flip_weight.clamp_min(float(min_weight)),
        torch.zeros_like(flip_weight),
    )
    state["accepted_label"] = torch.where(
        active, adjusted_label, torch.full_like(adjusted_label, -1)
    )

    result = {
        "base_label": base_label,
        "initial_label": initial_label.clone(),
        "adjusted_label": adjusted_label,
        "adjusted_probability": adjusted_probability,
        "penalty": penalty,
        "semantic_similarity": semantic_similarity,
        "flip_margin": flip_margin,
        "candidate_mask": eligible,
        "stable_mask": stable,
        "active_mask": active,
        "weight": flip_weight,
        "switch_count": state["switch_count"].clone(),
        "class_prior": (
            state["class_count"] / state["class_count"].sum().clamp_min(epsilon)
        ),
        "class_mean_confidence": class_mean_confidence,
    }
    return state, result


def boundary_flip_loss(
    logits: torch.Tensor,
    early_label: torch.Tensor,
    late_label: torch.Tensor,
    weight: torch.Tensor,
    *,
    negative_weight: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Positive late-label CE plus complementary early-label suppression."""
    if logits.ndim != 2:
        raise ValueError("logits must be [samples, classes]")
    sample_count = logits.size(0)
    if any(
        value.shape != (sample_count,)
        for value in (early_label, late_label, weight)
    ):
        raise ValueError("labels and weights must align with logits")
    if negative_weight < 0:
        raise ValueError("negative_weight must be non-negative")
    if sample_count == 0:
        return logits.sum() * 0.0

    probability = F.softmax(logits, dim=1)
    row_index = torch.arange(sample_count, device=logits.device)
    positive = -torch.log(
        probability[row_index, late_label].clamp_min(epsilon)
    )
    negative = -torch.log(
        (1.0 - probability[row_index, early_label]).clamp_min(epsilon)
    )
    safe_weight = weight.float().clamp_min(0.0)
    return (
        (positive + float(negative_weight) * negative) * safe_weight
    ).sum() / safe_weight.sum().clamp_min(epsilon)
