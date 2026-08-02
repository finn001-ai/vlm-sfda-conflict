"""Label-free support conditioning for DUET's CLIP KL target."""

from __future__ import annotations

import torch


def condition_clip_on_task_clip_top2_union(
    task_probability: torch.Tensor,
    clip_probability: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Condition CLIP mass on the union of task and CLIP top-2 classes.

    The task model contributes only the support set.  Within that support the
    relative CLIP probabilities are preserved, so this does not select a
    task/CLIP winner or introduce a fitted mixing weight.
    """
    if task_probability.ndim != 2 or clip_probability.ndim != 2:
        raise ValueError("task and CLIP probabilities must be 2-D")
    if task_probability.shape != clip_probability.shape:
        raise ValueError("task and CLIP probabilities must have matching shapes")
    if task_probability.shape[1] < 2:
        raise ValueError("top-2 support requires at least two classes")
    if not task_probability.is_floating_point() or not clip_probability.is_floating_point():
        raise ValueError("task and CLIP probabilities must be floating point")
    if not torch.isfinite(task_probability).all() or not torch.isfinite(
        clip_probability
    ).all():
        raise ValueError("task and CLIP probabilities must be finite")
    if torch.any(task_probability < 0) or torch.any(clip_probability < 0):
        raise ValueError("task and CLIP probabilities must be non-negative")

    task_sum = task_probability.sum(dim=1)
    clip_sum = clip_probability.sum(dim=1)
    if not torch.allclose(task_sum, torch.ones_like(task_sum), atol=1e-5, rtol=1e-5):
        raise ValueError("task probability rows must sum to one")
    if not torch.allclose(clip_sum, torch.ones_like(clip_sum), atol=1e-5, rtol=1e-5):
        raise ValueError("CLIP probability rows must sum to one")

    support = torch.zeros_like(task_probability, dtype=torch.bool)
    support.scatter_(1, torch.topk(task_probability, k=2, dim=1).indices, True)
    support.scatter_(1, torch.topk(clip_probability, k=2, dim=1).indices, True)

    supported_clip = clip_probability * support.to(clip_probability.dtype)
    retained_mass = supported_clip.sum(dim=1, keepdim=True)
    if torch.any(retained_mass <= 0):
        raise RuntimeError("top-2 union retained no CLIP probability mass")
    probability = supported_clip / retained_mass

    if not torch.equal(probability.argmax(dim=1), clip_probability.argmax(dim=1)):
        raise RuntimeError("support conditioning changed the CLIP top-1 class")
    return {
        "probability": probability,
        "support": support,
        "retained_clip_mass": retained_mass.squeeze(1),
        "support_size": support.sum(dim=1),
    }
