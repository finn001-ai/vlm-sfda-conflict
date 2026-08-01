"""Probability-fusion utilities shared by DUET diagnostics and training."""

from __future__ import annotations

import torch


def _validate_probabilities(task_prob: torch.Tensor, clip_prob: torch.Tensor) -> None:
    if task_prob.shape != clip_prob.shape:
        raise ValueError(
            "task_prob and clip_prob must have the same shape, got "
            f"{tuple(task_prob.shape)} and {tuple(clip_prob.shape)}"
        )
    if task_prob.ndim < 1:
        raise ValueError("probability tensors must have at least one dimension")
    if not task_prob.is_floating_point() or not clip_prob.is_floating_point():
        raise TypeError("probability tensors must use a floating-point dtype")
    if torch.any(task_prob < 0) or torch.any(clip_prob < 0):
        raise ValueError("probability tensors must be non-negative")


def arithmetic_probability_fusion(
    task_prob: torch.Tensor,
    clip_prob: torch.Tensor,
) -> torch.Tensor:
    """Return the released DUET arithmetic-mean probability fusion."""
    _validate_probabilities(task_prob, clip_prob)
    return (task_prob + clip_prob) / 2.0


def rms_probability_fusion(
    task_prob: torch.Tensor,
    clip_prob: torch.Tensor,
    *,
    epsilon: float | None = None,
) -> torch.Tensor:
    """Fuse two probability vectors with a normalized element-wise RMS.

    The raw root-mean-square vector does not generally sum to one. DUET passes
    the fused vector to probability-based objectives, so the result is
    normalized across classes before it is returned.
    """
    _validate_probabilities(task_prob, clip_prob)
    if epsilon is None:
        epsilon = torch.finfo(task_prob.dtype).eps
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    fused = torch.sqrt((task_prob.square() + clip_prob.square()) / 2.0)
    normalizer = fused.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    return fused / normalizer


def fuse_probabilities(
    task_prob: torch.Tensor,
    clip_prob: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Dispatch an explicitly named two-view probability fusion."""
    normalized_mode = mode.strip().lower()
    if normalized_mode == "arithmetic":
        return arithmetic_probability_fusion(task_prob, clip_prob)
    if normalized_mode == "rms":
        return rms_probability_fusion(task_prob, clip_prob)
    raise ValueError(
        f"Unsupported DUET probability fusion {mode!r}; expected 'arithmetic' or 'rms'"
    )
