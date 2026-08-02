"""Parameter-free compatibility control for DUET conflict PCGrad."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.pcgrad_parameter_audit import symmetric_pcgrad_output_correction


def compatibility_fraction_from_norms(
    baseline_norm: np.ndarray,
    candidate_norm: np.ndarray,
    correction_norm: np.ndarray,
    *,
    epsilon: float = 1e-15,
) -> dict[str, np.ndarray]:
    """Recover ``<baseline, correction>`` and a clipped projection fraction.

    Since ``candidate = baseline + correction``, the dot product is exactly
    recoverable from the three locked norms.  The fraction contains no fitted
    threshold: ``clip(dot / ||correction||^2, 0, 1)``.
    """
    baseline = np.asarray(baseline_norm, dtype=np.float64)
    candidate = np.asarray(candidate_norm, dtype=np.float64)
    correction = np.asarray(correction_norm, dtype=np.float64)
    if not (baseline.shape == candidate.shape == correction.shape):
        raise ValueError("gradient norm arrays must have identical shapes")
    if baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("gradient norm arrays must be non-empty vectors")
    if not all(np.isfinite(value).all() for value in (baseline, candidate, correction)):
        raise ValueError("gradient norms must be finite")
    if bool((baseline < 0.0).any() or (candidate < 0.0).any() or (correction < 0.0).any()):
        raise ValueError("gradient norms must be non-negative")
    correction_norm_sq = correction**2
    dot = 0.5 * (candidate**2 - baseline**2 - correction_norm_sq)
    fraction = np.zeros_like(dot)
    nonzero = correction_norm_sq > epsilon
    fraction[nonzero] = np.clip(
        dot[nonzero] / correction_norm_sq[nonzero], 0.0, 1.0
    )
    return {
        "baseline_correction_dot": dot,
        "correction_norm_sq": correction_norm_sq,
        "fraction": fraction,
    }


def reconstruct_fractional_metrics(
    *,
    fraction: np.ndarray,
    baseline_norm: np.ndarray,
    candidate_norm: np.ndarray,
    correction_norm: np.ndarray,
    baseline_unit_projection: np.ndarray,
    candidate_unit_projection: np.ndarray,
    baseline_first_order: np.ndarray,
    candidate_first_order: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exactly reconstruct locked oracle metrics for a fractional correction."""
    alpha = np.asarray(fraction, dtype=np.float64)
    values = [
        np.asarray(value, dtype=np.float64)
        for value in (
            baseline_norm,
            candidate_norm,
            correction_norm,
            baseline_unit_projection,
            candidate_unit_projection,
            baseline_first_order,
            candidate_first_order,
        )
    ]
    if any(value.shape != alpha.shape for value in values):
        raise ValueError("fractional metric arrays must have identical shapes")
    if alpha.ndim != 1 or alpha.size == 0 or not np.isfinite(alpha).all():
        raise ValueError("fraction must be a finite non-empty vector")
    if bool(((alpha < 0.0) | (alpha > 1.0)).any()):
        raise ValueError("fraction must lie in [0, 1]")
    (
        baseline_norm_value,
        candidate_norm_value,
        correction_norm_value,
        baseline_projection,
        candidate_projection,
        baseline_first,
        candidate_first,
    ) = values
    recovered = compatibility_fraction_from_norms(
        baseline_norm_value,
        candidate_norm_value,
        correction_norm_value,
    )
    dot = recovered["baseline_correction_dot"]
    fractional_norm = np.sqrt(
        np.maximum(
            baseline_norm_value**2
            + 2.0 * alpha * dot
            + alpha**2 * correction_norm_value**2,
            0.0,
        )
    )
    fractional_projection = baseline_projection + alpha * (
        candidate_projection - baseline_projection
    )
    fractional_first = baseline_first + alpha * (
        candidate_first - baseline_first
    )
    fractional_cosine = np.divide(
        fractional_projection,
        fractional_norm,
        out=np.zeros_like(fractional_projection),
        where=fractional_norm > 0.0,
    )
    baseline_cosine = np.divide(
        baseline_projection,
        baseline_norm_value,
        out=np.zeros_like(baseline_projection),
        where=baseline_norm_value > 0.0,
    )
    # Preserve exact equality for rejected corrections instead of allowing
    # floating reconstruction noise to turn zeros into positive/negative rows.
    fractional_norm = np.where(alpha == 0.0, baseline_norm_value, fractional_norm)
    fractional_projection = np.where(
        alpha == 0.0, baseline_projection, fractional_projection
    )
    fractional_first = np.where(alpha == 0.0, baseline_first, fractional_first)
    fractional_cosine = np.where(alpha == 0.0, baseline_cosine, fractional_cosine)
    return {
        "norm": fractional_norm,
        "oracle_unit_projection": fractional_projection,
        "first_order": fractional_first,
        "cosine": fractional_cosine,
    }


def _gradient_dot(
    first: Iterable[torch.Tensor], second: Iterable[torch.Tensor]
) -> torch.Tensor:
    pairs = tuple(zip(first, second))
    if not pairs:
        raise ValueError("gradient tuples must not be empty")
    return sum((left * right).sum() for left, right in pairs)


def build_pcgrad_parameter_correction(
    *,
    weak_logits: torch.Tensor,
    strong_logits: torch.Tensor,
    weak_probability: torch.Tensor,
    strong_probability: torch.Tensor,
    clip_target: torch.Tensor,
    unresolved_mask: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    consistency_weight: float,
    clip_weight: float,
) -> dict[str, Any] | None:
    """Build the exact unresolved-conflict PCGrad parameter correction."""
    mask = unresolved_mask.bool()
    if mask.shape != (weak_logits.shape[0],):
        raise ValueError("unresolved_mask must contain one value per batch row")
    if not bool(mask.any()):
        return None
    consistency_sum = float(consistency_weight) * F.kl_div(
        strong_probability[mask].log(),
        weak_probability[mask],
        reduction="sum",
    )
    consistency_weak_grad, consistency_strong_grad = torch.autograd.grad(
        consistency_sum,
        (weak_logits, strong_logits),
        retain_graph=True,
    )
    clip_sum = float(clip_weight) * F.kl_div(
        weak_probability[mask].log(), clip_target[mask], reduction="sum"
    )
    clip_weak_grad = torch.autograd.grad(
        clip_sum, weak_logits, retain_graph=True
    )[0]
    consistency_descent = torch.cat(
        (-consistency_weak_grad, -consistency_strong_grad), dim=1
    ).detach()
    clip_descent = torch.cat(
        (-clip_weak_grad, torch.zeros_like(clip_weak_grad)), dim=1
    ).detach()
    surgery = symmetric_pcgrad_output_correction(
        consistency_descent, clip_descent, mask
    )
    correction = surgery["correction"]
    class_count = weak_logits.shape[1]
    batch_size = weak_logits.shape[0]
    parameter_correction = torch.autograd.grad(
        (weak_logits, strong_logits),
        parameters,
        grad_outputs=(
            -correction[:, :class_count] / batch_size,
            -correction[:, class_count:] / batch_size,
        ),
        retain_graph=True,
    )
    return {
        "parameter_correction": parameter_correction,
        "unresolved": int(mask.sum().detach().cpu()),
        "output_pcgrad_active": int((surgery["active"] & mask).sum().detach().cpu()),
        "mean_component_cosine": float(
            surgery["component_cosine"][mask].mean().detach().cpu()
        ),
    }


def merge_compatible_parameter_correction_(
    parameters: tuple[torch.nn.Parameter, ...],
    correction: tuple[torch.Tensor, ...],
    *,
    epsilon: float = 1e-15,
) -> dict[str, float]:
    """Merge a correction into populated baseline grads using the fixed rule."""
    if len(parameters) == 0 or len(parameters) != len(correction):
        raise ValueError("parameters and correction must be same-sized tuples")
    baseline = tuple(parameter.grad for parameter in parameters)
    if any(value is None for value in baseline):
        raise RuntimeError("baseline gradients must be populated before merging")
    baseline_tensors = tuple(value for value in baseline if value is not None)
    dot = _gradient_dot(baseline_tensors, correction)
    correction_norm_sq = _gradient_dot(correction, correction)
    if not bool(torch.isfinite(dot)) or not bool(torch.isfinite(correction_norm_sq)):
        raise RuntimeError("PCGrad compatibility metrics must be finite")
    if float(correction_norm_sq.detach().cpu()) <= epsilon:
        fraction = torch.zeros_like(dot)
    else:
        fraction = torch.clamp(dot / correction_norm_sq, min=0.0, max=1.0)
    fraction_value = float(fraction.detach().cpu())
    with torch.no_grad():
        for parameter, value in zip(parameters, correction):
            parameter.grad.add_(value, alpha=fraction_value)
    return {
        "baseline_correction_dot": float(dot.detach().cpu()),
        "correction_norm_sq": float(correction_norm_sq.detach().cpu()),
        "fraction": fraction_value,
    }

