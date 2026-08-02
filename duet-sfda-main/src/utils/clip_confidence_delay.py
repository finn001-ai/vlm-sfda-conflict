"""Class-balanced first-cycle admission delay for DUET agreements."""

from __future__ import annotations

import math

import numpy as np
import torch


LOCKED_DELAY_FRACTION = 0.10


def class_balanced_clip_confidence_delay(
    matching: torch.Tensor,
    common_prediction: torch.Tensor,
    clip_probability: torch.Tensor,
    *,
    fraction: float = LOCKED_DELAY_FRACTION,
) -> dict[str, torch.Tensor | dict[int, int]]:
    """Delay the lowest-CLIP-confidence agreements within each pseudo class.

    The operation is label-free and deterministic.  It changes only the hard
    pseudo-label admission mask; probabilities and labels are returned untouched.
    """
    agreement = torch.as_tensor(matching, dtype=torch.bool)
    prediction = torch.as_tensor(common_prediction, dtype=torch.long)
    probability = torch.as_tensor(clip_probability)
    if agreement.ndim != 1 or prediction.shape != agreement.shape:
        raise ValueError("matching and common_prediction must be same-shaped vectors")
    if (
        probability.ndim != 2
        or probability.shape[0] != agreement.numel()
        or probability.shape[1] < 2
    ):
        raise ValueError("clip_probability must have shape [sample, class]")
    if not torch.isfinite(probability).all() or torch.any(probability < 0.0):
        raise ValueError("clip_probability must be finite and non-negative")
    if not torch.allclose(
        probability.sum(dim=1),
        torch.ones(probability.shape[0], dtype=probability.dtype, device=probability.device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("clip_probability rows must sum to one")
    if torch.any(prediction < 0) or torch.any(prediction >= probability.shape[1]):
        raise ValueError("common_prediction is outside the class range")
    if not math.isclose(
        float(fraction), LOCKED_DELAY_FRACTION, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("the predeclared delay fraction is locked to 0.10")

    delayed = torch.zeros_like(agreement)
    counts_by_class: dict[int, int] = {}
    probability_cpu = probability.detach().cpu().numpy()
    agreement_cpu = agreement.detach().cpu().numpy()
    prediction_cpu = prediction.detach().cpu().numpy()
    for class_index in range(probability.shape[1]):
        indices = np.flatnonzero(
            agreement_cpu & (prediction_cpu == class_index)
        )
        if indices.size == 0:
            continue
        count = max(1, int(math.ceil(indices.size * LOCKED_DELAY_FRACTION)))
        confidence = probability_cpu[indices, class_index]
        order = np.argsort(confidence, kind="stable")
        chosen = torch.from_numpy(indices[order[:count]]).to(delayed.device)
        delayed[chosen] = True
        counts_by_class[class_index] = count

    retained = agreement & ~delayed
    if torch.any(delayed & ~agreement):
        raise RuntimeError("delay selector chose a non-agreement sample")
    if int(delayed.sum().item()) + int(retained.sum().item()) != int(
        agreement.sum().item()
    ):
        raise RuntimeError("delay selector did not partition agreements")
    return {
        "delayed": delayed,
        "retained_matching": retained,
        "counts_by_class": counts_by_class,
    }
