"""First-cycle-only class-prior calibration for DUET probabilities."""

from __future__ import annotations

import torch


def prior_calibrate(
    probability: torch.Tensor,
    *,
    power: float,
    epsilon: float,
) -> torch.Tensor:
    """Divide probabilities by their empirical class marginal."""
    if probability.ndim != 2:
        raise ValueError("probability must be [samples, classes]")
    if power < 0:
        raise ValueError("power must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    probability = probability.float()
    prior = probability.mean(dim=0).clamp_min(epsilon)
    calibrated = probability / prior.pow(power).unsqueeze(0)
    return calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(epsilon)


def apply_first_cycle_prior(
    source_probability: torch.Tensor,
    clip_probability: torch.Tensor,
    *,
    curr_cycle: int,
    power: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Calibrate both DUET views only when ``curr_cycle == 0``."""
    if source_probability.shape != clip_probability.shape:
        raise ValueError("source and CLIP probabilities must have matching shapes")
    if curr_cycle < 0:
        raise ValueError("curr_cycle must be non-negative")
    if curr_cycle != 0:
        return source_probability, clip_probability, False

    return (
        prior_calibrate(
            source_probability,
            power=power,
            epsilon=epsilon,
        ),
        prior_calibrate(
            clip_probability,
            power=power,
            epsilon=epsilon,
        ),
        True,
    )
