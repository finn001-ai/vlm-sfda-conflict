"""Structural target-head transformations."""

from __future__ import annotations

import torch


def bounded_residual_logits(
    source_logits,
    residual_logits,
    gate_logit,
    max_gate: float,
    epsilon: float,
):
    if source_logits.shape != residual_logits.shape:
        raise ValueError("Source and residual logits must have the same shape")
    if not 0.0 < max_gate <= 1.0:
        raise ValueError("Residual max gate must be in (0, 1]")
    if epsilon <= 0.0:
        raise ValueError("Residual epsilon must be positive")

    source_scale = source_logits.detach().float().std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(epsilon)
    gate = float(max_gate) * torch.sigmoid(gate_logit.float())
    bounded_delta = source_scale * torch.tanh(residual_logits.float())
    return source_logits + (gate * bounded_delta).to(source_logits.dtype)
