"""Pseudo-label supervision utilities shared by DCCL memory modes."""

import torch
import torch.nn.functional as F


def dual_tier_supervision(
    current_mask,
    stable_mask,
    mix_conf,
    pending_scale,
    *,
    warmup,
):
    """Build Stable/Pending/Conflict masks and per-sample CE weights."""
    if not 0.0 <= float(pending_scale) <= 1.0:
        raise ValueError("pending_scale must be in [0, 1]")
    if current_mask.shape != stable_mask.shape:
        raise ValueError("current_mask and stable_mask must have equal shapes")
    if current_mask.shape != mix_conf.shape:
        raise ValueError("mix_conf must have one value per pseudo label")
    if (stable_mask & ~current_mask).any():
        raise ValueError("stable samples must also be currently consistent")

    pending_mask = current_mask & ~stable_mask
    selected_mask = stable_mask | pending_mask
    weights = torch.zeros_like(mix_conf)
    if warmup:
        weights[current_mask] = 1.0
    else:
        weights[stable_mask] = 1.0
        weights[pending_mask] = float(pending_scale) * mix_conf[pending_mask]
    return selected_mask, pending_mask, weights


def weighted_cross_entropy(logits, labels, weights, epsilon=1e-6):
    """Normalize hard-label CE by the effective supervision weight."""
    if logits.size(0) != labels.numel() or labels.numel() != weights.numel():
        raise ValueError("logits, labels, and weights must share batch size")
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return (per_sample * weights).sum() / weights.sum().clamp_min(epsilon)
