"""Per-sample delayed expert credit for Task/CLIP co-adaptation.

The update is label-free.  Each sample keeps its own discounted history of
how well the previous Task and CLIP distributions predicted the next shared
Task/CLIP observation.  No sample is used to train a router for another
sample, and no hard confidence threshold is used.
"""

from __future__ import annotations

import math

import torch


def _normalize_probability(
    probability: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if probability.ndim != 2:
        raise ValueError("probability must be [samples, classes]")
    if probability.shape[1] < 2:
        raise ValueError("delayed credit requires at least two classes")
    probability = probability.float().clamp_min(epsilon)
    return probability / probability.sum(dim=1, keepdim=True)


def normalized_js_divergence(
    left: torch.Tensor,
    right: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return row-wise Jensen-Shannon divergence normalized to ``[0, 1]``."""
    if left.shape != right.shape:
        raise ValueError("left and right probabilities must have the same shape")
    left = _normalize_probability(left, epsilon)
    right = _normalize_probability(right, epsilon)
    midpoint = 0.5 * (left + right)
    left_kl = (left * (left.log() - midpoint.log())).sum(dim=1)
    right_kl = (right * (right.log() - midpoint.log())).sum(dim=1)
    divergence = 0.5 * (left_kl + right_kl) / math.log(2.0)
    return divergence.clamp(0.0, 1.0)


def initialize_delayed_credit(
    task_probability: torch.Tensor,
    clip_probability: torch.Tensor,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Initialize a full-distribution memory without an initial-model anchor."""
    if task_probability.shape != clip_probability.shape:
        raise ValueError("Task and CLIP probabilities must have the same shape")
    task_probability = _normalize_probability(task_probability, epsilon).cpu()
    clip_probability = _normalize_probability(clip_probability, epsilon).cpu()
    sample_count = int(task_probability.shape[0])
    memory = 0.5 * (task_probability + clip_probability)
    return {
        "memory": memory,
        "previous_task": task_probability.clone(),
        "previous_clip": clip_probability.clone(),
        "task_loss_sum": torch.zeros(sample_count, dtype=torch.float32),
        "clip_loss_sum": torch.zeros(sample_count, dtype=torch.float32),
        "feedback_mass": torch.zeros(sample_count, dtype=torch.float32),
        "task_weight": torch.full((sample_count,), 0.5, dtype=torch.float32),
        "clip_weight": torch.full((sample_count,), 0.5, dtype=torch.float32),
    }


@torch.no_grad()
def update_delayed_credit(
    state: dict[str, torch.Tensor],
    task_probability: torch.Tensor,
    clip_probability: torch.Tensor,
    *,
    decay: float = 0.9,
    credit_eta: float = 4.0,
    memory_update_rate: float = 0.5,
    epsilon: float = 1e-8,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Apply one label-free delayed-credit update to every sample.

    The current arithmetic Task/CLIP mean is treated as the newly observed
    outcome.  Previous expert distributions receive a continuous discounted
    loss according to how well they predicted this outcome.  Current expert
    weights are derived from those historical losses, then written into a
    moving full-distribution memory.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    if credit_eta <= 0.0:
        raise ValueError("credit_eta must be positive")
    if not 0.0 < memory_update_rate <= 1.0:
        raise ValueError("memory_update_rate must be in (0, 1]")

    task_probability = _normalize_probability(task_probability, epsilon).cpu()
    clip_probability = _normalize_probability(clip_probability, epsilon).cpu()
    memory_before = _normalize_probability(state["memory"], epsilon).cpu()
    previous_task = _normalize_probability(
        state["previous_task"], epsilon
    ).cpu()
    previous_clip = _normalize_probability(
        state["previous_clip"], epsilon
    ).cpu()
    expected_shape = memory_before.shape
    if task_probability.shape != expected_shape or clip_probability.shape != expected_shape:
        raise ValueError("current probabilities do not match delayed-credit state")

    outcome = 0.5 * (task_probability + clip_probability)
    agreement = 1.0 - normalized_js_divergence(
        task_probability,
        clip_probability,
        epsilon,
    )
    temporal_stability = 1.0 - normalized_js_divergence(
        outcome,
        memory_before,
        epsilon,
    )
    feedback = (agreement * temporal_stability).clamp(0.0, 1.0)

    task_delayed_loss = normalized_js_divergence(
        previous_task,
        outcome,
        epsilon,
    )
    clip_delayed_loss = normalized_js_divergence(
        previous_clip,
        outcome,
        epsilon,
    )
    task_loss_sum = decay * state["task_loss_sum"].float().cpu()
    clip_loss_sum = decay * state["clip_loss_sum"].float().cpu()
    feedback_mass = decay * state["feedback_mass"].float().cpu()
    task_loss_sum = task_loss_sum + feedback * task_delayed_loss
    clip_loss_sum = clip_loss_sum + feedback * clip_delayed_loss
    feedback_mass = feedback_mass + feedback

    valid_history = feedback_mass > epsilon
    average_task_loss = torch.where(
        valid_history,
        task_loss_sum / feedback_mass.clamp_min(epsilon),
        torch.zeros_like(feedback_mass),
    )
    average_clip_loss = torch.where(
        valid_history,
        clip_loss_sum / feedback_mass.clamp_min(epsilon),
        torch.zeros_like(feedback_mass),
    )
    expert_weight = torch.softmax(
        -credit_eta * torch.stack([average_task_loss, average_clip_loss], dim=1),
        dim=1,
    )
    task_weight = expert_weight[:, 0]
    clip_weight = expert_weight[:, 1]
    instantaneous = (
        task_weight.unsqueeze(1) * task_probability
        + clip_weight.unsqueeze(1) * clip_probability
    )

    # Every sample is updated.  Stable agreement only changes the update rate
    # continuously; it never gates a sample out of the memory or loss.
    update_rate = memory_update_rate * (0.5 + 0.5 * feedback)
    memory = (
        (1.0 - update_rate.unsqueeze(1)) * memory_before
        + update_rate.unsqueeze(1) * instantaneous
    )
    memory = _normalize_probability(memory, epsilon)

    new_state = {
        "memory": memory,
        "previous_task": task_probability.clone(),
        "previous_clip": clip_probability.clone(),
        "task_loss_sum": task_loss_sum,
        "clip_loss_sum": clip_loss_sum,
        "feedback_mass": feedback_mass,
        "task_weight": task_weight,
        "clip_weight": clip_weight,
    }
    diagnostics = {
        "feedback": feedback,
        "agreement": agreement,
        "temporal_stability": temporal_stability,
        "task_delayed_loss": task_delayed_loss,
        "clip_delayed_loss": clip_delayed_loss,
        "average_task_loss": average_task_loss,
        "average_clip_loss": average_clip_loss,
        "task_weight": task_weight,
        "clip_weight": clip_weight,
        "update_rate": update_rate,
        "memory_shift_l1": (memory - memory_before).abs().sum(dim=1),
    }
    return new_state, diagnostics
