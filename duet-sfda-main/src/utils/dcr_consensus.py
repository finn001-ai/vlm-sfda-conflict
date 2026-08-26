"""Label-free consensus primitives used while building DCR memory.

The functions in this module operate on complete class distributions.  They
do not select a conflict subset and never consume target labels.
"""

from __future__ import annotations

import torch


def _validate_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("expert predictions must be [samples, classes]")
    if left.shape != right.shape:
        raise ValueError("expert predictions must have matching shapes")
    if left.shape[1] < 2:
        raise ValueError("anchored consensus requires at least two classes")


def centered_log_probability(
    probability: torch.Tensor,
    *,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Return the unique zero-mean log-probability coordinates."""
    if probability.ndim != 2:
        raise ValueError("probability must be [samples, classes]")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    log_probability = probability.float().clamp_min(epsilon).log()
    return log_probability - log_probability.mean(dim=1, keepdim=True)


def entropy_concentration(
    probability: torch.Tensor,
    *,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Compute max(log(K) - H(p), epsilon) for every sample."""
    if probability.ndim != 2:
        raise ValueError("probability must be [samples, classes]")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    probability = probability.float().clamp_min(epsilon)
    probability = probability / probability.sum(dim=1, keepdim=True)
    entropy = -(probability * probability.log()).sum(dim=1)
    log_classes = probability.new_tensor(probability.shape[1]).log()
    return (log_classes - entropy).clamp_min(epsilon)


def entropy_weighted_poe(
    left_probability: torch.Tensor,
    right_probability: torch.Tensor,
    *,
    epsilon: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Entropy-conditioned reverse-KL barycenter of two predictions.

    Returns both its centered-logit state and normalized probability, together
    with the two within-sample expert weights.
    """
    _validate_pair(left_probability, right_probability)
    left = left_probability.float()
    right = right_probability.float()
    left = left / left.sum(dim=1, keepdim=True).clamp_min(epsilon)
    right = right / right.sum(dim=1, keepdim=True).clamp_min(epsilon)
    left_score = entropy_concentration(left, epsilon=epsilon)
    right_score = entropy_concentration(right, epsilon=epsilon)
    score_sum = (left_score + right_score).clamp_min(epsilon)
    left_weight = left_score / score_sum
    right_weight = right_score / score_sum
    centered = (
        left_weight.unsqueeze(1)
        * centered_log_probability(left, epsilon=epsilon)
        + right_weight.unsqueeze(1)
        * centered_log_probability(right, epsilon=epsilon)
    )
    return {
        "centered": centered,
        "probability": centered.softmax(dim=1),
        "left_weight": left_weight,
        "right_weight": right_weight,
    }


def average_rank(values: torch.Tensor) -> torch.Tensor:
    """Zero-based ascending ranks with average ranks for exact ties."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.numel() == 0:
        return values.float()
    sorted_values, order = torch.sort(values.float(), stable=True)
    _, inverse, counts = torch.unique_consecutive(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    ends = counts.cumsum(dim=0).float() - 1.0
    starts = ends - counts.float() + 1.0
    group_average = 0.5 * (starts + ends)
    sorted_ranks = group_average[inverse]
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def consensus_shift_factors(
    consensus_probability: torch.Tensor,
    *,
    epoch: int,
    total_epochs: int,
    strength: float = 0.5,
    epsilon: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Compute entropy-rank CSM factors for one complete target set."""
    if consensus_probability.ndim != 2:
        raise ValueError("consensus_probability must be [samples, classes]")
    if total_epochs < 2:
        raise ValueError("total_epochs must be at least 2")
    if epoch < 0 or epoch >= total_epochs:
        raise ValueError("epoch must be in [0, total_epochs)")
    if not 0.0 <= strength < 1.0:
        raise ValueError("strength must satisfy 0 <= strength < 1")

    probability = consensus_probability.float().clamp_min(epsilon)
    probability = probability / probability.sum(dim=1, keepdim=True)
    entropy = -(probability * probability.log()).sum(dim=1)
    ranks = average_rank(entropy)
    if probability.shape[0] == 1:
        uncertainty_rank = torch.zeros_like(ranks)
    else:
        uncertainty_rank = 2.0 * ranks / float(probability.shape[0] - 1) - 1.0
    decay = 1.0 - float(epoch) / float(total_epochs - 1)
    gamma = 1.0 + float(strength) * decay * uncertainty_rank
    return {
        "entropy": entropy,
        "rank": uncertainty_rank,
        "gamma": gamma,
    }


def modulate_anchored_consensus(
    anchor_centered: torch.Tensor,
    dynamic_centered: torch.Tensor,
    gamma: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rescale a dynamic consensus displacement from its fixed anchor."""
    if anchor_centered.shape != dynamic_centered.shape:
        raise ValueError("anchor and dynamic centered logits must match")
    if anchor_centered.ndim != 2:
        raise ValueError("centered logits must be [samples, classes]")
    if gamma.ndim != 1 or gamma.shape[0] != anchor_centered.shape[0]:
        raise ValueError("gamma must contain one value per sample")
    centered = anchor_centered + gamma.float().unsqueeze(1) * (
        dynamic_centered - anchor_centered
    )
    return {"centered": centered, "probability": centered.softmax(dim=1)}


def iic_mutual_information_loss(
    prediction: torch.Tensor,
    consensus: torch.Tensor,
    *,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Negative batch mutual information used for consensus alignment."""
    _validate_pair(prediction, consensus)
    if prediction.shape[0] < 1:
        raise ValueError("IIC requires a non-empty batch")
    prediction = prediction.float()
    consensus = consensus.float()
    joint = prediction.transpose(0, 1) @ consensus
    joint = joint / joint.sum().clamp_min(epsilon)
    left_marginal = joint.sum(dim=1, keepdim=True)
    right_marginal = joint.sum(dim=0, keepdim=True)
    joint_safe = joint.clamp_min(epsilon)
    return -(
        joint
        * (
            joint_safe.log()
            - left_marginal.clamp_min(epsilon).log()
            - right_marginal.clamp_min(epsilon).log()
        )
    ).sum()


def prediction_diversity_entropy(
    prediction: torch.Tensor,
    *,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Entropy of the batch-mean prediction (to be maximized)."""
    if prediction.ndim != 2 or prediction.shape[0] < 1:
        raise ValueError("prediction must be a non-empty [samples, classes] tensor")
    mean_prediction = prediction.float().mean(dim=0)
    mean_prediction = mean_prediction / mean_prediction.sum().clamp_min(epsilon)
    return -(
        mean_prediction * mean_prediction.clamp_min(epsilon).log()
    ).sum()
