"""Label-free attribute-reliability targets for DUET task/CLIP conflicts."""

from __future__ import annotations

from typing import Any

import torch


def pairwise_attribute_margin(
    image_feature: torch.Tensor,
    attribute_text_feature: torch.Tensor,
    task_prediction: torch.Tensor,
    clip_prediction: torch.Tensor,
) -> torch.Tensor:
    """Return task-minus-CLIP cosine margins for each template/family."""
    if image_feature.ndim != 2:
        raise ValueError("image_feature must have shape [sample, embedding]")
    if attribute_text_feature.ndim != 4:
        raise ValueError(
            "attribute_text_feature must have shape "
            "[class, template, family, embedding]"
        )
    if attribute_text_feature.shape[-1] != image_feature.shape[-1]:
        raise ValueError("image and attribute embeddings do not match")
    sample_count = image_feature.shape[0]
    if task_prediction.shape != (sample_count,) or clip_prediction.shape != (
        sample_count,
    ):
        raise ValueError("predictions must contain one class per sample")
    if task_prediction.dtype != torch.long or clip_prediction.dtype != torch.long:
        raise TypeError("predictions must use torch.long")
    class_count = attribute_text_feature.shape[0]
    if torch.any(task_prediction < 0) or torch.any(task_prediction >= class_count):
        raise ValueError("task_prediction is outside the class range")
    if torch.any(clip_prediction < 0) or torch.any(clip_prediction >= class_count):
        raise ValueError("clip_prediction is outside the class range")
    task_text = attribute_text_feature[task_prediction]
    clip_text = attribute_text_feature[clip_prediction]
    return torch.einsum(
        "nd,ntfd->ntf",
        image_feature,
        task_text - clip_text,
    )


def entropy_anchored_attribute_target(
    task_probability: torch.Tensor,
    clip_probability: torch.Tensor,
    task_prediction: torch.Tensor,
    clip_prediction: torch.Tensor,
    attribute_margin: torch.Tensor,
    *,
    clip_logit_scale: float,
) -> dict[str, Any]:
    """Build the locked entropy-anchored soft target without fitted knobs.

    Inputs must contain conflict rows only. Probability outside each row's
    task/CLIP candidate pair remains unchanged.
    """
    if task_probability.shape != clip_probability.shape:
        raise ValueError("task_probability and clip_probability shapes must match")
    if task_probability.ndim != 2:
        raise ValueError("probabilities must have shape [sample, class]")
    if (
        not task_probability.is_floating_point()
        or not clip_probability.is_floating_point()
    ):
        raise TypeError("probabilities must be floating point")
    sample_count, class_count = task_probability.shape
    if task_prediction.shape != (sample_count,) or clip_prediction.shape != (
        sample_count,
    ):
        raise ValueError("predictions must contain one class per sample")
    if task_prediction.dtype != torch.long or clip_prediction.dtype != torch.long:
        raise TypeError("predictions must use torch.long")
    if attribute_margin.shape != (sample_count, 2, 4):
        raise ValueError("attribute_margin must have shape [sample, 2, 4]")
    if torch.any(task_prediction == clip_prediction):
        raise ValueError("attribute target accepts conflict rows only")
    if torch.any(task_prediction < 0) or torch.any(task_prediction >= class_count):
        raise ValueError("task_prediction is outside the class range")
    if torch.any(clip_prediction < 0) or torch.any(clip_prediction >= class_count):
        raise ValueError("clip_prediction is outside the class range")
    for name, value in (
        ("task_probability", task_probability),
        ("clip_probability", clip_probability),
        ("attribute_margin", attribute_margin),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if torch.any(task_probability < 0) or torch.any(clip_probability < 0):
        raise ValueError("probabilities must be non-negative")
    ones = torch.ones(
        sample_count, dtype=task_probability.dtype, device=task_probability.device
    )
    if not torch.allclose(task_probability.sum(dim=1), ones, atol=1e-5, rtol=1e-5):
        raise ValueError("task_probability rows must sum to one")
    if not torch.allclose(clip_probability.sum(dim=1), ones, atol=1e-5, rtol=1e-5):
        raise ValueError("clip_probability rows must sum to one")
    if not torch.isfinite(torch.tensor(clip_logit_scale)) or clip_logit_scale <= 0.0:
        raise ValueError("clip_logit_scale must be finite and positive")

    row = torch.arange(sample_count, device=task_probability.device)
    task_prediction = task_prediction.to(task_probability.device)
    clip_prediction = clip_prediction.to(task_probability.device)
    attribute_margin = attribute_margin.to(
        device=task_probability.device,
        dtype=task_probability.dtype,
    )
    clip_pair_mass = (
        clip_probability[row, task_prediction] + clip_probability[row, clip_prediction]
    )
    task_pair_mass = (
        task_probability[row, task_prediction] + task_probability[row, clip_prediction]
    )
    if torch.any(clip_pair_mass <= 0.0) or torch.any(task_pair_mass <= 0.0):
        raise ValueError("candidate-pair probability mass must be positive")

    epsilon = torch.finfo(task_probability.dtype).eps
    clip_fraction = (clip_probability[row, task_prediction] / clip_pair_mass).clamp(
        epsilon, 1.0 - epsilon
    )
    task_fraction = (task_probability[row, task_prediction] / task_pair_mass).clamp(
        epsilon, 1.0 - epsilon
    )

    def normalized_binary_entropy(fraction: torch.Tensor) -> torch.Tensor:
        return -(
            fraction * torch.log(fraction) + (1.0 - fraction) * torch.log1p(-fraction)
        ) / torch.log(torch.tensor(2.0, dtype=fraction.dtype, device=fraction.device))

    clip_entropy = normalized_binary_entropy(clip_fraction)
    task_entropy = normalized_binary_entropy(task_fraction)
    attribute_weight = clip_entropy * (1.0 - task_entropy)
    clip_log_odds = torch.log(clip_fraction) - torch.log1p(-clip_fraction)
    attribute_mean_margin = attribute_margin.mean(dim=(1, 2))
    attribute_log_odds = clip_logit_scale * attribute_mean_margin
    anchored_log_odds = (
        1.0 - attribute_weight
    ) * clip_log_odds + attribute_weight * attribute_log_odds
    anchored_fraction = torch.sigmoid(anchored_log_odds)

    probability = clip_probability.clone()
    probability[row, task_prediction] = clip_pair_mass * anchored_fraction
    probability[row, clip_prediction] = clip_pair_mass * (1.0 - anchored_fraction)
    if not torch.allclose(
        probability.sum(dim=1),
        clip_probability.sum(dim=1),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("attribute target did not conserve probability mass")
    return {
        "probability": probability,
        "attribute_mean_margin": attribute_mean_margin,
        "clip_pair_fraction": clip_fraction,
        "task_pair_fraction": task_fraction,
        "clip_pair_entropy": clip_entropy,
        "task_pair_entropy": task_entropy,
        "attribute_weight": attribute_weight,
        "anchored_fraction": anchored_fraction,
        "pair_mass": clip_pair_mass,
    }
