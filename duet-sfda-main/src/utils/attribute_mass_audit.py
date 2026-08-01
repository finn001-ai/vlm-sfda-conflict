"""Label-free probability-mass redistribution for task/CLIP conflicts.

The utility changes only the probability mass already assigned by CLIP to the
two conflicting top-1 candidates.  It never reads target labels and does not
contain training or model-update code.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _stable_sigmoid(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value, dtype=np.float64)
    nonnegative = value >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exp_value = np.exp(value[~nonnegative])
    result[~nonnegative] = exp_value / (1.0 + exp_value)
    return result


def redistribute_pairwise_attribute_mass(
    clip_probability: np.ndarray,
    task_prediction: np.ndarray,
    clip_prediction: np.ndarray,
    attribute_margin: np.ndarray,
    *,
    clip_logit_scale: float,
) -> dict[str, np.ndarray]:
    """Redistribute CLIP pair mass with its own scaled attribute log odds.

    ``attribute_margin`` is the task-candidate minus CLIP-candidate cosine
    margin for two fixed templates and four fixed visible-attribute families.
    The CLIP model's frozen learned logit scale converts the mean cosine margin
    into a two-candidate posterior.  All probability outside the candidate pair
    remains bit-for-bit unchanged.
    """
    probability = np.asarray(clip_probability, dtype=np.float64)
    task = np.asarray(task_prediction, dtype=np.int64)
    clip = np.asarray(clip_prediction, dtype=np.int64)
    margin = np.asarray(attribute_margin, dtype=np.float64)

    if probability.ndim != 2 or probability.shape[1] < 2:
        raise ValueError("clip_probability must have shape [sample, class]")
    sample_count, class_count = probability.shape
    if task.shape != (sample_count,) or clip.shape != (sample_count,):
        raise ValueError("predictions must contain one entry per sample")
    if margin.shape != (sample_count, 2, 4):
        raise ValueError("attribute_margin must have shape [sample, 2, 4]")
    if np.any(task < 0) or np.any(task >= class_count):
        raise ValueError("task_prediction is outside the class range")
    if np.any(clip < 0) or np.any(clip >= class_count):
        raise ValueError("clip_prediction is outside the class range")
    if np.any(task == clip):
        raise ValueError("mass redistribution accepts conflict rows only")
    if not np.isfinite(clip_logit_scale) or clip_logit_scale <= 0.0:
        raise ValueError("clip_logit_scale must be finite and positive")
    if not np.isfinite(probability).all() or not np.isfinite(margin).all():
        raise ValueError("probabilities and attribute margins must be finite")
    if np.any(probability < 0.0):
        raise ValueError("clip_probability must be non-negative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("clip_probability rows must sum to one")

    mean_margin = margin.mean(axis=(1, 2))
    task_fraction = _stable_sigmoid(clip_logit_scale * mean_margin)
    row = np.arange(sample_count)
    pair_mass = probability[row, task] + probability[row, clip]
    redistributed = probability.copy()
    redistributed[row, task] = pair_mass * task_fraction
    redistributed[row, clip] = pair_mass * (1.0 - task_fraction)

    if not np.isfinite(redistributed).all() or np.any(redistributed < 0.0):
        raise RuntimeError("redistributed probabilities are invalid")
    if not np.allclose(
        redistributed.sum(axis=1),
        probability.sum(axis=1),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("redistribution did not conserve probability mass")
    outside = np.ones_like(probability, dtype=bool)
    outside[row, task] = False
    outside[row, clip] = False
    if not np.array_equal(redistributed[outside], probability[outside]):
        raise RuntimeError("redistribution changed probability outside the pair")

    return {
        "probability": redistributed,
        "attribute_mean_margin": mean_margin,
        "task_fraction": task_fraction,
        "pair_mass": pair_mass,
    }


def entropy_anchored_attribute_mass(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
    task_prediction: np.ndarray,
    clip_prediction: np.ndarray,
    attribute_margin: np.ndarray,
    *,
    clip_logit_scale: float,
) -> dict[str, np.ndarray]:
    """Temper attribute evidence with two independent confidence signals.

    The candidate keeps CLIP's pair posterior when CLIP is certain. Attribute
    evidence receives weight only to the extent that CLIP is uncertain and the
    task model is certain within the same candidate pair. The weight is the
    product of normalized binary CLIP entropy and task certainty, so the rule
    has no fitted threshold or tunable mixing coefficient.
    """
    raw_attribute = redistribute_pairwise_attribute_mass(
        clip_probability,
        task_prediction,
        clip_prediction,
        attribute_margin,
        clip_logit_scale=clip_logit_scale,
    )
    task_probability = np.asarray(task_probability, dtype=np.float64)
    clip_probability = np.asarray(clip_probability, dtype=np.float64)
    task = np.asarray(task_prediction, dtype=np.int64)
    clip = np.asarray(clip_prediction, dtype=np.int64)
    if task_probability.shape != clip_probability.shape:
        raise ValueError("task_probability and clip_probability shapes must match")
    if not np.isfinite(task_probability).all() or np.any(task_probability < 0.0):
        raise ValueError("task_probability must be finite and non-negative")
    if not np.allclose(task_probability.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("task_probability rows must sum to one")

    row = np.arange(task.size)
    clip_pair_mass = raw_attribute["pair_mass"]
    task_pair_mass = task_probability[row, task] + task_probability[row, clip]
    if np.any(clip_pair_mass <= 0.0) or np.any(task_pair_mass <= 0.0):
        raise ValueError("candidate-pair probability mass must be positive")

    epsilon = np.finfo(np.float64).eps
    clip_fraction = np.clip(
        clip_probability[row, task] / clip_pair_mass, epsilon, 1.0 - epsilon
    )
    task_fraction = np.clip(
        task_probability[row, task] / task_pair_mass, epsilon, 1.0 - epsilon
    )

    def normalized_binary_entropy(fraction: np.ndarray) -> np.ndarray:
        return -(
            fraction * np.log(fraction) + (1.0 - fraction) * np.log1p(-fraction)
        ) / np.log(2.0)

    clip_entropy = normalized_binary_entropy(clip_fraction)
    task_entropy = normalized_binary_entropy(task_fraction)
    attribute_weight = clip_entropy * (1.0 - task_entropy)
    clip_log_odds = np.log(clip_fraction) - np.log1p(-clip_fraction)
    attribute_log_odds = clip_logit_scale * raw_attribute["attribute_mean_margin"]
    anchored_log_odds = (
        1.0 - attribute_weight
    ) * clip_log_odds + attribute_weight * attribute_log_odds
    anchored_fraction = _stable_sigmoid(anchored_log_odds)

    probability = clip_probability.copy()
    probability[row, task] = clip_pair_mass * anchored_fraction
    probability[row, clip] = clip_pair_mass * (1.0 - anchored_fraction)
    if not np.allclose(
        probability.sum(axis=1),
        clip_probability.sum(axis=1),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("entropy anchoring did not conserve probability mass")
    outside = np.ones_like(probability, dtype=bool)
    outside[row, task] = False
    outside[row, clip] = False
    if not np.array_equal(probability[outside], clip_probability[outside]):
        raise RuntimeError("entropy anchoring changed probability outside the pair")

    return {
        "probability": probability,
        "attribute_mean_margin": raw_attribute["attribute_mean_margin"],
        "attribute_fraction": raw_attribute["task_fraction"],
        "clip_pair_fraction": clip_fraction,
        "task_pair_fraction": task_fraction,
        "clip_pair_entropy": clip_entropy,
        "task_pair_entropy": task_entropy,
        "attribute_weight": attribute_weight,
        "anchored_fraction": anchored_fraction,
        "pair_mass": clip_pair_mass,
    }


def paired_mean_bootstrap_ci(
    improvement: np.ndarray,
    *,
    repeats: int = 2_000,
    seed: int = 2_020,
    batch_size: int = 100,
) -> tuple[float, float]:
    """Return a paired-bootstrap 95% CI for a mean improvement."""
    values = np.asarray(improvement, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("improvement must be a non-empty 1-D array")
    if not np.isfinite(values).all():
        raise ValueError("improvement must be finite")
    if repeats <= 0 or batch_size <= 0:
        raise ValueError("repeats and batch_size must be positive")

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats, dtype=np.float64)
    written = 0
    while written < repeats:
        current = min(batch_size, repeats - written)
        indices = rng.integers(0, values.size, size=(current, values.size))
        bootstrap[written : written + current] = values[indices].mean(axis=1)
        written += current
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def evaluate_attribute_mass_gate(
    *,
    input_contract_valid: bool,
    comparison_checks: dict[str, dict[str, bool]],
    car_nll_improvement: float,
    truck_nll_improvement: float,
    car_brier_improvement: float,
    truck_brier_improvement: float,
    max_abs_class_mass_shift_pp: float,
    max_allowed_class_mass_shift_pp: float = 1.0,
) -> dict[str, Any]:
    """Apply a fixed oracle-diagnostic gate without fitting any threshold."""
    required_baselines = ("fixed_clip", "arithmetic", "rms")
    comparison_passed = {
        baseline: bool(
            comparison_checks.get(baseline, {}).get("nll_ci_lower_positive", False)
            and comparison_checks.get(baseline, {}).get(
                "brier_ci_lower_positive", False
            )
            and comparison_checks.get(baseline, {}).get(
                "true_probability_ci_lower_positive", False
            )
        )
        for baseline in required_baselines
    }
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        **{
            f"beats_{baseline}_on_all_three_paired_metrics": passed
            for baseline, passed in comparison_passed.items()
        },
        "car_nll_nonworse": car_nll_improvement >= 0.0,
        "truck_nll_nonworse": truck_nll_improvement >= 0.0,
        "car_brier_nonworse": car_brier_improvement >= 0.0,
        "truck_brier_nonworse": truck_brier_improvement >= 0.0,
        "class_mass_shift_at_most_1pp": (
            max_abs_class_mass_shift_pp <= max_allowed_class_mass_shift_pp
        ),
    }
    return {
        "decision": "PASS_OFFLINE_GATE" if all(checks.values()) else "REJECT",
        "thresholds": {
            "paired_metric_ci_lower_must_be_positive": True,
            "car_and_truck_nll_and_brier_must_be_nonworse": True,
            "max_abs_class_mass_shift_pp": max_allowed_class_mass_shift_pp,
            "target_label_fitted_thresholds": False,
        },
        "checks": checks,
    }


def evaluate_attribute_reliability_gate(
    *,
    input_contract_valid: bool,
    fixed_clip_checks: dict[str, bool],
    accuracy_gain_pp: float,
    accuracy_ci_pp: tuple[float, float],
    car_metrics: dict[str, float],
    truck_metrics: dict[str, float],
    noncar_net_corrections: int,
    max_abs_class_mass_shift_pp: float,
    min_accuracy_gain_pp: float = 1.0,
    max_allowed_class_mass_shift_pp: float = 1.0,
) -> dict[str, Any]:
    """Gate the parameter-free entropy-anchored attribute audit."""
    all_soft_metrics_better = all(
        fixed_clip_checks.get(f"{name}_ci_lower_positive", False)
        for name in ("nll", "brier", "true_probability")
    )
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "beats_fixed_clip_on_all_three_paired_soft_metrics": bool(
            all_soft_metrics_better
        ),
        "accuracy_gain_at_least_1pp": accuracy_gain_pp >= min_accuracy_gain_pp,
        "accuracy_gain_ci_lower_positive": accuracy_ci_pp[0] > 0.0,
        "car_nll_nonworse": car_metrics["nll_improvement"] >= 0.0,
        "car_brier_nonworse": car_metrics["brier_improvement"] >= 0.0,
        "car_accuracy_nonworse": car_metrics["accuracy_gain_pp"] >= 0.0,
        "truck_nll_nonworse": truck_metrics["nll_improvement"] >= 0.0,
        "truck_brier_nonworse": truck_metrics["brier_improvement"] >= 0.0,
        "truck_accuracy_nonworse": truck_metrics["accuracy_gain_pp"] >= 0.0,
        "noncar_net_corrections_nonnegative": noncar_net_corrections >= 0,
        "class_mass_shift_at_most_1pp": (
            max_abs_class_mass_shift_pp <= max_allowed_class_mass_shift_pp
        ),
    }
    return {
        "decision": "PASS_OFFLINE_GATE" if all(checks.values()) else "REJECT",
        "thresholds": {
            "min_accuracy_gain_pp": min_accuracy_gain_pp,
            "accuracy_and_soft_metric_ci_lower_must_be_positive": True,
            "car_and_truck_metrics_must_be_nonworse": True,
            "noncar_net_corrections_must_be_nonnegative": True,
            "max_abs_class_mass_shift_pp": max_allowed_class_mass_shift_pp,
            "target_label_fitted_thresholds": False,
        },
        "checks": checks,
    }
