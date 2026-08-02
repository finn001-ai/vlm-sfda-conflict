"""Helpers for mapping locked DUET logit descents through the frozen head."""

from __future__ import annotations

from typing import Any

import numpy as np


def effective_weight_normalized_linear(
    weight_v: np.ndarray,
    weight_g: np.ndarray,
) -> np.ndarray:
    """Recover the effective weight of PyTorch's legacy ``weight_norm``.

    DUET applies ``weight_norm`` to the frozen classifier's linear weight with
    its default output dimension.  Each output-class row is therefore scaled
    to the magnitude stored in ``weight_g``.
    """
    vector = np.asarray(weight_v, dtype=np.float64)
    magnitude = np.asarray(weight_g, dtype=np.float64)
    if vector.ndim != 2 or vector.shape[0] < 2 or vector.shape[1] < 2:
        raise ValueError("weight_v must have shape [class, feature]")
    if magnitude.shape not in {(vector.shape[0],), (vector.shape[0], 1)}:
        raise ValueError("weight_g must contain one magnitude per class")
    if not np.isfinite(vector).all() or not np.isfinite(magnitude).all():
        raise ValueError("weight-normalization tensors must be finite")
    row_norm = np.linalg.norm(vector, axis=1, keepdims=True)
    if np.any(row_norm <= 0.0):
        raise ValueError("weight_v rows must have positive norm")
    return vector * magnitude.reshape(-1, 1) / row_norm


def map_joint_logit_descent_to_feature(
    joint_logit_descent: np.ndarray,
    classifier_weight: np.ndarray,
) -> np.ndarray:
    """Map concatenated weak/strong logit descents to feature descents."""
    descent = np.asarray(joint_logit_descent, dtype=np.float64)
    weight = np.asarray(classifier_weight, dtype=np.float64)
    if descent.ndim != 2 or descent.shape[0] == 0:
        raise ValueError("joint_logit_descent must be a non-empty matrix")
    if weight.ndim != 2 or weight.shape[0] < 2 or weight.shape[1] < 2:
        raise ValueError("classifier_weight must have shape [class, feature]")
    if descent.shape[1] != 2 * weight.shape[0]:
        raise ValueError("joint descent must concatenate weak and strong classes")
    if not np.isfinite(descent).all() or not np.isfinite(weight).all():
        raise ValueError("descent and classifier weight must be finite")
    class_count = weight.shape[0]
    weak = descent[:, :class_count] @ weight
    strong = descent[:, class_count:] @ weight
    result = np.concatenate((weak, strong), axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError("mapped feature descent must be finite")
    return result


def classifier_probability(
    feature: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
) -> np.ndarray:
    """Replay the frozen linear classifier without constructing a model."""
    values = np.asarray(feature, dtype=np.float64)
    weight = np.asarray(classifier_weight, dtype=np.float64)
    bias = np.asarray(classifier_bias, dtype=np.float64)
    if values.ndim != 2 or weight.ndim != 2:
        raise ValueError("feature and classifier weight must be matrices")
    if values.shape[1] != weight.shape[1]:
        raise ValueError("feature dimension does not match classifier weight")
    if bias.shape != (weight.shape[0],):
        raise ValueError("classifier_bias must contain one value per class")
    if not all(np.isfinite(value).all() for value in (values, weight, bias)):
        raise ValueError("classifier replay inputs must be finite")
    logits = values @ weight.T + bias[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    return exponential / exponential.sum(axis=1, keepdims=True)


def evaluate_feature_jacobian_gate(
    *,
    input_contract_valid: bool,
    classifier_top1_reproduced: bool,
    max_probability_replay_error: float,
    overall_first_order: dict[str, Any],
    active_first_order: dict[str, Any],
    baseline_negative_burden: float,
    candidate_negative_burden: float,
    helpful_retention_pct: float,
    candidate_to_baseline_mean_norm_ratio: float,
    group_first_order_delta: dict[str, float],
) -> dict[str, Any]:
    """Gate the frozen-head Jacobian filter; passing authorizes no training."""
    for result in (overall_first_order, active_first_order):
        interval = result.get("paired_bootstrap_95_ci")
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or not all(
                np.isfinite(value)
                for value in (result.get("mean_difference"), *interval)
            )
        ):
            raise ValueError("invalid paired first-order comparison")
    if set(group_first_order_delta) != {"car", "person", "truck", "other_nine"}:
        raise ValueError("group deltas must contain car, person, truck, other_nine")
    numeric = (
        max_probability_replay_error,
        baseline_negative_burden,
        candidate_negative_burden,
        helpful_retention_pct,
        candidate_to_baseline_mean_norm_ratio,
        *group_first_order_delta.values(),
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate metrics must be finite")

    def positive_with_ci(result: dict[str, Any]) -> bool:
        return bool(
            result["mean_difference"] > 0.0
            and result["paired_bootstrap_95_ci"][0] > 0.0
        )

    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "frozen_classifier_top1_reproduced": bool(classifier_top1_reproduced),
        "max_probability_replay_error_at_most_5e_4": (
            max_probability_replay_error <= 5e-4
        ),
        "feature_overall_first_order_gain_ci_lower_positive": positive_with_ci(
            overall_first_order
        ),
        "feature_active_first_order_gain_ci_lower_positive": positive_with_ci(
            active_first_order
        ),
        "feature_negative_burden_not_worse": (
            candidate_negative_burden >= baseline_negative_burden
        ),
        "feature_helpful_first_order_retention_at_least_99pct": (
            helpful_retention_pct >= 99.0
        ),
        "feature_mean_descent_norm_inflation_at_most_1_5x": (
            candidate_to_baseline_mean_norm_ratio <= 1.5
        ),
        "car_feature_first_order_delta_nonnegative": (
            group_first_order_delta["car"] >= 0.0
        ),
        "person_feature_first_order_delta_nonnegative": (
            group_first_order_delta["person"] >= 0.0
        ),
        "truck_feature_first_order_delta_nonnegative": (
            group_first_order_delta["truck"] >= 0.0
        ),
        "other_nine_feature_first_order_delta_nonnegative": (
            group_first_order_delta["other_nine"] >= 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "NEEDS_EXACT_CONTROL_PARAMETER_AUDIT" if passed else "REJECT"
        ),
        "checks": checks,
        "thresholds": {
            "max_probability_replay_error": "<= 5e-4",
            "paired_first_order_mean_and_ci_lower": "> 0",
            "candidate_negative_burden_minus_baseline": ">= 0",
            "helpful_first_order_retention_pct": ">= 99",
            "candidate_to_baseline_mean_norm_ratio": "<= 1.5",
            "car_person_truck_other_nine_first_order_delta": ">= 0",
        },
        "training_authorized": False,
        "proxy_authorized": False,
        "gpu_authorized": False,
    }
