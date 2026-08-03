"""Pure helpers for patch-selected CLIP-KL suppression diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def consistency_logit_descent(
    weak_probability: np.ndarray,
    strong_probability: np.ndarray,
    batch_size: np.ndarray,
    *,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact DUET KL(weak || strong) weak/strong logit descents.

    ``batch_size`` is per row because the final full-target loader batch can
    be shorter.  The result includes both the positive loss coefficient and
    the production ``batchmean`` reduction.
    """
    weak = np.asarray(weak_probability, dtype=np.float64)
    strong = np.asarray(strong_probability, dtype=np.float64)
    size = np.asarray(batch_size, dtype=np.float64)
    if weak.ndim != 2 or weak.shape != strong.shape or weak.shape[0] == 0:
        raise ValueError("weak and strong probabilities must be aligned matrices")
    if size.shape != (weak.shape[0],) or np.any(size <= 0.0):
        raise ValueError("batch_size must contain one positive value per row")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and non-negative")
    for name, value in (("weak", weak), ("strong", strong)):
        if not np.isfinite(value).all() or np.any(value <= 0.0):
            raise ValueError(f"{name} probabilities must be finite and positive")
        if not np.allclose(value.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError(f"{name} probabilities must sum to one")
    weak = weak / weak.sum(axis=1, keepdims=True)
    strong = strong / strong.sum(axis=1, keepdims=True)
    log_ratio = np.log(weak) - np.log(strong)
    centered_log_ratio = log_ratio - np.sum(weak * log_ratio, axis=1, keepdims=True)
    scale = float(weight) / size
    weak_descent = -scale[:, None] * weak * centered_log_ratio
    strong_descent = scale[:, None] * (weak - strong)
    for value in (weak_descent, strong_descent):
        if not np.isfinite(value).all():
            raise RuntimeError("consistency descent must be finite")
        if not np.allclose(value.sum(axis=1), 0.0, atol=1e-12, rtol=1e-12):
            raise RuntimeError("consistency descent must sum to zero per row")
    return weak_descent, strong_descent


def evaluate_patch_cls_kl_suppression_gate(
    *,
    input_contract_valid: bool,
    heldout_selector_passed: bool,
    selected_coverage_pct: float,
    strong_replay_max_abs_error: float,
    output_first_order: dict[str, Any],
    feature_first_order: dict[str, Any],
    output_negative_burden_baseline: float,
    output_negative_burden_candidate: float,
    feature_negative_burden_baseline: float,
    feature_negative_burden_candidate: float,
    feature_helpful_retention_pct: float,
    feature_mean_norm_ratio: float,
    class_macro_feature_first_order_delta: float,
    heldout_accuracy_gain_vs_clip_pp: float,
    heldout_accuracy_ci_lower_pp: float,
    car_accuracy_delta_pp: float,
    truck_accuracy_delta_pp: float,
) -> dict[str, Any]:
    """Gate one exact no-update resident-parameter audit design."""

    def positive_with_ci(result: dict[str, Any]) -> bool:
        interval = result.get("paired_bootstrap_95_ci")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("paired result requires a two-sided interval")
        values = (result.get("mean_difference"), *interval)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("paired result must be finite")
        return bool(result["mean_difference"] > 0.0 and interval[0] > 0.0)

    numeric = (
        selected_coverage_pct,
        strong_replay_max_abs_error,
        output_negative_burden_baseline,
        output_negative_burden_candidate,
        feature_negative_burden_baseline,
        feature_negative_burden_candidate,
        feature_helpful_retention_pct,
        feature_mean_norm_ratio,
        class_macro_feature_first_order_delta,
        heldout_accuracy_gain_vs_clip_pp,
        heldout_accuracy_ci_lower_pp,
        car_accuracy_delta_pp,
        truck_accuracy_delta_pp,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate inputs must be finite")
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "heldout_selector_gate_passed": bool(heldout_selector_passed),
        "selected_coverage_between_2_and_10pct": (
            2.0 <= float(selected_coverage_pct) <= 10.0
        ),
        "exact_consistency_strong_replay_error_at_most_5e_8": (
            float(strong_replay_max_abs_error) <= 5e-8
        ),
        "output_first_order_gain_ci_lower_positive": positive_with_ci(
            output_first_order
        ),
        "feature_first_order_gain_ci_lower_positive": positive_with_ci(
            feature_first_order
        ),
        "output_negative_burden_not_worse": (
            output_negative_burden_candidate >= output_negative_burden_baseline
        ),
        "feature_negative_burden_not_worse": (
            feature_negative_burden_candidate >= feature_negative_burden_baseline
        ),
        "feature_helpful_retention_at_least_99pct": (
            feature_helpful_retention_pct >= 99.0
        ),
        "feature_mean_norm_inflation_at_most_1_5x": feature_mean_norm_ratio <= 1.5,
        "class_macro_feature_first_order_delta_positive": (
            class_macro_feature_first_order_delta > 0.0
        ),
        "heldout_accuracy_gain_vs_clip_at_least_1pp": (
            heldout_accuracy_gain_vs_clip_pp >= 1.0
        ),
        "heldout_accuracy_gain_ci_lower_positive": heldout_accuracy_ci_lower_pp > 0.0,
        "car_accuracy_regression_at_most_0_5pp": car_accuracy_delta_pp >= -0.5,
        "truck_accuracy_regression_at_most_0_5pp": truck_accuracy_delta_pp >= -0.5,
    }
    passed = all(checks.values())
    return {
        "decision": ("NEEDS_EXACT_PARAMETER_AUDIT" if passed else "REJECT"),
        "checks": checks,
        "exact_parameter_audit_authorized": passed,
        "proxy_authorized": False,
        "training_authorized": False,
    }
