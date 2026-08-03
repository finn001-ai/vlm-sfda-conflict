"""Pure helpers for patch-selected pairwise CLIP-target neutralization."""

from __future__ import annotations

from typing import Any

import numpy as np


def neutralize_candidate_pair(
    clip_probability: np.ndarray,
    task_candidate: np.ndarray,
    clip_candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Equalize only the task/CLIP candidate mass in a CLIP distribution.

    All non-candidate class probabilities and the total mass assigned to the
    candidate pair are preserved exactly.  The returned second value is the
    signed mass transferred from the CLIP candidate to the task candidate.
    """
    probability = np.asarray(clip_probability, dtype=np.float64)
    task = np.asarray(task_candidate, dtype=np.int64)
    clip = np.asarray(clip_candidate, dtype=np.int64)
    if probability.ndim != 2 or probability.shape[0] == 0:
        raise ValueError("clip_probability must be a non-empty matrix")
    if task.shape != (probability.shape[0],) or clip.shape != task.shape:
        raise ValueError("candidate arrays must align with probability rows")
    if np.any(task < 0) or np.any(task >= probability.shape[1]):
        raise ValueError("task candidate outside class range")
    if np.any(clip < 0) or np.any(clip >= probability.shape[1]):
        raise ValueError("CLIP candidate outside class range")
    if np.any(task == clip):
        raise ValueError("candidate pair must contain two distinct classes")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("CLIP probabilities must be finite and non-negative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("CLIP probabilities must sum to one")

    row = np.arange(probability.shape[0])
    task_mass = probability[row, task]
    clip_mass = probability[row, clip]
    pair_mean = 0.5 * (task_mass + clip_mass)
    result = probability.copy()
    result[row, task] = pair_mean
    result[row, clip] = pair_mean
    transferred = pair_mean - task_mass
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("neutralized probabilities must preserve row mass")
    return result, transferred


def evaluate_patch_pair_neutralization_gate(
    *,
    input_contract_valid: bool,
    source_suppression_reject_preserved: bool,
    heldout_selector_passed: bool,
    selected_coverage_pct: float,
    target_replay_max_abs_error: float,
    nonpair_target_max_abs_error: float,
    pair_mass_max_abs_error: float,
    baseline_output_first_order: dict[str, Any],
    baseline_feature_first_order: dict[str, Any],
    suppression_output_first_order: dict[str, Any],
    suppression_feature_first_order: dict[str, Any],
    output_negative_burden_baseline: float,
    output_negative_burden_candidate: float,
    feature_negative_burden_baseline: float,
    feature_negative_burden_candidate: float,
    feature_helpful_retention_pct: float,
    feature_mean_norm_ratio: float,
    max_full_target_class_mass_shift_pp: float,
    class_macro_feature_first_order_delta: float,
    car_feature_first_order_delta: float,
    person_feature_first_order_delta: float,
    truck_feature_first_order_delta: float,
    other_nine_feature_first_order_delta: float,
) -> dict[str, Any]:
    """Gate one exact resident-parameter audit; never authorize training."""

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
        target_replay_max_abs_error,
        nonpair_target_max_abs_error,
        pair_mass_max_abs_error,
        output_negative_burden_baseline,
        output_negative_burden_candidate,
        feature_negative_burden_baseline,
        feature_negative_burden_candidate,
        feature_helpful_retention_pct,
        feature_mean_norm_ratio,
        max_full_target_class_mass_shift_pp,
        class_macro_feature_first_order_delta,
        car_feature_first_order_delta,
        person_feature_first_order_delta,
        truck_feature_first_order_delta,
        other_nine_feature_first_order_delta,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate inputs must be finite")
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "source_kl_suppression_reject_preserved": bool(
            source_suppression_reject_preserved
        ),
        "heldout_selector_gate_passed": bool(heldout_selector_passed),
        "selected_coverage_between_2_and_10pct": (
            2.0 <= float(selected_coverage_pct) <= 10.0
        ),
        "recovered_clip_target_error_at_most_5e_6": (
            float(target_replay_max_abs_error) <= 5e-6
        ),
        "nonpair_target_probability_unchanged": (
            float(nonpair_target_max_abs_error) <= 1e-12
        ),
        "candidate_pair_mass_preserved": float(pair_mass_max_abs_error) <= 1e-12,
        "output_first_order_gain_vs_duet_ci_lower_positive": positive_with_ci(
            baseline_output_first_order
        ),
        "feature_first_order_gain_vs_duet_ci_lower_positive": positive_with_ci(
            baseline_feature_first_order
        ),
        "output_first_order_gain_vs_suppression_ci_lower_positive": (
            positive_with_ci(suppression_output_first_order)
        ),
        "feature_first_order_gain_vs_suppression_ci_lower_positive": (
            positive_with_ci(suppression_feature_first_order)
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
        "max_full_target_class_mass_shift_at_most_1pp": (
            max_full_target_class_mass_shift_pp <= 1.0
        ),
        "class_macro_feature_first_order_delta_positive": (
            class_macro_feature_first_order_delta > 0.0
        ),
        "car_feature_first_order_delta_nonnegative": (
            car_feature_first_order_delta >= 0.0
        ),
        "person_feature_first_order_delta_nonnegative": (
            person_feature_first_order_delta >= 0.0
        ),
        "truck_feature_first_order_delta_nonnegative": (
            truck_feature_first_order_delta >= 0.0
        ),
        "other_nine_feature_first_order_delta_nonnegative": (
            other_nine_feature_first_order_delta >= 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "decision": "NEEDS_EXACT_PARAMETER_AUDIT" if passed else "REJECT",
        "checks": checks,
        "exact_parameter_audit_authorized": passed,
        "proxy_authorized": False,
        "training_authorized": False,
    }
