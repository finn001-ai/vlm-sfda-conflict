"""Label-free helpers for auditing cross-cycle mutual task/CLIP rise."""

from __future__ import annotations

from typing import Any

import numpy as np


MATCHED_BASELINES = {
    "fixed_task",
    "fixed_clip",
    "confidence_choice",
    "arithmetic",
    "rms",
}


def _probability_matrix(probability: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] < 2:
        raise ValueError(f"{name} must have shape [sample, class]")
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    row_sum = value.sum(axis=1)
    if np.any(row_sum <= 0.0) or not np.allclose(
        row_sum, 1.0, atol=1e-5, rtol=1e-5
    ):
        raise ValueError(f"{name} rows must sum to one")
    return value / row_sum[:, None]


def centered_log_velocity(
    previous_probability: np.ndarray,
    current_probability: np.ndarray,
) -> np.ndarray:
    """Return the cross-cycle change in every within-model log odds.

    Centering log probabilities removes the arbitrary row-wise logit constant.
    The resulting difference is therefore a signed change in relative class
    support, not a comparison of task and CLIP calibration scales.
    """
    previous = _probability_matrix(previous_probability, name="previous_probability")
    current = _probability_matrix(current_probability, name="current_probability")
    if previous.shape != current.shape:
        raise ValueError("previous and current probabilities must match")
    floor = float(np.finfo(np.float32).tiny)
    previous_log = np.log(np.maximum(previous, floor))
    current_log = np.log(np.maximum(current, floor))
    previous_centered = previous_log - previous_log.mean(axis=1, keepdims=True)
    current_centered = current_log - current_log.mean(axis=1, keepdims=True)
    velocity = current_centered - previous_centered
    if not np.isfinite(velocity).all():
        raise RuntimeError("centered-log velocity must be finite")
    return velocity


def route_mutual_rise(
    previous_task_probability: np.ndarray,
    current_task_probability: np.ndarray,
    previous_clip_probability: np.ndarray,
    current_clip_probability: np.ndarray,
    candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Route only a candidate whose relative support rose in both models.

    The current CLIP top-1 is the fallback.  Within the fixed candidate set,
    the candidate maximizing the weaker of the two signed velocities is chosen.
    Its decision is used only when both velocities are strictly positive and it
    differs from CLIP.  The soft target swaps only the selected and CLIP-top1
    probabilities, preserving row mass and all other CLIP probabilities.
    """
    previous_task = _probability_matrix(
        previous_task_probability, name="previous_task_probability"
    )
    current_task = _probability_matrix(
        current_task_probability, name="current_task_probability"
    )
    previous_clip = _probability_matrix(
        previous_clip_probability, name="previous_clip_probability"
    )
    current_clip = _probability_matrix(
        current_clip_probability, name="current_clip_probability"
    )
    if not (
        previous_task.shape
        == current_task.shape
        == previous_clip.shape
        == current_clip.shape
    ):
        raise ValueError("all probability matrices must have matching shapes")

    candidate = np.asarray(candidates, dtype=np.int64)
    if candidate.ndim != 2 or candidate.shape[0] != current_task.shape[0]:
        raise ValueError("candidates must align with probability rows")
    valid = (candidate >= 0) & (candidate < current_task.shape[1])
    if not np.all(valid.any(axis=1)):
        raise ValueError("every row must contain a valid candidate")

    task_velocity = centered_log_velocity(previous_task, current_task)
    clip_velocity = centered_log_velocity(previous_clip, current_clip)
    mutual_score = np.minimum(task_velocity, clip_velocity)
    safe_candidate = np.maximum(candidate, 0)
    candidate_score = np.take_along_axis(mutual_score, safe_candidate, axis=1)
    candidate_score = np.where(valid, candidate_score, -np.inf)
    selected_slot = candidate_score.argmax(axis=1)
    row = np.arange(candidate.shape[0])
    selected_class = candidate[row, selected_slot]
    selected_task_velocity = task_velocity[row, selected_class]
    selected_clip_velocity = clip_velocity[row, selected_class]
    fallback = current_clip.argmax(axis=1)
    fallback_in_set = (candidate == fallback[:, None]).any(axis=1)
    if not np.all(fallback_in_set):
        raise RuntimeError("current CLIP top-1 must be inside every candidate set")

    routed = (
        (selected_class != fallback)
        & (selected_task_velocity > 0.0)
        & (selected_clip_velocity > 0.0)
    )
    prediction = fallback.copy()
    prediction[routed] = selected_class[routed]
    target = current_clip.copy()
    routed_row = row[routed]
    routed_selected = selected_class[routed]
    routed_fallback = fallback[routed]
    selected_mass = target[routed_row, routed_selected].copy()
    fallback_mass = target[routed_row, routed_fallback].copy()
    target[routed_row, routed_selected] = fallback_mass
    target[routed_row, routed_fallback] = selected_mass
    if not np.allclose(target.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("probability swap changed target row mass")
    if not np.array_equal(target.argmax(axis=1), prediction):
        raise RuntimeError("swapped soft-target top-1 does not match routing")

    fallback_score = mutual_score[row, fallback]
    return {
        "prediction": prediction,
        "target_probability": target,
        "routed": routed,
        "selected_class": selected_class,
        "selected_slot": selected_slot,
        "selected_task_velocity": selected_task_velocity,
        "selected_clip_velocity": selected_clip_velocity,
        "selected_mutual_score": np.minimum(
            selected_task_velocity, selected_clip_velocity
        ),
        "fallback_mutual_score": fallback_score,
        "candidate_score": candidate_score,
        "task_velocity": task_velocity,
        "clip_velocity": clip_velocity,
    }


def evaluate_temporal_mutual_rise_gate(
    *,
    input_contract_valid: bool,
    route_coverage_pct: float,
    routed_union_decision_stability_pct: float,
    candidate_set_coverage_pct: float,
    minimum_class_candidate_coverage_pct: float,
    comparisons: dict[str, dict[str, Any]],
    best_baseline_name: str,
    car_delta_pp: float,
    truck_delta_pp: float,
    car_truck_mean_delta_pp: float,
    other_ten_mean_delta_pp: float,
    max_full_target_mass_shift_pp: float,
) -> dict[str, Any]:
    """Apply the fixed no-training gate for mutual-rise routing."""
    if set(comparisons) != MATCHED_BASELINES:
        raise ValueError("matched comparator contract is incomplete")
    if best_baseline_name not in MATCHED_BASELINES:
        raise ValueError("best baseline is not a matched comparator")
    numeric = (
        route_coverage_pct,
        routed_union_decision_stability_pct,
        candidate_set_coverage_pct,
        minimum_class_candidate_coverage_pct,
        car_delta_pp,
        truck_delta_pp,
        car_truck_mean_delta_pp,
        other_ten_mean_delta_pp,
        max_full_target_mass_shift_pp,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate metrics must be finite")
    for name, comparison in comparisons.items():
        interval = comparison.get("paired_bootstrap_95_ci_pp")
        values = (comparison.get("gain_pp"), *(interval or ()))
        if len(values) != 3 or not all(np.isfinite(value) for value in values):
            raise ValueError(f"invalid comparison: {name}")

    best = comparisons[best_baseline_name]
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "route_coverage_between_5_and_50pct": 5.0 <= route_coverage_pct <= 50.0,
        "routed_union_decision_stability_at_least_90pct": (
            routed_union_decision_stability_pct >= 90.0
        ),
        "top2_union_oracle_coverage_at_least_90pct": (
            candidate_set_coverage_pct >= 90.0
        ),
        "every_class_candidate_coverage_at_least_85pct": (
            minimum_class_candidate_coverage_pct >= 85.0
        ),
        "accuracy_gain_vs_best_baseline_at_least_1pp": best["gain_pp"] >= 1.0,
        "accuracy_gain_vs_best_baseline_ci_lower_positive": (
            best["paired_bootstrap_95_ci_pp"][0] > 0.0
        ),
        "beats_every_matched_baseline": all(
            comparison["gain_pp"] > 0.0 for comparison in comparisons.values()
        ),
        "car_regression_at_most_0_5pp": car_delta_pp >= -0.5,
        "truck_regression_at_most_0_5pp": truck_delta_pp >= -0.5,
        "car_truck_mean_nonnegative": car_truck_mean_delta_pp >= 0.0,
        "other_ten_mean_nonnegative": other_ten_mean_delta_pp >= 0.0,
        "max_full_target_mass_shift_at_most_1pp": (
            max_full_target_mass_shift_pp <= 1.0
        ),
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_TEMPORAL_MUTUAL_RISE_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "route_coverage_pct": [5.0, 50.0],
            "min_routed_union_decision_stability_pct": 90.0,
            "min_candidate_set_coverage_pct": 90.0,
            "min_per_class_candidate_set_coverage_pct": 85.0,
            "min_accuracy_gain_vs_best_baseline_pp": 1.0,
            "best_baseline_paired_ci_lower": "> 0",
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
            "max_full_target_mass_shift_pp": 1.0,
        },
        "training_authorized": False,
        "proxy_authorized": False,
        "gpu_authorized": False,
    }
