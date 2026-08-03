"""Label-free risk controls for the locked patch-to-CLS rescue signal."""

from __future__ import annotations

from typing import Any

import numpy as np


def select_upper_median_mass_capped_rescues(
    task_candidate: np.ndarray,
    clip_candidate: np.ndarray,
    full_task_margin: np.ndarray,
    stable_task_rescue: np.ndarray,
    *,
    full_sample_count: int,
    max_class_mass_shift_fraction: float,
    class_count: int,
) -> dict[str, Any]:
    """Select stable upper-median rescues under a pseudo-class mass cap.

    The median is computed only from the already locked stable positive
    contribution margins.  Candidates are visited from largest to smallest
    margin and accepted only if the resulting task-minus-CLIP pseudo-class
    count shift stays within the declared full-sample fraction for every
    class.  Neither target labels nor class-specific routing rules enter.
    """
    task = np.asarray(task_candidate, dtype=np.int64)
    clip = np.asarray(clip_candidate, dtype=np.int64)
    margin = np.asarray(full_task_margin, dtype=np.float64)
    stable = np.asarray(stable_task_rescue, dtype=bool)
    if task.ndim != 1 or clip.shape != task.shape:
        raise ValueError("candidate vectors must be aligned and one-dimensional")
    if margin.shape != task.shape or stable.shape != task.shape:
        raise ValueError("margin and stable mask must align with candidates")
    if task.size == 0 or not stable.any():
        raise ValueError("at least one stable contribution rescue is required")
    if np.any(task == clip):
        raise ValueError("task and CLIP candidates must conflict")
    if not np.isfinite(margin).all():
        raise ValueError("contribution margins must be finite")
    if class_count < 2:
        raise ValueError("class_count must be at least two")
    if np.any(task < 0) or np.any(task >= class_count):
        raise ValueError("task candidate outside class range")
    if np.any(clip < 0) or np.any(clip >= class_count):
        raise ValueError("CLIP candidate outside class range")
    if full_sample_count <= 0:
        raise ValueError("full_sample_count must be positive")
    if not 0.0 < max_class_mass_shift_fraction < 1.0:
        raise ValueError("mass-shift fraction must be in (0, 1)")
    if np.any(margin[stable] <= 0.0):
        raise ValueError("stable rescue margins must be strictly positive")

    threshold = float(np.median(margin[stable]))
    upper_median = stable & (margin >= threshold)
    order = np.flatnonzero(upper_median)
    order = order[np.argsort(-margin[order], kind="stable")]
    count_cap = int(np.floor(full_sample_count * max_class_mass_shift_fraction))
    if count_cap < 1:
        raise ValueError("mass-shift fraction produces an empty count cap")

    selected = np.zeros(task.size, dtype=bool)
    class_count_shift = np.zeros(class_count, dtype=np.int64)
    rejected_by_mass_cap = np.zeros(task.size, dtype=bool)
    for row in order:
        proposed = class_count_shift.copy()
        proposed[task[row]] += 1
        proposed[clip[row]] -= 1
        if np.abs(proposed).max() <= count_cap:
            selected[row] = True
            class_count_shift = proposed
        else:
            rejected_by_mass_cap[row] = True

    prediction = clip.copy()
    prediction[selected] = task[selected]
    class_mass_shift_fraction = class_count_shift / float(full_sample_count)
    if not np.array_equal(
        class_count_shift,
        np.bincount(task[selected], minlength=class_count)
        - np.bincount(clip[selected], minlength=class_count),
    ):
        raise RuntimeError("incremental class-mass accounting is inconsistent")
    if np.abs(class_mass_shift_fraction).max() > max_class_mass_shift_fraction + 1e-12:
        raise RuntimeError("class-mass cap was violated")
    return {
        "prediction": prediction,
        "selected": selected,
        "upper_median": upper_median,
        "rejected_by_mass_cap": rejected_by_mass_cap,
        "threshold": threshold,
        "count_cap": count_cap,
        "class_count_shift": class_count_shift,
        "class_mass_shift_fraction": class_mass_shift_fraction,
    }


def evaluate_patch_cls_risk_control_gate(
    *,
    input_contract_valid: bool,
    source_reject_preserved: bool,
    selected_coverage_pct: float,
    paired_adjudication_precision_pct: float,
    comparisons: dict[str, dict[str, Any]],
    full_proxy_macro_gain_pp: float,
    car_delta_pp: float,
    truck_delta_pp: float,
    car_truck_mean_delta_pp: float,
    other_ten_mean_delta_pp: float,
    max_class_mass_shift_pp: float,
) -> dict[str, Any]:
    """Evaluate the exploratory risk-control gate after the signal lock."""
    required = {"fixed_task", "fixed_clip", "confidence_choice", "arithmetic", "rms"}
    if set(comparisons) != required:
        raise ValueError("comparisons must contain the five matched selectors")
    best_name = max(
        comparisons,
        key=lambda name: (comparisons[name]["baseline_accuracy_pct"], name),
    )
    best = comparisons[best_name]
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "source_patch_contribution_reject_preserved": bool(source_reject_preserved),
        "selected_coverage_between_2_and_10pct": (
            2.0 <= float(selected_coverage_pct) <= 10.0
        ),
        "paired_adjudication_precision_at_least_60pct": (
            float(paired_adjudication_precision_pct) >= 60.0
        ),
        "accuracy_gain_vs_best_baseline_at_least_1pp": (float(best["gain_pp"]) >= 1.0),
        "accuracy_gain_vs_best_baseline_ci_lower_positive": (
            float(best["paired_bootstrap_95_ci_pp"][0]) > 0.0
        ),
        "beats_every_matched_baseline": all(
            float(value["gain_pp"]) > 0.0 for value in comparisons.values()
        ),
        "full_proxy_macro_gain_at_least_0_20pp": (
            float(full_proxy_macro_gain_pp) >= 0.20
        ),
        "car_regression_at_most_0_5pp": float(car_delta_pp) >= -0.5,
        "truck_regression_at_most_0_5pp": float(truck_delta_pp) >= -0.5,
        "car_truck_mean_nonnegative": float(car_truck_mean_delta_pp) >= 0.0,
        "other_ten_mean_nonnegative": float(other_ten_mean_delta_pp) >= 0.0,
        "max_class_mass_shift_at_most_1pp": float(max_class_mass_shift_pp) <= 1.0,
    }
    passed = all(checks.values())
    return {
        "decision": ("PASS_EXPLORATORY_PATCH_CLS_RISK_CONTROL" if passed else "REJECT"),
        "checks": checks,
        "best_baseline_name": best_name,
        "heldout_full_audit_authorized": passed,
        "parameter_audit_authorized": False,
        "proxy_authorized": False,
        "training_authorized": False,
    }


def evaluate_patch_cls_holdout_gate(
    *,
    input_contract_valid: bool,
    exploratory_pass_preserved: bool,
    heldout_is_disjoint: bool,
    selected_coverage_pct: float,
    paired_adjudication_precision_pct: float,
    comparisons: dict[str, dict[str, Any]],
    heldout_macro_gain_pp: float,
    car_delta_pp: float,
    truck_delta_pp: float,
    car_truck_mean_delta_pp: float,
    other_ten_mean_delta_pp: float,
    max_class_mass_shift_pp: float,
) -> dict[str, Any]:
    """Evaluate the frozen rule on the disjoint full-target complement.

    The primary matched control is fixed CLIP because it was the strongest
    conflict selector in the exploratory proxy.  Fixed task and the simple
    confidence selector remain explicit comparators, but cannot replace the
    predeclared fixed-CLIP gate.
    """
    required = {"fixed_task", "fixed_clip", "confidence_choice"}
    if set(comparisons) != required:
        raise ValueError("comparisons must contain the three fixed selectors")
    fixed_clip = comparisons["fixed_clip"]
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "exploratory_same_proxy_pass_preserved": bool(exploratory_pass_preserved),
        "heldout_paths_disjoint_from_proxy25": bool(heldout_is_disjoint),
        "selected_coverage_between_2_and_10pct": (
            2.0 <= float(selected_coverage_pct) <= 10.0
        ),
        "paired_adjudication_precision_at_least_60pct": (
            float(paired_adjudication_precision_pct) >= 60.0
        ),
        "gain_vs_fixed_clip_at_least_1pp": float(fixed_clip["gain_pp"]) >= 1.0,
        "gain_vs_fixed_clip_ci_lower_positive": (
            float(fixed_clip["paired_bootstrap_95_ci_pp"][0]) > 0.0
        ),
        "beats_fixed_task_clip_and_confidence": all(
            float(value["gain_pp"]) > 0.0 for value in comparisons.values()
        ),
        "heldout_macro_gain_at_least_0_20pp": float(heldout_macro_gain_pp) >= 0.20,
        "car_regression_at_most_0_5pp": float(car_delta_pp) >= -0.5,
        "truck_regression_at_most_0_5pp": float(truck_delta_pp) >= -0.5,
        "car_truck_mean_nonnegative": float(car_truck_mean_delta_pp) >= 0.0,
        "other_ten_mean_nonnegative": float(other_ten_mean_delta_pp) >= 0.0,
        "max_class_mass_shift_at_most_1pp": float(max_class_mass_shift_pp) <= 1.0,
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_HELDOUT_PATCH_CLS_RISK_CONTROL" if passed else "REJECT",
        "checks": checks,
        "primary_matched_control": "fixed_clip",
        "parameter_audit_authorized": passed,
        "proxy_authorized": False,
        "full_training_authorized": False,
    }
