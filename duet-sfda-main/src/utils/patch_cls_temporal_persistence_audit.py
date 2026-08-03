"""Utilities for the frozen patch-to-CLS temporal-memory audit."""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_frozen_patch_memory(
    current_task_prediction: np.ndarray,
    query_position: np.ndarray,
    selected: np.ndarray,
    initial_task_candidate: np.ndarray,
) -> dict[str, np.ndarray]:
    """Override current task top-1 only on the already locked patch cohort.

    The function contains no score threshold, class route, or label access.  A
    selected row whose current task prediction already equals the frozen
    candidate is retained and is not counted as an effective correction.
    """
    current = np.asarray(current_task_prediction, dtype=np.int64)
    position = np.asarray(query_position, dtype=np.int64)
    route = np.asarray(selected, dtype=bool)
    memory = np.asarray(initial_task_candidate, dtype=np.int64)
    if current.ndim != 1:
        raise ValueError("current task prediction must be one-dimensional")
    if position.ndim != 1 or route.shape != position.shape:
        raise ValueError("query positions and selected mask must align")
    if memory.shape != position.shape:
        raise ValueError("initial task candidates must align with query positions")
    if np.unique(position).size != position.size:
        raise ValueError("query positions must be unique")
    if np.any(position < 0) or np.any(position >= current.size):
        raise ValueError("query position is outside the current prediction array")
    if np.any(current < 0) or np.any(memory < 0):
        raise ValueError("predictions must be nonnegative")

    prediction = current.copy()
    selected_position = position[route]
    prediction[selected_position] = memory[route]
    effective = route & (current[position] != memory)
    return {
        "prediction": prediction,
        "selected": route.copy(),
        "effective_correction": effective,
    }


def evaluate_patch_temporal_persistence_gate(
    *,
    input_contract_valid: bool,
    exploratory_selector_pass_preserved: bool,
    heldout_selector_pass_preserved: bool,
    selected_coverage_pct: float,
    effective_corrections: int,
    selected_comparisons: dict[str, dict[str, Any]],
    effective_task_comparison: dict[str, Any],
    full_proxy_task_macro_gain_pp: float,
    car_delta_pp: float,
    truck_delta_pp: float,
    car_truck_mean_delta_pp: float,
    other_ten_mean_delta_pp: float,
    max_class_mass_shift_pp: float,
) -> dict[str, Any]:
    """Evaluate whether frozen patch evidence survives one adaptation cycle.

    This is deliberately a strict mechanism gate.  Passing authorizes only a
    pure-control snapshot confirmation because the available cycle-2 snapshot
    came from the first-cycle support-conditioned-CLIP run.
    """
    required = {
        "cycle2_task",
        "cycle2_clip",
        "cycle2_confidence",
        "cycle2_arithmetic",
        "cycle2_rms",
        "cycle2_mix",
    }
    if set(selected_comparisons) != required:
        raise ValueError("selected comparisons do not match the declared controls")
    best_name = max(
        selected_comparisons,
        key=lambda name: (
            float(selected_comparisons[name]["baseline_accuracy_pct"]),
            name,
        ),
    )
    best = selected_comparisons[best_name]
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "exploratory_selector_pass_preserved": bool(
            exploratory_selector_pass_preserved
        ),
        "disjoint_heldout_selector_pass_preserved": bool(
            heldout_selector_pass_preserved
        ),
        "selected_coverage_between_2_and_10pct": (
            2.0 <= float(selected_coverage_pct) <= 10.0
        ),
        "at_least_20_effective_cycle2_corrections": int(effective_corrections) >= 20,
        "memory_gain_vs_best_selected_baseline_at_least_1pp": (
            float(best["gain_pp"]) >= 1.0
        ),
        "memory_gain_vs_best_selected_baseline_ci_lower_positive": (
            float(best["paired_bootstrap_95_ci_pp"][0]) > 0.0
        ),
        "memory_beats_every_matched_selected_baseline": all(
            float(value["gain_pp"]) > 0.0
            for value in selected_comparisons.values()
        ),
        "effective_gain_vs_cycle2_task_ci_lower_positive": (
            float(effective_task_comparison["paired_bootstrap_95_ci_pp"][0]) > 0.0
        ),
        "full_proxy_task_macro_gain_at_least_0_20pp": (
            float(full_proxy_task_macro_gain_pp) >= 0.20
        ),
        "car_regression_at_most_0_5pp": float(car_delta_pp) >= -0.5,
        "truck_regression_at_most_0_5pp": float(truck_delta_pp) >= -0.5,
        "car_truck_mean_nonnegative": float(car_truck_mean_delta_pp) >= 0.0,
        "other_ten_mean_nonnegative": float(other_ten_mean_delta_pp) >= 0.0,
        "max_class_mass_shift_at_most_1pp": (
            float(max_class_mass_shift_pp) <= 1.0
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "PASS_EXPLORATORY_PATCH_TEMPORAL_PERSISTENCE"
            if passed
            else "REJECT"
        ),
        "checks": checks,
        "best_selected_baseline": best_name,
        "pure_duet_cycle2_snapshot_confirmation_authorized": passed,
        "proxy_training_authorized": False,
        "full_training_authorized": False,
    }
