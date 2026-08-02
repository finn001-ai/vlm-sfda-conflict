"""Helpers for auditing shared runner-up evidence inside DUET agreements."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.candidate_set_audit import stable_topk


def shared_runner_up_candidate(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build a two-label set when task and CLIP share top-1 and runner-up."""
    task = np.asarray(task_probability, dtype=np.float64)
    clip = np.asarray(clip_probability, dtype=np.float64)
    if task.ndim != 2 or task.shape != clip.shape or task.shape[1] < 2:
        raise ValueError("task and CLIP probabilities must be same-shaped NxC matrices")
    if not np.isfinite(task).all() or not np.isfinite(clip).all():
        raise ValueError("probabilities must be finite")
    task_top2 = stable_topk(task, 2)
    clip_top2 = stable_topk(clip, 2)
    agreement = task_top2[:, 0] == clip_top2[:, 0]
    selected = agreement & (task_top2[:, 1] == clip_top2[:, 1])
    candidate_mask = np.zeros(task.shape, dtype=bool)
    row = np.arange(task.shape[0])
    candidate_mask[row, task_top2[:, 0]] = True
    candidate_mask[row[selected], task_top2[selected, 1]] = True
    return {
        "task_top2": task_top2,
        "clip_top2": clip_top2,
        "agreement": agreement,
        "selected": selected,
        "common_top1": task_top2[:, 0],
        "shared_runner_up": task_top2[:, 1],
        "candidate_mask": candidate_mask,
    }


def evaluate_shared_runner_up_gate(
    *,
    input_contract_valid: bool,
    selected_fraction_pct: float,
    selected_candidate_coverage_pct: float,
    selected_top1_miss_recovery_pct: float,
    delta_vs_top1_ci: tuple[float, float],
    delta_vs_zero_delay_ci: tuple[float, float],
    car_first_order_delta: float,
    person_first_order_delta: float,
    truck_first_order_delta: float,
    nonhard_first_order_delta: float,
    max_full_mass_shift_pp: float,
) -> dict[str, Any]:
    """Apply a predeclared offline gate; passing never starts training."""
    numeric = (
        selected_fraction_pct,
        selected_candidate_coverage_pct,
        selected_top1_miss_recovery_pct,
        *delta_vs_top1_ci,
        *delta_vs_zero_delay_ci,
        car_first_order_delta,
        person_first_order_delta,
        truck_first_order_delta,
        nonhard_first_order_delta,
        max_full_mass_shift_pp,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate values must be finite")
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "selected_coverage_between_5_and_80pct": (
            5.0 <= selected_fraction_pct <= 80.0
        ),
        "selected_candidate_coverage_at_least_98pct": (
            selected_candidate_coverage_pct >= 98.0
        ),
        "selected_top1_miss_recovery_at_least_50pct": (
            selected_top1_miss_recovery_pct >= 50.0
        ),
        "first_order_gain_vs_top1_ci_lower_positive": delta_vs_top1_ci[0] > 0.0,
        "first_order_gain_vs_zero_delay_ci_lower_positive": (
            delta_vs_zero_delay_ci[0] > 0.0
        ),
        "car_first_order_delta_nonnegative": car_first_order_delta >= 0.0,
        "person_first_order_delta_nonnegative": person_first_order_delta >= 0.0,
        "truck_first_order_delta_nonnegative": truck_first_order_delta >= 0.0,
        "other_nine_first_order_delta_nonnegative": nonhard_first_order_delta >= 0.0,
        "max_full_mass_shift_at_most_1pp": max_full_mass_shift_pp <= 1.0,
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_SHARED_RUNNER_UP_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "selected_fraction_pct": [5.0, 80.0],
            "min_selected_candidate_coverage_pct": 98.0,
            "min_selected_top1_miss_recovery_pct": 50.0,
            "paired_first_order_ci_lower_vs_top1_and_zero_delay": "> 0",
            "car_person_truck_other9_first_order_delta": ">= 0",
            "max_full_mass_shift_pp": 1.0,
        },
        "training_authorized": False,
    }
