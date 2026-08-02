"""Helpers for auditing reversible DUET agreement admission."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalized_mask_weight(mask: np.ndarray) -> np.ndarray:
    """Return non-negative population weights with mean one over all rows."""
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 1 or not selected.any():
        raise ValueError("mask must be a non-empty vector")
    weight = np.zeros(selected.size, dtype=np.float64)
    weight[selected] = selected.size / selected.sum()
    return weight


def evaluate_agreement_revocation_gate(
    *,
    input_contract_valid: bool,
    stale_fraction_of_admitted_pct: float,
    stale_error_enrichment: float,
    captured_error_gains: dict[str, int],
    precision_gain_cis: dict[str, tuple[float, float]],
    retained_accuracy_gain_pp: float,
    first_order_delta_vs_baseline_ci: tuple[float, float],
    first_order_delta_vs_confidence_cis: dict[str, tuple[float, float]],
    car_first_order_delta: float,
    person_first_order_delta: float,
    truck_first_order_delta: float,
    nonhard_first_order_delta: float,
    max_class_mass_shift_pp: float,
) -> dict[str, Any]:
    """Apply the predeclared CPU-only gate; passing never starts training."""
    required = {
        "task_confidence",
        "clip_confidence",
        "arithmetic_confidence",
        "rms_confidence",
    }
    if set(captured_error_gains) != required:
        raise ValueError("captured-error gains must contain four confidence baselines")
    if set(precision_gain_cis) != required:
        raise ValueError("precision intervals must contain four confidence baselines")
    if set(first_order_delta_vs_confidence_cis) != required:
        raise ValueError("first-order intervals must contain four confidence baselines")
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "stale_fraction_between_1_and_30pct": (
            1.0 <= stale_fraction_of_admitted_pct <= 30.0
        ),
        "stale_error_enrichment_at_least_2x": stale_error_enrichment >= 2.0,
        "beats_all_matched_confidence_error_captures": all(
            value > 0 for value in captured_error_gains.values()
        ),
        "all_precision_gain_ci_lowers_positive": all(
            interval[0] > 0.0 for interval in precision_gain_cis.values()
        ),
        "retained_accuracy_gain_at_least_0_25pp": retained_accuracy_gain_pp >= 0.25,
        "first_order_gain_vs_monotonic_mask_ci_lower_positive": (
            first_order_delta_vs_baseline_ci[0] > 0.0
        ),
        "first_order_gain_vs_all_confidence_revocations_ci_lower_positive": all(
            interval[0] > 0.0
            for interval in first_order_delta_vs_confidence_cis.values()
        ),
        "car_first_order_delta_nonnegative": car_first_order_delta >= 0.0,
        "person_first_order_delta_nonnegative": person_first_order_delta >= 0.0,
        "truck_first_order_delta_nonnegative": truck_first_order_delta >= 0.0,
        "other_nine_first_order_delta_nonnegative": nonhard_first_order_delta >= 0.0,
        "max_class_mass_shift_at_most_1pp": max_class_mass_shift_pp <= 1.0,
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_AGREEMENT_REVOCATION_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "stale_fraction_of_admitted_pct": [1.0, 30.0],
            "min_stale_error_enrichment": 2.0,
            "paired_precision_ci_lower_vs_all_confidence_baselines": "> 0",
            "min_retained_accuracy_gain_pp": 0.25,
            "paired_first_order_ci_lower_vs_monotonic_and_confidence": "> 0",
            "car_person_truck_other9_first_order_delta": ">= 0",
            "max_class_mass_shift_pp": 1.0,
        },
        "training_authorized": False,
    }
