"""Helpers for the cycle-2 initial-conflict-memory preflight."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.support_conditioned_clip_audit import (
    normalize_probability_matrix,
    support_conditioned_probability,
)


def stable_top2_union_mask(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
) -> np.ndarray:
    """Return the deterministic union of task and CLIP top-2 supports."""
    task = normalize_probability_matrix(task_probability, name="task_probability")
    clip = normalize_probability_matrix(clip_probability, name="clip_probability")
    if task.shape != clip.shape:
        raise ValueError("task and CLIP probabilities must have matching shapes")
    task_top2 = np.argsort(-task, axis=1, kind="stable")[:, :2]
    clip_top2 = np.argsort(-clip, axis=1, kind="stable")[:, :2]
    support = np.zeros_like(task, dtype=bool)
    row = np.arange(task.shape[0])[:, None]
    support[row, task_top2] = True
    support[row, clip_top2] = True
    return support


def build_cycle2_conflict_memory_target(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    """Condition cycle-2 CLIP mass using the current top-2 union."""
    task = normalize_probability_matrix(task_probability, name="task_probability")
    clip = normalize_probability_matrix(clip_probability, name="clip_probability")
    support = stable_top2_union_mask(task, clip)
    conditioned = support_conditioned_probability(clip, support)
    probability = conditioned["probability"]
    if not np.array_equal(probability.argmax(axis=1), clip.argmax(axis=1)):
        raise RuntimeError("cycle-2 conditioning changed the CLIP top-1 class")
    return {
        "probability": probability,
        "support": support,
        "retained_clip_mass": conditioned["retained_mass"],
        "support_size": support.sum(axis=1),
    }


def evaluate_cycle2_conflict_memory_gate(
    *,
    input_contract_valid: bool,
    candidate_top1_matches_clip: bool,
    overall_comparison: dict[str, dict[str, Any]],
    resolved_first_order: dict[str, Any],
    still_conflict_first_order: dict[str, Any],
    minimum_class_first_order_delta: float,
    candidate_negative_burden: float,
    clip_negative_burden: float,
    top2_union_oracle_coverage_pct: float,
    max_full_mass_shift_pp: float,
) -> dict[str, Any]:
    """Apply the predeclared cycle-2 gate; passing never starts training."""
    metrics = ("cosine", "oracle_unit_projection", "first_order")

    def positive_with_ci(result: dict[str, Any]) -> bool:
        interval = result.get("paired_bootstrap_95_ci")
        return bool(
            isinstance(interval, (list, tuple))
            and len(interval) == 2
            and np.isfinite(result.get("mean_difference", np.nan))
            and np.isfinite(interval).all()
            and result["mean_difference"] > 0.0
            and interval[0] > 0.0
        )

    for metric in metrics:
        if metric not in overall_comparison:
            raise ValueError(f"missing overall comparison metric: {metric}")
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "candidate_top1_matches_clip": bool(candidate_top1_matches_clip),
        "overall_cosine_gain_ci_lower_positive": positive_with_ci(
            overall_comparison["cosine"]
        ),
        "overall_projection_gain_ci_lower_positive": positive_with_ci(
            overall_comparison["oracle_unit_projection"]
        ),
        "overall_first_order_gain_ci_lower_positive": positive_with_ci(
            overall_comparison["first_order"]
        ),
        "resolved_first_order_gain_ci_lower_positive": positive_with_ci(
            resolved_first_order
        ),
        "still_conflict_first_order_gain_ci_lower_positive": positive_with_ci(
            still_conflict_first_order
        ),
        "every_class_first_order_delta_nonnegative": (
            minimum_class_first_order_delta >= 0.0
        ),
        "candidate_negative_burden_not_worse": (
            candidate_negative_burden >= clip_negative_burden
        ),
        "top2_union_oracle_coverage_at_least_90pct": (
            top2_union_oracle_coverage_pct >= 90.0
        ),
        "max_full_mass_shift_at_most_1pp": max_full_mass_shift_pp <= 1.0,
    }
    passed = all(checks.values())
    return {
        "decision": (
            "PASS_CYCLE2_CONFLICT_MEMORY_PREFLIGHT" if passed else "REJECT"
        ),
        "checks": checks,
        "thresholds": {
            "paired_mean_and_bootstrap_ci_lower": "> 0",
            "minimum_class_first_order_delta": ">= 0",
            "negative_first_order_burden": "not worse than original CLIP KL",
            "top2_union_oracle_coverage_pct": ">= 90",
            "max_full_target_equivalent_class_mass_shift_pp": "<= 1",
        },
        "training_authorized": False,
    }
