"""Helpers for label-free support conditioning of DUET's CLIP KL target."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_probability_matrix(
    probability: np.ndarray, *, name: str = "probability"
) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(f"{name} must have shape [sample, class]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    row_sum = values.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError(f"{name} rows must sum to one")
    return values / row_sum[:, None]


def support_conditioned_probability(
    probability: np.ndarray,
    support_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Condition each probability row on a fixed non-empty support set."""
    values = normalize_probability_matrix(probability)
    mask = np.asarray(support_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("support mask and probability shapes must match")
    if not np.all(mask.any(axis=1)):
        raise ValueError("every support row must be non-empty")
    retained_mass = np.where(mask, values, 0.0).sum(axis=1)
    if not np.isfinite(retained_mass).all() or np.any(retained_mass <= 0.0):
        raise ValueError("support must retain positive probability mass")
    conditioned = np.where(mask, values / retained_mass[:, None], 0.0)
    if not np.allclose(conditioned.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("conditioned probability rows must sum to one")
    return {"probability": conditioned, "retained_mass": retained_mass}


def probability_entropy(probability: np.ndarray) -> np.ndarray:
    values = normalize_probability_matrix(probability)
    positive = values > 0.0
    terms = np.zeros_like(values)
    terms[positive] = values[positive] * np.log(values[positive])
    return -terms.sum(axis=1)


def full_target_class_mass_shift_pp(
    candidate_probability: np.ndarray,
    baseline_probability: np.ndarray,
    *,
    full_target_samples: int,
) -> np.ndarray:
    """Return class-mass shift in pp when only these conflict rows change."""
    candidate = normalize_probability_matrix(
        candidate_probability, name="candidate_probability"
    )
    baseline = normalize_probability_matrix(
        baseline_probability, name="baseline_probability"
    )
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline probability shapes must match")
    if full_target_samples < candidate.shape[0]:
        raise ValueError("full_target_samples must cover all candidate rows")
    return (candidate - baseline).sum(axis=0) / float(full_target_samples) * 100.0


def negative_first_order_burden(first_order: np.ndarray) -> float:
    values = np.asarray(first_order, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("first_order must be a finite non-empty 1-D array")
    return float(np.minimum(values, 0.0).mean())


def evaluate_support_conditioned_clip_gate(
    *,
    input_contract_valid: bool,
    versus_clip: dict[str, dict[str, Any]],
    versus_top1_union: dict[str, dict[str, Any]],
    minimum_class_first_order_delta_vs_clip: float,
    candidate_negative_burden: float,
    clip_negative_burden: float,
    top1_union_negative_burden: float,
    clip_top2_negative_burden: float,
    candidate_max_full_mass_shift_pp: float,
    top1_union_max_full_mass_shift_pp: float,
    clip_top2_max_full_mass_shift_pp: float,
    candidate_top1_matches_clip: bool,
    max_allowed_full_mass_shift_pp: float = 1.0,
) -> dict[str, Any]:
    """Apply a predeclared oracle gate; passing authorizes no training."""
    metrics = ("cosine", "oracle_unit_projection", "first_order")
    for comparison_name, comparison in (
        ("versus_clip", versus_clip),
        ("versus_top1_union", versus_top1_union),
    ):
        for metric in metrics:
            result = comparison.get(metric, {})
            mean = result.get("mean_difference")
            interval = result.get("paired_bootstrap_95_ci")
            if (
                not isinstance(interval, (list, tuple))
                or len(interval) != 2
                or not all(np.isfinite(value) for value in (mean, *interval))
            ):
                raise ValueError(f"invalid {comparison_name} {metric} result")
    numeric = (
        minimum_class_first_order_delta_vs_clip,
        candidate_negative_burden,
        clip_negative_burden,
        top1_union_negative_burden,
        clip_top2_negative_burden,
        candidate_max_full_mass_shift_pp,
        top1_union_max_full_mass_shift_pp,
        clip_top2_max_full_mass_shift_pp,
        max_allowed_full_mass_shift_pp,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate values must be finite")

    def positive_with_ci(comparison: dict[str, Any], metric: str) -> bool:
        result = comparison[metric]
        return bool(
            result["mean_difference"] > 0.0
            and result["paired_bootstrap_95_ci"][0] > 0.0
        )

    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "candidate_top1_matches_clip": bool(candidate_top1_matches_clip),
        "candidate_cosine_beats_clip_with_positive_ci": positive_with_ci(
            versus_clip, "cosine"
        ),
        "candidate_projection_beats_clip_with_positive_ci": positive_with_ci(
            versus_clip, "oracle_unit_projection"
        ),
        "candidate_first_order_beats_clip_with_positive_ci": positive_with_ci(
            versus_clip, "first_order"
        ),
        "candidate_cosine_beats_top1_union_with_positive_ci": positive_with_ci(
            versus_top1_union, "cosine"
        ),
        "every_class_first_order_delta_vs_clip_nonnegative": (
            minimum_class_first_order_delta_vs_clip >= 0.0
        ),
        "candidate_negative_burden_not_worse_than_clip": (
            candidate_negative_burden >= clip_negative_burden
        ),
        "candidate_negative_burden_not_worse_than_top1_union": (
            candidate_negative_burden >= top1_union_negative_burden
        ),
        "candidate_negative_burden_better_than_clip_top2": (
            candidate_negative_burden > clip_top2_negative_burden
        ),
        "candidate_full_mass_shift_at_most_1pp": (
            candidate_max_full_mass_shift_pp <= max_allowed_full_mass_shift_pp
        ),
        "candidate_mass_shift_below_top1_union": (
            candidate_max_full_mass_shift_pp < top1_union_max_full_mass_shift_pp
        ),
        "candidate_mass_shift_below_clip_top2": (
            candidate_max_full_mass_shift_pp < clip_top2_max_full_mass_shift_pp
        ),
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_SUPPORT_CONDITIONED_CLIP_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "candidate_minus_clip_primary_means": "> 0",
            "candidate_minus_clip_primary_ci_lower": "> 0",
            "candidate_minus_top1_union_cosine_ci_lower": "> 0",
            "minimum_class_first_order_delta_vs_clip": ">= 0",
            "negative_first_order_burden": "not worse than CLIP/top1; better than CLIP-top2",
            "max_full_target_equivalent_class_mass_shift_pp": (
                max_allowed_full_mass_shift_pp
            ),
            "candidate_mass_shift": "below both sharpening controls",
        },
        "training_authorized": False,
    }
