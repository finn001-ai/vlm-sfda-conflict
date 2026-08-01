"""Pure analysis helpers for the DUET feature-gravity offline audit."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


HARD_CLASSES = ("car", "person", "truck")


def _one_dimensional_pair(
    scores: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    if scores.ndim != 1 or target.ndim != 1 or scores.shape != target.shape:
        raise ValueError("scores and target must be same-shaped 1-D arrays")
    if scores.size == 0:
        raise ValueError("scores and target must not be empty")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if target.all() or (~target).all():
        raise ValueError("binary AUROC requires both positive and negative samples")
    return scores, target


def binary_auroc(scores: np.ndarray, target: np.ndarray) -> float:
    """Compute tie-aware binary AUROC without a scikit-learn dependency."""
    scores, target = _one_dimensional_pair(scores, target)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based; tied values receive their average rank.
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end

    positive_count = int(target.sum())
    negative_count = int((~target).sum())
    positive_rank_sum = float(ranks[target].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def stratified_bootstrap_auc_difference(
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    target: np.ndarray,
    *,
    repeats: int = 1_000,
    seed: int = 2_020,
) -> tuple[float, float]:
    """Return a stratified-bootstrap 95% CI for candidate-minus-baseline AUROC."""
    candidate_scores, target = _one_dimensional_pair(candidate_scores, target)
    baseline_scores, baseline_target = _one_dimensional_pair(baseline_scores, target)
    if not np.array_equal(target, baseline_target):
        raise ValueError("candidate and baseline targets differ")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    positive = np.flatnonzero(target)
    negative = np.flatnonzero(~target)
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = np.concatenate(
            (
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            )
        )
        sampled_target = target[sampled]
        differences[repeat] = binary_auroc(
            candidate_scores[sampled], sampled_target
        ) - binary_auroc(baseline_scores[sampled], sampled_target)
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def fixed_tail_masks(
    scores: np.ndarray, *, fraction: float = 0.20
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic bottom/top masks fixed without using labels."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not 0.0 < fraction <= 0.5:
        raise ValueError("fraction must be in (0, 0.5]")
    count = max(1, int(np.ceil(scores.size * fraction)))
    if 2 * count > scores.size:
        raise ValueError("tail masks would overlap")
    order = np.argsort(scores, kind="mergesort")
    bottom = np.zeros(scores.size, dtype=bool)
    top = np.zeros(scores.size, dtype=bool)
    bottom[order[:count]] = True
    top[order[-count:]] = True
    return bottom, top


def gradient_projection_summary(
    descent: np.ndarray, oracle_direction: np.ndarray
) -> dict[str, Any]:
    """Project a label-free descent vector onto an oracle CE direction.

    Labels are used only to construct ``oracle_direction`` outside this helper.
    Positive projection is first-order helpful; negative projection is harmful.
    """
    descent = np.asarray(descent, dtype=np.float64)
    oracle_direction = np.asarray(oracle_direction, dtype=np.float64)
    if descent.ndim != 2 or descent.shape != oracle_direction.shape:
        raise ValueError("descent and oracle_direction must be same-shaped 2-D arrays")
    if descent.shape[0] == 0:
        raise ValueError("gradient arrays must not be empty")
    if not np.isfinite(descent).all() or not np.isfinite(oracle_direction).all():
        raise ValueError("gradient arrays must be finite")

    oracle_norm = np.linalg.norm(oracle_direction, axis=1)
    if np.any(oracle_norm <= 0.0):
        raise ValueError("oracle directions must have non-zero norm")
    oracle_unit = oracle_direction / oracle_norm[:, None]
    projection = np.einsum("ij,ij->i", descent, oracle_unit)
    helpful = np.maximum(projection, 0.0)
    harmful = np.maximum(-projection, 0.0)
    return {
        "projection": projection,
        "helpful_mass": float(helpful.sum()),
        "harmful_mass": float(harmful.sum()),
        "helpful_samples": int((projection > 0.0).sum()),
        "harmful_samples": int((projection < 0.0).sum()),
        "zero_samples": int((projection == 0.0).sum()),
        "mean_projection": float(projection.mean()),
    }


def classwise_gradient_mass(
    labels: np.ndarray,
    class_names: list[str],
    current_projection: np.ndarray,
    candidate_projection: np.ndarray,
) -> list[dict[str, Any]]:
    """Summarize oracle helpful/harmful projection mass by true class."""
    labels = np.asarray(labels, dtype=np.int64)
    current_projection = np.asarray(current_projection, dtype=np.float64)
    candidate_projection = np.asarray(candidate_projection, dtype=np.float64)
    if not (
        labels.ndim == current_projection.ndim == candidate_projection.ndim == 1
        and labels.shape == current_projection.shape == candidate_projection.shape
    ):
        raise ValueError("labels and projections must be same-shaped 1-D arrays")

    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        current_harmful = float(np.maximum(-current_projection[mask], 0.0).sum())
        candidate_harmful = float(np.maximum(-candidate_projection[mask], 0.0).sum())
        current_helpful = float(np.maximum(current_projection[mask], 0.0).sum())
        candidate_helpful = float(np.maximum(candidate_projection[mask], 0.0).sum())
        harmful_change = (
            100.0 * (candidate_harmful - current_harmful) / current_harmful
            if current_harmful > 0.0
            else (0.0 if candidate_harmful == 0.0 else None)
        )
        rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "samples": int(mask.sum()),
                "current_harmful_mass": current_harmful,
                "candidate_harmful_mass": candidate_harmful,
                "harmful_change_percent": harmful_change,
                "current_helpful_mass": current_helpful,
                "candidate_helpful_mass": candidate_helpful,
            }
        )
    return rows


def evaluate_preflight_gate(
    *,
    reproduction_passed: bool,
    auc_gain: float,
    auc_ci: tuple[float, float],
    quintile_accuracy_gap_pp: float,
    harmful_reduction_percent: float,
    helpful_retention_percent: float,
    classwise: Iterable[dict[str, Any]],
    min_auc_gain: float = 0.02,
    min_quintile_gap_pp: float = 5.0,
    min_harmful_reduction_percent: float = 10.0,
    min_helpful_retention_percent: float = 95.0,
) -> dict[str, Any]:
    """Apply the predeclared binary gate; no threshold is fitted to labels."""
    by_name = {str(row["class"]).strip().lower(): row for row in classwise}
    hard_class_checks = {
        name: (
            name in by_name
            and int(by_name[name]["samples"]) > 0
            and float(by_name[name]["candidate_harmful_mass"])
            <= float(by_name[name]["current_harmful_mass"]) + 1e-12
        )
        for name in HARD_CLASSES
    }
    checks = {
        "baseline_reproduced": bool(reproduction_passed),
        "feature_auc_gain_at_least_0.02": auc_gain >= min_auc_gain,
        "feature_auc_gain_ci_lower_positive": auc_ci[0] > 0.0,
        "top_bottom_quintile_gap_at_least_5pp": (
            quintile_accuracy_gap_pp >= min_quintile_gap_pp
        ),
        "harmful_gradient_mass_reduced_at_least_10pct": (
            harmful_reduction_percent >= min_harmful_reduction_percent
        ),
        "helpful_gradient_mass_retained_at_least_95pct": (
            helpful_retention_percent >= min_helpful_retention_percent
        ),
        **{f"{name}_harmful_mass_nonincreasing": passed for name, passed in hard_class_checks.items()},
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_OFFLINE_GATE" if passed else "REJECT",
        "thresholds": {
            "min_auc_gain": min_auc_gain,
            "auc_ci_lower_must_be_positive": True,
            "min_quintile_accuracy_gap_pp": min_quintile_gap_pp,
            "min_harmful_reduction_percent": min_harmful_reduction_percent,
            "min_helpful_retention_percent": min_helpful_retention_percent,
            "hard_classes": list(HARD_CLASSES),
            "hard_class_harmful_mass_must_be_nonincreasing": True,
        },
        "checks": checks,
    }
