"""Helpers for a label-free continuous confidence-weight audit.

The candidate keeps every cycle-1 DUET agreement and redistributes only the
hard pseudo-label cross-entropy weight.  Weights are normalized independently
inside each pseudo class, so the total hard-label mass of every class is
unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def class_mean_normalized_confidence_weight(
    confidence: np.ndarray,
    eligible: np.ndarray,
    group: np.ndarray,
) -> np.ndarray:
    """Return confidence/within-class-mean on eligible rows and zero elsewhere."""
    score = np.asarray(confidence, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    labels = np.asarray(group, dtype=np.int64)
    if score.ndim != 1 or mask.shape != score.shape or labels.shape != score.shape:
        raise ValueError("confidence, eligible, and group must be same-shaped vectors")
    if not np.isfinite(score).all() or np.any(score <= 0.0):
        raise ValueError("confidence must be finite and strictly positive")
    if np.any(labels < 0):
        raise ValueError("group indices must be non-negative")

    weight = np.zeros(score.size, dtype=np.float64)
    for class_index in np.unique(labels[mask]):
        indices = np.flatnonzero(mask & (labels == class_index))
        class_mean = float(score[indices].mean())
        if not np.isfinite(class_mean) or class_mean <= 0.0:
            raise ValueError("each eligible class must have positive mean confidence")
        weight[indices] = score[indices] / class_mean
    return weight


def class_balanced_bottom_fraction_reference_weight(
    confidence: np.ndarray,
    eligible: np.ndarray,
    group: np.ndarray,
    *,
    fraction: float,
) -> dict[str, Any]:
    """Population-level weight equivalent of a class-balanced bottom-fraction delay.

    Delayed rows receive zero weight.  Retained rows are rescaled within each
    pseudo class so that the class mean remains one.  This is a deterministic
    population reference, not an exact replay of minibatch normalization.
    """
    score = np.asarray(confidence, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    labels = np.asarray(group, dtype=np.int64)
    if score.ndim != 1 or mask.shape != score.shape or labels.shape != score.shape:
        raise ValueError("confidence, eligible, and group must be same-shaped vectors")
    if not np.isfinite(score).all():
        raise ValueError("confidence must be finite")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")

    delayed = np.zeros(score.size, dtype=bool)
    weight = np.zeros(score.size, dtype=np.float64)
    counts: dict[int, int] = {}
    for class_index in np.unique(labels[mask]):
        indices = np.flatnonzero(mask & (labels == class_index))
        count = max(1, int(np.ceil(indices.size * fraction)))
        order = np.lexsort((indices, score[indices]))
        delayed_indices = indices[order[:count]]
        retained_indices = indices[order[count:]]
        if retained_indices.size == 0:
            raise ValueError("bottom-fraction delay removed an entire pseudo class")
        delayed[delayed_indices] = True
        weight[retained_indices] = indices.size / retained_indices.size
        counts[int(class_index)] = int(count)
    return {"weight": weight, "delayed": delayed, "counts_by_group": counts}


def ce_logit_descent(probability: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Negative gradient of cross entropy with respect to logits."""
    prob = np.asarray(probability, dtype=np.float64)
    target = np.asarray(label, dtype=np.int64)
    if prob.ndim != 2 or target.shape != (prob.shape[0],):
        raise ValueError("probability must be NxC and label must be length N")
    if not np.isfinite(prob).all() or np.any(prob < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(prob.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probabilities must sum to one")
    if np.any(target < 0) or np.any(target >= prob.shape[1]):
        raise ValueError("label is outside the class range")
    descent = -prob.copy()
    descent[np.arange(prob.shape[0]), target] += 1.0
    return descent


def weighted_logit_alignment(
    pseudo_descent: np.ndarray,
    oracle_descent: np.ndarray,
    weight: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    """Summarize a weighted pseudo-label descent against an oracle direction."""
    pseudo = np.asarray(pseudo_descent, dtype=np.float64)
    oracle = np.asarray(oracle_descent, dtype=np.float64)
    sample_weight = np.asarray(weight, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if pseudo.shape != oracle.shape or pseudo.ndim != 2:
        raise ValueError("pseudo and oracle descents must be same-shaped matrices")
    if sample_weight.shape != (pseudo.shape[0],) or mask.shape != sample_weight.shape:
        raise ValueError("weight and eligible must match the number of rows")
    if not (
        np.isfinite(pseudo).all()
        and np.isfinite(oracle).all()
        and np.isfinite(sample_weight).all()
    ):
        raise ValueError("alignment arrays must be finite")
    if np.any(sample_weight < 0.0) or not np.any(mask):
        raise ValueError("weights must be non-negative and eligibility non-empty")

    weighted = pseudo * sample_weight[:, None]
    row_first_order = np.einsum("ij,ij->i", weighted, oracle)
    selected_score = row_first_order[mask]
    weighted_flat = weighted[mask].ravel()
    oracle_flat = oracle[mask].ravel()
    denominator = float(np.linalg.norm(weighted_flat) * np.linalg.norm(oracle_flat))
    cosine = float(np.dot(weighted_flat, oracle_flat) / denominator)
    selected_weight = sample_weight[mask]
    effective_size = float(
        np.square(selected_weight.sum()) / np.square(selected_weight).sum()
    )
    return {
        "row_first_order": row_first_order,
        "mean_first_order": float(selected_score.mean()),
        "aggregate_cosine": cosine,
        "negative_burden": float(np.maximum(-selected_score, 0.0).mean()),
        "positive_support": float(np.maximum(selected_score, 0.0).mean()),
        "effective_sample_size": effective_size,
        "effective_sample_size_pct": float(effective_size / mask.sum() * 100.0),
    }


def paired_mean_bootstrap_ci(
    candidate: np.ndarray,
    reference: np.ndarray,
    eligible: np.ndarray,
    *,
    repeats: int = 2_000,
    seed: int = 2_020,
    batch_size: int = 50,
) -> tuple[float, float]:
    """Paired bootstrap CI for a candidate-minus-reference row mean."""
    first = np.asarray(candidate, dtype=np.float64)
    second = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if first.shape != second.shape or mask.shape != first.shape or first.ndim != 1:
        raise ValueError("paired arrays and eligibility must be same-shaped vectors")
    difference = (first - second)[mask]
    if difference.size == 0 or not np.isfinite(difference).all():
        raise ValueError("paired difference must be finite and non-empty")
    if repeats <= 0 or batch_size <= 0:
        raise ValueError("bootstrap settings must be positive")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats, dtype=np.float64)
    written = 0
    while written < repeats:
        current = min(batch_size, repeats - written)
        indices = rng.integers(0, difference.size, size=(current, difference.size))
        bootstrap[written : written + current] = difference[indices].mean(axis=1)
        written += current
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def evaluate_agreement_confidence_weight_gate(
    *,
    input_contract_valid: bool,
    max_pseudo_class_mean_weight_error: float,
    effective_sample_size_pct: float,
    delta_vs_unweighted_ci: tuple[float, float],
    delta_vs_hard_delay_ci: tuple[float, float],
    candidate_negative_burden: float,
    baseline_negative_burden: float,
    candidate_positive_support: float,
    hard_delay_positive_support: float,
    car_first_order_delta: float,
    truck_first_order_delta: float,
    noncar_first_order_delta: float,
) -> dict[str, Any]:
    """Apply the predeclared CPU-only gate; passing never authorizes training."""
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "pseudo_class_mean_weight_preserved": (
            max_pseudo_class_mean_weight_error <= 1e-10
        ),
        "effective_sample_size_at_least_90pct": effective_sample_size_pct >= 90.0,
        "first_order_gain_vs_unweighted_ci_lower_positive": (
            delta_vs_unweighted_ci[0] > 0.0
        ),
        "first_order_gain_vs_hard_delay_ci_lower_positive": (
            delta_vs_hard_delay_ci[0] > 0.0
        ),
        "negative_burden_lower_than_unweighted": (
            candidate_negative_burden < baseline_negative_burden
        ),
        "positive_support_nonworse_than_hard_delay": (
            candidate_positive_support >= hard_delay_positive_support
        ),
        "car_first_order_delta_nonnegative": car_first_order_delta >= 0.0,
        "truck_first_order_delta_nonnegative": truck_first_order_delta >= 0.0,
        "noncar_first_order_delta_nonnegative": noncar_first_order_delta >= 0.0,
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_AGREEMENT_CONFIDENCE_WEIGHT_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "max_pseudo_class_mean_weight_error": 1e-10,
            "min_effective_sample_size_pct": 90.0,
            "paired_first_order_ci_lower_vs_both_references": "> 0",
            "car_truck_noncar_first_order_delta": ">= 0",
        },
        "training_authorized": False,
    }
