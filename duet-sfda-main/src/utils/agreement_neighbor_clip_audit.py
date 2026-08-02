"""Label-free helpers for cross-sample CLIP evidence in DUET conflicts."""

from __future__ import annotations

from typing import Any

import numpy as np


def _probability_matrix(probability: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(f"{name} must have shape [sample, class]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    row_sum = values.sum(axis=1)
    if np.any(row_sum <= 0.0) or not np.allclose(row_sum, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} rows must sum to one")
    return values / row_sum[:, None]


def agreement_neighbor_clip_posterior(
    task_feature: np.ndarray,
    clip_probability: np.ndarray,
    agreement_mask: np.ndarray,
    query_mask: np.ndarray,
    *,
    neighbors: int = 5,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Average CLIP predictions of exact task-feature agreement neighbors.

    References are selected only by the label-free DUET top-1 agreement mask.
    Exact cosine neighbors are returned in deterministic similarity/index order.
    """
    feature = np.asarray(task_feature, dtype=np.float64)
    clip = _probability_matrix(clip_probability, name="clip_probability")
    if feature.ndim != 2 or feature.shape[0] != clip.shape[0]:
        raise ValueError("task features must align with CLIP probabilities")
    if not np.isfinite(feature).all():
        raise ValueError("task features must be finite")
    agreement = np.asarray(agreement_mask, dtype=bool)
    query = np.asarray(query_mask, dtype=bool)
    if agreement.shape != query.shape or agreement.shape != (feature.shape[0],):
        raise ValueError("agreement and query masks must align with samples")
    if np.any(agreement & query):
        raise ValueError("agreement references and queries must be disjoint")
    reference_index = np.flatnonzero(agreement)
    query_index = np.flatnonzero(query)
    if query_index.size == 0 or reference_index.size < neighbors:
        raise ValueError("insufficient queries or agreement references")
    if neighbors <= 1 or chunk_size <= 0:
        raise ValueError("neighbors must exceed one and chunk_size must be positive")

    norm = np.linalg.norm(feature, axis=1)
    if np.any(norm <= 0.0):
        raise ValueError("task features must have non-zero norm")
    normalized = feature / norm[:, None]
    reference_feature = normalized[reference_index]
    neighbor_index = np.empty((query_index.size, neighbors), dtype=np.int64)
    neighbor_similarity = np.empty((query_index.size, neighbors), dtype=np.float64)
    for start in range(0, query_index.size, chunk_size):
        stop = min(start + chunk_size, query_index.size)
        similarity = normalized[query_index[start:stop]] @ reference_feature.T
        local_top = np.argpartition(-similarity, kth=neighbors - 1, axis=1)[
            :, :neighbors
        ]
        for row in range(stop - start):
            local = local_top[row]
            score = similarity[row, local]
            order = np.lexsort((reference_index[local], -score))
            local = local[order]
            neighbor_index[start + row] = reference_index[local]
            neighbor_similarity[start + row] = similarity[row, local]

    posterior = clip[neighbor_index].mean(axis=1)
    posterior /= posterior.sum(axis=1, keepdims=True)
    posterior_leave_farthest = clip[neighbor_index[:, :-1]].mean(axis=1)
    posterior_leave_farthest /= posterior_leave_farthest.sum(axis=1, keepdims=True)
    neighbor_clip_top1 = clip[neighbor_index].argmax(axis=2)
    consensus = np.max(
        np.stack(
            [
                (neighbor_clip_top1 == class_index).mean(axis=1)
                for class_index in range(clip.shape[1])
            ],
            axis=1,
        ),
        axis=1,
    )
    return {
        "query_index": query_index,
        "reference_index": reference_index,
        "neighbor_index": neighbor_index,
        "neighbor_similarity": neighbor_similarity,
        "posterior": posterior,
        "posterior_leave_farthest": posterior_leave_farthest,
        "neighbor_clip_top1_consensus": consensus,
    }


def select_from_candidate_set(
    posterior: np.ndarray,
    candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select the highest posterior class from each padded candidate set."""
    probability = _probability_matrix(posterior, name="posterior")
    candidate = np.asarray(candidates, dtype=np.int64)
    if candidate.ndim != 2 or candidate.shape[0] != probability.shape[0]:
        raise ValueError("candidates must align with posterior rows")
    valid = (candidate >= 0) & (candidate < probability.shape[1])
    if not np.all(valid.any(axis=1)):
        raise ValueError("every row must contain at least one valid candidate")
    safe = np.maximum(candidate, 0)
    score = np.take_along_axis(probability, safe, axis=1)
    score = np.where(valid, score, -np.inf)
    slot = score.argmax(axis=1)
    prediction = candidate[np.arange(candidate.shape[0]), slot]
    ordered = np.sort(score, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return {
        "prediction": prediction,
        "selected_slot": slot,
        "candidate_score": score,
        "margin": margin,
    }


def evaluate_agreement_neighbor_clip_gate(
    *,
    input_contract_valid: bool,
    neighbors: int,
    decision_stability_pct: float,
    candidate_set_coverage_pct: float,
    minimum_class_candidate_coverage_pct: float,
    neighbor_label_match_pct: float,
    comparisons: dict[str, dict[str, Any]],
    best_baseline_name: str,
    car_delta_pp: float,
    truck_delta_pp: float,
    car_truck_mean_delta_pp: float,
    other_ten_mean_delta_pp: float,
    max_class_mass_shift_pp: float,
) -> dict[str, Any]:
    """Apply the fixed no-training gate for cross-sample CLIP evidence."""
    required = {
        "fixed_task",
        "fixed_clip",
        "confidence_choice",
        "arithmetic",
        "rms",
    }
    if set(comparisons) != required or best_baseline_name not in required:
        raise ValueError("matched comparator contract is incomplete")
    numeric = (
        decision_stability_pct,
        candidate_set_coverage_pct,
        minimum_class_candidate_coverage_pct,
        neighbor_label_match_pct,
        car_delta_pp,
        truck_delta_pp,
        car_truck_mean_delta_pp,
        other_ten_mean_delta_pp,
        max_class_mass_shift_pp,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate metrics must be finite")
    for name, result in comparisons.items():
        interval = result.get("paired_bootstrap_95_ci_pp")
        values = (result.get("gain_pp"), *(interval or ()))
        if len(values) != 3 or not all(np.isfinite(value) for value in values):
            raise ValueError(f"invalid comparison: {name}")
    best = comparisons[best_baseline_name]
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "uses_predeclared_k5_neighbors": neighbors == 5,
        "leave_farthest_decision_stability_at_least_90pct": (
            decision_stability_pct >= 90.0
        ),
        "top2_union_oracle_coverage_at_least_90pct": (
            candidate_set_coverage_pct >= 90.0
        ),
        "every_class_candidate_coverage_at_least_85pct": (
            minimum_class_candidate_coverage_pct >= 85.0
        ),
        "neighbor_oracle_label_match_at_least_60pct": (
            neighbor_label_match_pct >= 60.0
        ),
        "accuracy_gain_vs_best_baseline_at_least_1pp": best["gain_pp"] >= 1.0,
        "accuracy_gain_vs_best_baseline_ci_lower_positive": (
            best["paired_bootstrap_95_ci_pp"][0] > 0.0
        ),
        "beats_every_matched_baseline": all(
            result["gain_pp"] > 0.0 for result in comparisons.values()
        ),
        "car_regression_at_most_0_5pp": car_delta_pp >= -0.5,
        "truck_regression_at_most_0_5pp": truck_delta_pp >= -0.5,
        "car_truck_mean_nonnegative": car_truck_mean_delta_pp >= 0.0,
        "other_ten_mean_nonnegative": other_ten_mean_delta_pp >= 0.0,
        "max_class_mass_shift_at_most_1pp": max_class_mass_shift_pp <= 1.0,
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_AGREEMENT_NEIGHBOR_CLIP_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "neighbors": 5,
            "min_leave_farthest_decision_stability_pct": 90.0,
            "min_candidate_set_coverage_pct": 90.0,
            "min_per_class_candidate_set_coverage_pct": 85.0,
            "min_neighbor_oracle_label_match_pct": 60.0,
            "min_accuracy_gain_vs_best_baseline_pp": 1.0,
            "best_baseline_ci_lower": "> 0",
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
            "max_class_mass_shift_pp": 1.0,
        },
        "training_authorized": False,
    }
