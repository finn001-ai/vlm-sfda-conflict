"""Pure NumPy helpers for the DUET conflict-gradient interference audit."""

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
    if not np.allclose(row_sum, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} rows must sum to one")
    return values / row_sum[:, None]


def duet_output_descent_components(
    weak_probability: np.ndarray,
    strong_probability: np.ndarray,
    clip_probability: np.ndarray,
    *,
    consistency_weight: float = 0.2,
    clip_weight: float = 0.4,
    probability_floor: float | None = None,
) -> dict[str, np.ndarray]:
    """Return the released DUET consistency and CLIP-KL logit descents.

    The consistency loss is ``KL(weak || strong)`` and the CLIP term is
    ``KL(clip || weak)``.  Both weak and strong probabilities receive the
    legacy consistency gradient.  The positive batch-mean scalar is omitted
    because it is shared by both components and does not change their angle.

    Snapshots store float32 probabilities rather than raw logits.  Exact zeros
    can therefore represent underflowed positive softmax values.  The caller
    supplies a log floor and must audit stability across multiple floors.
    """
    weak = _probability_matrix(weak_probability, name="weak_probability")
    strong = _probability_matrix(strong_probability, name="strong_probability")
    clip = _probability_matrix(clip_probability, name="clip_probability")
    if weak.shape != strong.shape or weak.shape != clip.shape:
        raise ValueError("weak, strong, and CLIP probability shapes must match")
    weights = np.asarray([consistency_weight, clip_weight], dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("loss weights must be finite and positive")
    if probability_floor is None:
        probability_floor = float(np.finfo(np.float32).tiny)
    if not np.isfinite(probability_floor) or not 0.0 < probability_floor < 1.0:
        raise ValueError("probability_floor must be finite and in (0, 1)")

    log_ratio = np.log(np.maximum(weak, probability_floor)) - np.log(
        np.maximum(strong, probability_floor)
    )
    weak_strong_kl = np.sum(
        np.where(weak > 0.0, weak * log_ratio, 0.0), axis=1
    )
    consistency_weak = -float(consistency_weight) * weak * (
        log_ratio - weak_strong_kl[:, None]
    )
    consistency_strong = float(consistency_weight) * (weak - strong)
    clip_weak = float(clip_weight) * (clip - weak)
    zeros = np.zeros_like(clip_weak)
    consistency_joint = np.concatenate(
        (consistency_weak, consistency_strong), axis=1
    )
    clip_joint = np.concatenate((clip_weak, zeros), axis=1)
    baseline = consistency_joint + clip_joint
    result = {
        "consistency_weak": consistency_weak,
        "consistency_strong": consistency_strong,
        "clip_weak": clip_weak,
        "consistency_joint": consistency_joint,
        "clip_joint": clip_joint,
        "baseline_joint": baseline,
        "weak_strong_kl": weak_strong_kl,
    }
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    class_count = weak.shape[1]
    for name in ("consistency_joint", "clip_joint", "baseline_joint"):
        value = result[name]
        if not (
            np.allclose(value[:, :class_count].sum(1), 0.0, atol=1e-10)
            and np.allclose(value[:, class_count:].sum(1), 0.0, atol=1e-10)
        ):
            raise RuntimeError(f"{name} must sum to zero in each logit branch")
    return result


def symmetric_pcgrad(
    first_descent: np.ndarray,
    second_descent: np.ndarray,
    *,
    epsilon: float = 1e-15,
) -> dict[str, np.ndarray]:
    """Apply the deterministic two-objective PCGrad projection row-wise.

    Each direction is projected onto the normal plane of the other only when
    their dot product is negative.  With exactly two objectives this is the
    symmetric, order-free form of PCGrad and introduces no threshold or fitted
    hyperparameter.
    """
    first = np.asarray(first_descent, dtype=np.float64)
    second = np.asarray(second_descent, dtype=np.float64)
    if first.ndim != 2 or first.shape != second.shape or first.shape[0] == 0:
        raise ValueError("descent arrays must be non-empty same-shaped matrices")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("descent arrays must be finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    dot = np.einsum("ij,ij->i", first, second)
    first_norm_sq = np.einsum("ij,ij->i", first, first)
    second_norm_sq = np.einsum("ij,ij->i", second, second)
    active = (
        (dot < 0.0)
        & (first_norm_sq > epsilon)
        & (second_norm_sq > epsilon)
    )
    first_projected = first.copy()
    second_projected = second.copy()
    first_projected[active] -= (
        dot[active] / second_norm_sq[active]
    )[:, None] * second[active]
    second_projected[active] -= (
        dot[active] / first_norm_sq[active]
    )[:, None] * first[active]

    denominator = np.sqrt(first_norm_sq * second_norm_sq)
    cosine = np.zeros(first.shape[0], dtype=np.float64)
    nonzero = denominator > epsilon
    cosine[nonzero] = dot[nonzero] / denominator[nonzero]
    result = {
        "candidate_joint": first_projected + second_projected,
        "first_projected": first_projected,
        "second_projected": second_projected,
        "component_dot": dot,
        "component_cosine": np.clip(cosine, -1.0, 1.0),
        "gradient_conflict": active,
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("PCGrad result must be finite")
    return result


def decision_stability(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if reference.ndim != 1 or reference.shape != candidate.shape:
        raise ValueError("decision arrays must be same-shaped 1-D masks")
    if reference.size == 0:
        raise ValueError("decision arrays must not be empty")
    return float(np.mean(reference == candidate) * 100.0)


def direction_stability(
    reference: np.ndarray, candidate: np.ndarray, *, epsilon: float = 1e-15
) -> float:
    """Return median row-wise cosine, treating two zero rows as identical."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 2 or reference.shape != candidate.shape:
        raise ValueError("direction arrays must be same-shaped 2-D matrices")
    if reference.shape[0] == 0:
        raise ValueError("direction arrays must not be empty")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("direction arrays must be finite")
    reference_norm = np.linalg.norm(reference, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    denominator = reference_norm * candidate_norm
    cosine = np.zeros(reference.shape[0], dtype=np.float64)
    nonzero = denominator > epsilon
    cosine[nonzero] = np.einsum(
        "ij,ij->i", reference[nonzero], candidate[nonzero]
    ) / denominator[nonzero]
    both_zero = (reference_norm <= epsilon) & (candidate_norm <= epsilon)
    cosine[both_zero] = 1.0
    return float(np.median(np.clip(cosine, -1.0, 1.0)) * 100.0)


def evaluate_conflict_pcgrad_gate(
    *,
    input_contract_valid: bool,
    conflict_coverage_pct: float,
    floor_decision_stability_pct: float,
    floor_direction_stability_pct: float,
    floor_mean_norm_ratio_max_deviation: float,
    overall_first_order: dict[str, Any],
    conflict_first_order: dict[str, Any],
    baseline_negative_burden: float,
    candidate_negative_burden: float,
    helpful_retention_pct: float,
    candidate_to_baseline_mean_norm_ratio: float,
    group_first_order_delta: dict[str, float],
) -> dict[str, Any]:
    """Gate an output-space diagnosis; passing authorizes no training."""
    for result in (overall_first_order, conflict_first_order):
        interval = result.get("paired_bootstrap_95_ci")
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or not all(
                np.isfinite(value)
                for value in (result.get("mean_difference"), *interval)
            )
        ):
            raise ValueError("invalid paired first-order comparison")
    if set(group_first_order_delta) != {"car", "person", "truck", "other_nine"}:
        raise ValueError("group deltas must contain car, person, truck, other_nine")
    numeric = (
        conflict_coverage_pct,
        floor_decision_stability_pct,
        floor_direction_stability_pct,
        floor_mean_norm_ratio_max_deviation,
        baseline_negative_burden,
        candidate_negative_burden,
        helpful_retention_pct,
        candidate_to_baseline_mean_norm_ratio,
        *group_first_order_delta.values(),
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate metrics must be finite")

    def positive_with_ci(result: dict[str, Any]) -> bool:
        return bool(
            result["mean_difference"] > 0.0
            and result["paired_bootstrap_95_ci"][0] > 0.0
        )

    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "gradient_conflict_coverage_at_least_5pct": conflict_coverage_pct >= 5.0,
        "underflow_floor_decision_stability_at_least_95pct": (
            floor_decision_stability_pct >= 95.0
        ),
        "underflow_floor_direction_stability_at_least_99pct": (
            floor_direction_stability_pct >= 99.0
        ),
        "underflow_floor_mean_norm_ratio_within_5pct": (
            floor_mean_norm_ratio_max_deviation <= 0.05
        ),
        "overall_first_order_gain_ci_lower_positive": positive_with_ci(
            overall_first_order
        ),
        "conflict_subset_first_order_gain_ci_lower_positive": positive_with_ci(
            conflict_first_order
        ),
        "candidate_negative_burden_not_worse": (
            candidate_negative_burden >= baseline_negative_burden
        ),
        "helpful_first_order_retention_at_least_99pct": helpful_retention_pct >= 99.0,
        "mean_descent_norm_inflation_at_most_1_5x": (
            candidate_to_baseline_mean_norm_ratio <= 1.5
        ),
        "car_first_order_delta_nonnegative": group_first_order_delta["car"] >= 0.0,
        "person_first_order_delta_nonnegative": (
            group_first_order_delta["person"] >= 0.0
        ),
        "truck_first_order_delta_nonnegative": (
            group_first_order_delta["truck"] >= 0.0
        ),
        "other_nine_first_order_delta_nonnegative": (
            group_first_order_delta["other_nine"] >= 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "decision": "NEEDS_PARAMETER_AUDIT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "gradient_conflict_coverage_pct": ">= 5",
            "underflow_floor_decision_stability_pct": ">= 95",
            "underflow_floor_direction_stability_pct": ">= 99",
            "underflow_floor_mean_norm_ratio_max_deviation": "<= 0.05",
            "paired_first_order_mean_and_ci_lower": "> 0",
            "candidate_negative_burden_minus_baseline": ">= 0",
            "helpful_first_order_retention_pct": ">= 99",
            "candidate_to_baseline_mean_norm_ratio": "<= 1.5",
            "car_person_truck_other_nine_first_order_delta": ">= 0",
        },
        "training_authorized": False,
        "proxy_authorized": False,
    }
