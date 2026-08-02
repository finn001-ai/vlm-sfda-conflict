"""Label-free helpers for capacity-preserving prototype transport audits."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def row_ordinal_cost(score: np.ndarray) -> np.ndarray:
    """Convert larger-is-better row scores to deterministic ordinal costs.

    The best class receives cost zero.  Stable sorting makes the contract
    deterministic without fitting a temperature, scale, or target-label
    threshold.
    """
    values = np.asarray(score, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("score must have shape [sample, class]")
    if not np.isfinite(values).all():
        raise ValueError("score must be finite")
    order = np.argsort(-values, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(values.shape[0])[:, None]
    rank[rows, order] = np.arange(values.shape[1])[None, :]
    return rank.astype(np.float64)


def prototype_cosine(
    feature: np.ndarray,
    classifier_weight: np.ndarray,
) -> np.ndarray:
    """Cosine similarity from target bottleneck features to source classes."""
    values = np.asarray(feature, dtype=np.float64)
    weight = np.asarray(classifier_weight, dtype=np.float64)
    if values.ndim != 2 or weight.ndim != 2:
        raise ValueError("feature and classifier_weight must be matrices")
    if values.shape[1] != weight.shape[1] or weight.shape[0] < 2:
        raise ValueError("feature and classifier dimensions do not match")
    if not np.isfinite(values).all() or not np.isfinite(weight).all():
        raise ValueError("feature and classifier_weight must be finite")
    feature_norm = np.linalg.norm(values, axis=1)
    weight_norm = np.linalg.norm(weight, axis=1)
    if np.any(feature_norm <= 0.0) or np.any(weight_norm <= 0.0):
        raise ValueError("feature and classifier rows must have positive norm")
    return (values / feature_norm[:, None]) @ (weight / weight_norm[:, None]).T


def capacity_preserving_transport(
    cost: np.ndarray,
    class_quota: np.ndarray,
) -> dict[str, Any]:
    """Solve an exact sample-to-class transportation linear program.

    Each sample supplies one unit and each class receives its predeclared
    integer quota.  The transportation constraint matrix is totally
    unimodular, so an optimal basic solution is integral for integer quotas.
    No entropic temperature or fitted solver hyperparameter is introduced.
    """
    values = np.asarray(cost, dtype=np.float64)
    quota = np.asarray(class_quota, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("cost must have shape [sample, class]")
    sample_count, class_count = values.shape
    if quota.shape != (class_count,):
        raise ValueError("class_quota must contain one value per class")
    if not np.isfinite(values).all() or np.any(quota < 0):
        raise ValueError("cost must be finite and quota non-negative")
    if int(quota.sum()) != sample_count:
        raise ValueError("class_quota must sum to the sample count")

    variable_count = sample_count * class_count
    variable = np.arange(variable_count, dtype=np.int64)
    sample_row = np.repeat(np.arange(sample_count, dtype=np.int64), class_count)
    class_index = np.tile(np.arange(class_count, dtype=np.int64), sample_count)
    # The final class equality is implied by all sample equalities and the
    # other class quotas.  Omitting that redundant row avoids a rank-deficient
    # HiGHS system that is disproportionately slow at VisDA scale.
    independent_class = class_index < class_count - 1
    class_row = sample_count + class_index[independent_class]
    row = np.concatenate((sample_row, class_row))
    column = np.concatenate((variable, variable[independent_class]))
    data = np.ones(row.size, dtype=np.float64)
    equality = sparse.coo_matrix(
        (data, (row, column)),
        shape=(sample_count + class_count - 1, variable_count),
    ).tocsr()
    rhs = np.concatenate((np.ones(sample_count, dtype=np.float64), quota[:-1]))
    result = linprog(
        values.reshape(-1),
        A_eq=equality,
        b_eq=rhs,
        bounds=(0.0, 1.0),
        method="highs-ds",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"transport solver failed: {result.message}")
    plan = np.asarray(result.x, dtype=np.float64).reshape(values.shape)
    row_error = float(np.max(np.abs(plan.sum(axis=1) - 1.0)))
    class_error = float(np.max(np.abs(plan.sum(axis=0) - quota)))
    integrality_error = float(np.max(np.minimum(plan, 1.0 - plan)))
    prediction = plan.argmax(axis=1).astype(np.int64)
    hard_count = np.bincount(prediction, minlength=class_count)
    if row_error > 1e-6 or class_error > 1e-6:
        raise RuntimeError("transport equality constraints were not satisfied")
    if integrality_error > 1e-6 or not np.array_equal(hard_count, quota):
        raise RuntimeError("transport solution is not an integral quota assignment")
    return {
        "prediction": prediction,
        "plan": plan,
        "objective": float(result.fun),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "row_sum_max_error": row_error,
        "class_sum_max_error": class_error,
        "integrality_max_error": integrality_error,
    }


def evaluate_prototype_transport_gate(
    *,
    input_contract_valid: bool,
    quota_exact: bool,
    integrality_max_error: float,
    changed_fraction_pct: float,
    comparisons: dict[str, dict[str, Any]],
    best_baseline_name: str,
    full_macro_delta_pp: float,
    car_delta_pp: float,
    truck_delta_pp: float,
    car_truck_mean_delta_pp: float,
    other_ten_mean_delta_pp: float,
) -> dict[str, Any]:
    """Apply the fixed offline gate; passing authorizes design review only."""
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
        integrality_max_error,
        changed_fraction_pct,
        full_macro_delta_pp,
        car_delta_pp,
        truck_delta_pp,
        car_truck_mean_delta_pp,
        other_ten_mean_delta_pp,
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
        "fixed_clip_conflict_quota_preserved_exactly": bool(quota_exact),
        "transport_integrality_error_at_most_1e_6": integrality_max_error <= 1e-6,
        "changed_conflicts_at_least_5pct": changed_fraction_pct >= 5.0,
        "beats_every_matched_baseline": all(
            result["gain_pp"] > 0.0 for result in comparisons.values()
        ),
        "gain_vs_best_baseline_at_least_1pp": best["gain_pp"] >= 1.0,
        "gain_vs_best_baseline_ci_lower_positive": (
            best["paired_bootstrap_95_ci_pp"][0] > 0.0
        ),
        "full_proxy_macro_gain_at_least_0_20pp": full_macro_delta_pp >= 0.2,
        "car_regression_at_most_0_5pp": car_delta_pp >= -0.5,
        "truck_regression_at_most_0_5pp": truck_delta_pp >= -0.5,
        "car_truck_mean_nonnegative": car_truck_mean_delta_pp >= 0.0,
        "other_ten_mean_nonnegative": other_ten_mean_delta_pp >= 0.0,
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_PROTOTYPE_TRANSPORT_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "min_changed_conflicts_pct": 5.0,
            "min_gain_vs_best_baseline_pp": 1.0,
            "best_baseline_paired_ci_lower_pp": "> 0",
            "min_full_proxy_macro_gain_pp": 0.2,
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
        },
        "training_authorized": False,
        "proxy_authorized": False,
        "gpu_authorized": False,
    }
