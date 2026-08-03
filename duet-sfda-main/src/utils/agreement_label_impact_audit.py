"""Agreement-anchored label-impact helpers for a CPU-only VisDA audit.

The audit treats cycle-1 task/CLIP agreements as a noisy partial-label
reference set.  For each candidate label, it asks whether one classifier-head
gradient step induced by that label is locally compatible with the empirical
loss landscape of agreement samples assigned to the same class.  This is a
diagnostic surrogate only: DUET keeps its source classifier frozen and this
module never updates a parameter.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _probability_matrix(probability: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] < 2:
        raise ValueError(f"{name} must have shape [sample, class]")
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    row_sum = value.sum(axis=1)
    if np.any(row_sum <= 0.0) or not np.allclose(row_sum, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} rows must sum to one")
    return value / row_sum[:, None]


def _feature_matrix(feature: np.ndarray, *, rows: int) -> np.ndarray:
    value = np.asarray(feature, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != rows or value.shape[1] == 0:
        raise ValueError("feature must have shape [sample, feature]")
    if not np.isfinite(value).all():
        raise ValueError("feature must be finite")
    # The appended one includes the classifier bias in the parameter-gradient
    # surrogate.  No data-derived scale or fitted coefficient is introduced.
    return np.concatenate((value, np.ones((rows, 1), dtype=np.float64)), axis=1)


def stratified_alternating_reference_masks(
    pseudo_label: np.ndarray,
    reference_mask: np.ndarray,
    sample_index: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split every pseudo class into deterministic alternating reference halves."""
    label = np.asarray(pseudo_label, dtype=np.int64)
    reference = np.asarray(reference_mask, dtype=bool)
    index = np.asarray(sample_index, dtype=np.int64)
    if label.ndim != 1 or label.shape != reference.shape or label.shape != index.shape:
        raise ValueError("labels, references, and sample indices must align")
    if np.unique(index).size != index.size:
        raise ValueError("sample indices must be unique")
    if class_count < 2 or np.any(label < 0) or np.any(label >= class_count):
        raise ValueError("pseudo label is outside the class range")

    first = np.zeros(label.size, dtype=bool)
    second = np.zeros(label.size, dtype=bool)
    for class_index in range(class_count):
        selected = np.flatnonzero(reference & (label == class_index))
        selected = selected[np.argsort(index[selected], kind="stable")]
        if selected.size < 4:
            raise ValueError(
                f"class {class_index} needs four references for a stable split"
            )
        first[selected[::2]] = True
        second[selected[1::2]] = True
    if np.any(first & second) or not np.array_equal(first | second, reference):
        raise RuntimeError("alternating reference masks are not a partition")
    return first, second


def fit_agreement_label_impact(
    task_probability: np.ndarray,
    task_feature: np.ndarray,
    pseudo_label: np.ndarray,
    reference_mask: np.ndarray,
    *,
    class_count: int,
) -> dict[str, np.ndarray]:
    """Fit a diagonal empirical-Fisher label-impact landscape.

    For an agreement sample ``(p, z, y)``, the frozen linear-head CE gradient
    is ``(p - one_hot(y)) outer [z, 1]``.  The global empirical Fisher supplies
    a scale for every head parameter, while the mean gradient of each agreement
    pseudo class is the class-conditional reference loss direction.  The only
    floor is a relative float64 safeguard and is not a fitted damping value.
    """
    probability = _probability_matrix(task_probability, name="task_probability")
    feature = _feature_matrix(task_feature, rows=probability.shape[0])
    label = np.asarray(pseudo_label, dtype=np.int64)
    reference = np.asarray(reference_mask, dtype=bool)
    if label.shape != reference.shape or label.shape != (probability.shape[0],):
        raise ValueError("pseudo labels and reference mask must align")
    if probability.shape[1] != class_count:
        raise ValueError("class_count does not match probability columns")
    if np.any(label < 0) or np.any(label >= class_count):
        raise ValueError("pseudo label is outside the class range")
    if not reference.any():
        raise ValueError("at least one agreement reference is required")

    residual = probability.copy()
    residual[np.arange(label.size), label] -= 1.0
    reference_residual = residual[reference]
    reference_feature = feature[reference]
    fisher = np.einsum(
        "nc,nd->cd",
        reference_residual**2,
        reference_feature**2,
        optimize=True,
    ) / float(reference.sum())
    maximum = float(fisher.max())
    numerical_floor = max(
        maximum * np.sqrt(np.finfo(np.float64).eps),
        np.finfo(np.float64).tiny,
    )
    fisher = np.maximum(fisher, numerical_floor)

    mean_gradient = np.empty(
        (class_count, class_count, feature.shape[1]), dtype=np.float64
    )
    reference_count = np.empty(class_count, dtype=np.int64)
    for class_index in range(class_count):
        selected = reference & (label == class_index)
        reference_count[class_index] = int(selected.sum())
        if reference_count[class_index] < 2:
            raise ValueError(
                f"class {class_index} needs at least two agreement references"
            )
        mean_gradient[class_index] = np.einsum(
            "nc,nd->cd",
            residual[selected],
            feature[selected],
            optimize=True,
        ) / float(reference_count[class_index])

    preconditioned_gradient = mean_gradient / fisher[None, :, :]
    if not all(
        np.isfinite(value).all()
        for value in (fisher, mean_gradient, preconditioned_gradient)
    ):
        raise RuntimeError("label-impact landscape must be finite")
    return {
        "fisher_diagonal": fisher,
        "mean_gradient": mean_gradient,
        "preconditioned_mean_gradient": preconditioned_gradient,
        "reference_count": reference_count,
        "numerical_floor": np.asarray(numerical_floor, dtype=np.float64),
    }


def label_impact_score(
    task_probability: np.ndarray,
    task_feature: np.ndarray,
    model: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the first-order agreement-loss improvement for every label.

    A candidate CE gradient produces the hypothetical parameter step
    ``-H^-1 g_candidate``.  Its first-order improvement on agreement references
    of the same pseudo class is therefore ``g_reference^T H^-1 g_candidate``.
    """
    probability = _probability_matrix(task_probability, name="task_probability")
    feature = _feature_matrix(task_feature, rows=probability.shape[0])
    preconditioned = np.asarray(model["preconditioned_mean_gradient"], dtype=np.float64)
    class_count = probability.shape[1]
    if preconditioned.shape != (class_count, class_count, feature.shape[1]):
        raise ValueError("preconditioned gradient shape is incompatible")
    if not np.isfinite(preconditioned).all():
        raise ValueError("preconditioned gradients must be finite")

    score = np.empty((probability.shape[0], class_count), dtype=np.float64)
    for candidate_class in range(class_count):
        # projection[n, output_class] is the dot product between the query
        # augmented feature and the corresponding preconditioned head row.
        projection = feature @ preconditioned[candidate_class].T
        candidate_residual = probability.copy()
        candidate_residual[:, candidate_class] -= 1.0
        score[:, candidate_class] = np.sum(candidate_residual * projection, axis=1)
    if not np.isfinite(score).all():
        raise RuntimeError("label-impact scores must be finite")
    return score


def select_candidate_by_label_impact(
    score: np.ndarray,
    candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select the maximum-impact member of each padded candidate set."""
    value = np.asarray(score, dtype=np.float64)
    candidate = np.asarray(candidates, dtype=np.int64)
    if value.ndim != 2 or candidate.ndim != 2:
        raise ValueError("scores and candidates must be matrices")
    if value.shape[0] != candidate.shape[0]:
        raise ValueError("scores and candidates must have matching rows")
    valid = (candidate >= 0) & (candidate < value.shape[1])
    if not np.all(valid.any(axis=1)):
        raise ValueError("every row needs at least one valid candidate")
    safe = np.maximum(candidate, 0)
    candidate_score = np.take_along_axis(value, safe, axis=1)
    candidate_score = np.where(valid, candidate_score, -np.inf)
    selected_slot = candidate_score.argmax(axis=1)
    prediction = candidate[np.arange(candidate.shape[0]), selected_slot]
    ordered = np.sort(candidate_score, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return {
        "prediction": prediction,
        "selected_slot": selected_slot,
        "candidate_score": candidate_score,
        "margin": margin,
    }


def evaluate_agreement_label_impact_gate(
    *,
    input_contract_valid: bool,
    agreement_reference_accuracy_pct: float,
    minimum_split_decision_stability_pct: float,
    candidate_set_coverage_pct: float,
    minimum_class_candidate_coverage_pct: float,
    comparisons: dict[str, dict[str, Any]],
    best_baseline_name: str,
    car_delta_pp: float,
    truck_delta_pp: float,
    car_truck_mean_delta_pp: float,
    other_ten_mean_delta_pp: float,
    max_class_mass_shift_pp: float,
) -> dict[str, Any]:
    """Apply the fixed no-training gate for agreement label impact."""
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
        agreement_reference_accuracy_pct,
        minimum_split_decision_stability_pct,
        candidate_set_coverage_pct,
        minimum_class_candidate_coverage_pct,
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
        "agreement_reference_accuracy_at_least_90pct": (
            agreement_reference_accuracy_pct >= 90.0
        ),
        "minimum_split_decision_stability_at_least_90pct": (
            minimum_split_decision_stability_pct >= 90.0
        ),
        "top2_union_oracle_coverage_at_least_90pct": (
            candidate_set_coverage_pct >= 90.0
        ),
        "every_class_candidate_coverage_at_least_85pct": (
            minimum_class_candidate_coverage_pct >= 85.0
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
    return {
        "decision": (
            "PASS_AGREEMENT_LABEL_IMPACT_PREFLIGHT"
            if all(checks.values())
            else "REJECT"
        ),
        "checks": checks,
        "thresholds": {
            "min_agreement_reference_accuracy_pct": 90.0,
            "min_split_decision_stability_pct": 90.0,
            "min_candidate_set_coverage_pct": 90.0,
            "min_per_class_candidate_set_coverage_pct": 85.0,
            "min_accuracy_gain_vs_best_baseline_pp": 1.0,
            "best_baseline_ci_lower": "> 0",
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
            "max_class_mass_shift_pp": 1.0,
        },
        "training_authorized": False,
        "proxy_authorized": False,
        "gpu_authorized": False,
    }
