"""Label-free class-conditional joint-evidence GMM helpers."""

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
    if np.any(row_sum <= 0.0) or not np.allclose(
        row_sum, 1.0, atol=1e-5, rtol=1e-5
    ):
        raise ValueError(f"{name} rows must sum to one")
    return value / row_sum[:, None]


def joint_centered_log_probability(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
) -> np.ndarray:
    """Return concatenated centered-log task and CLIP evidence.

    Centered log probabilities preserve every within-model log-odds ratio while
    removing each row's arbitrary additive logit constant.  No temperature is
    fitted or shared between the two models.
    """
    task = _probability_matrix(task_probability, name="task_probability")
    clip = _probability_matrix(clip_probability, name="clip_probability")
    if task.shape != clip.shape:
        raise ValueError("task and CLIP probabilities must have matching shape")
    floor = float(np.finfo(np.float32).tiny)
    task_log = np.log(np.maximum(task, floor))
    clip_log = np.log(np.maximum(clip, floor))
    task_clr = task_log - task_log.mean(axis=1, keepdims=True)
    clip_clr = clip_log - clip_log.mean(axis=1, keepdims=True)
    evidence = np.concatenate((task_clr, clip_clr), axis=1)
    if not np.isfinite(evidence).all():
        raise RuntimeError("joint centered-log evidence must be finite")
    return evidence


def fit_diagonal_class_gaussians(
    evidence: np.ndarray,
    pseudo_label: np.ndarray,
    reference_mask: np.ndarray,
    *,
    class_count: int,
) -> dict[str, np.ndarray]:
    """Fit one diagonal Gaussian per pseudo class without label priors.

    The only floor is a floating-point safeguard derived from the pooled
    reference variance.  Class frequencies are deliberately not used as
    priors, because DUET agreement counts are admission-biased.
    """
    value = np.asarray(evidence, dtype=np.float64)
    label = np.asarray(pseudo_label, dtype=np.int64)
    reference = np.asarray(reference_mask, dtype=bool)
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError("evidence must be a non-empty matrix")
    if label.shape != reference.shape or label.shape != (value.shape[0],):
        raise ValueError("pseudo labels and reference mask must align with evidence")
    if not np.isfinite(value).all():
        raise ValueError("evidence must be finite")
    if class_count < 2 or np.any(label < 0) or np.any(label >= class_count):
        raise ValueError("pseudo label is outside the class range")
    if not reference.any():
        raise ValueError("at least one reference is required")

    pooled_variance = value[reference].var(axis=0)
    numerical_floor = np.maximum(
        pooled_variance * np.sqrt(np.finfo(np.float64).eps),
        np.finfo(np.float64).tiny,
    )
    mean = np.empty((class_count, value.shape[1]), dtype=np.float64)
    variance = np.empty_like(mean)
    counts = np.empty(class_count, dtype=np.int64)
    for class_index in range(class_count):
        selected = reference & (label == class_index)
        counts[class_index] = int(selected.sum())
        if counts[class_index] < 2:
            raise ValueError(
                f"class {class_index} needs at least two agreement references"
            )
        mean[class_index] = value[selected].mean(axis=0)
        variance[class_index] = np.maximum(
            value[selected].var(axis=0), numerical_floor
        )
    return {
        "mean": mean,
        "variance": variance,
        "reference_count": counts,
        "numerical_variance_floor": numerical_floor,
    }


def stratified_alternating_reference_masks(
    pseudo_label: np.ndarray,
    reference_mask: np.ndarray,
    sample_index: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each pseudo class deterministically by ordered alternating rows."""
    label = np.asarray(pseudo_label, dtype=np.int64)
    reference = np.asarray(reference_mask, dtype=bool)
    index = np.asarray(sample_index, dtype=np.int64)
    if label.shape != reference.shape or label.shape != index.shape or label.ndim != 1:
        raise ValueError("labels, reference mask, and sample indices must align")
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
                f"class {class_index} needs four references for an alternating split"
            )
        first[selected[::2]] = True
        second[selected[1::2]] = True
    if np.any(first & second) or not np.array_equal(first | second, reference):
        raise RuntimeError("stratified reference split is not a partition")
    return first, second


def diagonal_gaussian_log_likelihood(
    evidence: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    """Evaluate uniform-prior diagonal Gaussian class log likelihoods."""
    value = np.asarray(evidence, dtype=np.float64)
    center = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(variance, dtype=np.float64)
    if value.ndim != 2 or center.ndim != 2 or scale.shape != center.shape:
        raise ValueError("evidence, mean, and variance shapes are invalid")
    if value.shape[1] != center.shape[1]:
        raise ValueError("evidence dimension does not match Gaussian parameters")
    if not all(np.isfinite(item).all() for item in (value, center, scale)):
        raise ValueError("Gaussian inputs must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("Gaussian variances must be positive")
    residual = value[:, None, :] - center[None, :, :]
    result = -0.5 * np.sum(
        np.log(scale)[None, :, :] + residual**2 / scale[None, :, :], axis=2
    )
    if not np.isfinite(result).all():
        raise RuntimeError("Gaussian log likelihood must be finite")
    return result


def select_candidate_by_log_likelihood(
    log_likelihood: np.ndarray,
    candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select the maximum-likelihood member of each padded candidate set."""
    score = np.asarray(log_likelihood, dtype=np.float64)
    candidate = np.asarray(candidates, dtype=np.int64)
    if score.ndim != 2 or candidate.ndim != 2:
        raise ValueError("scores and candidates must be matrices")
    if score.shape[0] != candidate.shape[0]:
        raise ValueError("scores and candidates must have matching rows")
    valid = (candidate >= 0) & (candidate < score.shape[1])
    if not np.all(valid.any(axis=1)):
        raise ValueError("every row needs at least one valid candidate")
    safe = np.maximum(candidate, 0)
    candidate_score = np.take_along_axis(score, safe, axis=1)
    candidate_score = np.where(valid, candidate_score, -np.inf)
    selected_slot = candidate_score.argmax(axis=1)
    prediction = candidate[np.arange(candidate.shape[0]), selected_slot]
    ordered = np.sort(candidate_score, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return {
        "prediction": prediction,
        "selected_slot": selected_slot,
        "candidate_log_likelihood": candidate_score,
        "margin": margin,
    }


def evaluate_agreement_gmm_gate(
    *,
    input_contract_valid: bool,
    reference_crossfit_accuracy_pct: float,
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
    """Apply the fixed, no-training gate for the joint-evidence GMM."""
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
        reference_crossfit_accuracy_pct,
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
        "agreement_crossfit_pseudo_accuracy_at_least_90pct": (
            reference_crossfit_accuracy_pct >= 90.0
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
            "PASS_AGREEMENT_GMM_PREFLIGHT" if all(checks.values()) else "REJECT"
        ),
        "checks": checks,
        "thresholds": {
            "min_reference_crossfit_accuracy_pct": 90.0,
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
