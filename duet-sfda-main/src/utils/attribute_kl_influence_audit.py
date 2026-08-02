"""First-order logit-space diagnostics for a changed KL soft target.

The functions in this module never construct a model or an optimizer.  They
compare two fixed KL targets at the same student probability.  For
``KL(target || student)`` the logit descent direction is ``target - student``;
therefore the candidate's incremental direction over the control is exactly
``candidate_target - control_target``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _probability_array(name: str, value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError(f"{name} must have shape [sample, class]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError(f"{name} rows must sum to one")
    return result


def kl_logit_descent_directions(
    student_probability: np.ndarray,
    control_target: np.ndarray,
    candidate_target: np.ndarray,
    *,
    kl_weight: float,
) -> dict[str, np.ndarray]:
    """Return fixed-target KL descent directions in student-logit space.

    This is an exact derivative with respect to the student logits at the
    supplied probability, but it is not a parameter-gradient calculation: the
    network Jacobian and optimizer dynamics are intentionally absent.
    """
    student = _probability_array("student_probability", student_probability)
    control = _probability_array("control_target", control_target)
    candidate = _probability_array("candidate_target", candidate_target)
    if student.shape != control.shape or student.shape != candidate.shape:
        raise ValueError("student and target probability shapes must match")
    if not np.isfinite(kl_weight) or kl_weight <= 0.0:
        raise ValueError("kl_weight must be finite and positive")

    control_direction = kl_weight * (control - student)
    candidate_direction = kl_weight * (candidate - student)
    incremental_direction = candidate_direction - control_direction
    expected_increment = kl_weight * (candidate - control)
    if not np.allclose(
        incremental_direction, expected_increment, atol=1e-14, rtol=1e-12
    ):
        raise RuntimeError("incremental KL direction identity failed")
    return {
        "control_direction": control_direction,
        "candidate_direction": candidate_direction,
        "incremental_direction": incremental_direction,
    }


def oracle_logit_influence(
    student_probability: np.ndarray,
    control_direction: np.ndarray,
    candidate_direction: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project KL descent directions onto supervised-CE descent directions.

    Target labels are required and every returned value is consequently an
    oracle diagnostic.  A positive incremental projection means the changed
    target is more aligned with supervised-CE descent than the control target
    at the same student probability.
    """
    student = _probability_array("student_probability", student_probability)
    control = np.asarray(control_direction, dtype=np.float64)
    candidate = np.asarray(candidate_direction, dtype=np.float64)
    if control.shape != student.shape or candidate.shape != student.shape:
        raise ValueError("direction shapes must match student_probability")
    if not np.isfinite(control).all() or not np.isfinite(candidate).all():
        raise ValueError("directions must be finite")
    label = np.asarray(labels, dtype=np.int64)
    if label.shape != (student.shape[0],):
        raise ValueError("labels must contain one class per sample")
    if np.any(label < 0) or np.any(label >= student.shape[1]):
        raise ValueError("label is outside the class range")

    oracle = -student.copy()
    oracle[np.arange(student.shape[0]), label] += 1.0
    incremental = candidate - control

    def projection(direction: np.ndarray) -> np.ndarray:
        return np.einsum("nc,nc->n", direction, oracle)

    def cosine(direction: np.ndarray) -> np.ndarray:
        numerator = projection(direction)
        denominator = np.linalg.norm(direction, axis=1) * np.linalg.norm(oracle, axis=1)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )

    return {
        "oracle_direction": oracle,
        "control_projection": projection(control),
        "candidate_projection": projection(candidate),
        "incremental_projection": projection(incremental),
        "control_cosine": cosine(control),
        "candidate_cosine": cosine(candidate),
        "incremental_cosine": cosine(incremental),
    }


def paired_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    repeats: int = 2_000,
) -> tuple[float, float]:
    """Return a deterministic percentile CI for a paired per-sample mean."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    generator = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 100):
        count = min(100, repeats - start)
        indices = generator.integers(0, array.size, size=(count, array.size))
        means[start : start + count] = array[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def evaluate_attribute_kl_influence(
    *,
    input_contract_valid: bool,
    active_conflict_count_matches: bool,
    changed_top1_count_matches: bool,
    mean_incremental_projection: float,
    incremental_projection_ci: tuple[float, float],
    macro_class_mean_projection: float,
    car_mean_projection: float,
    truck_mean_projection: float,
    observed_final_delta_pp: float,
    observed_hard_mean_delta_pp: float,
    min_observed_final_delta_pp: float = 0.20,
) -> dict[str, Any]:
    """Classify why an already-run candidate may or may not be salvageable.

    The observed proxy checks remain decisive.  Passing this diagnostic never
    authorizes another training run.
    """
    numeric = (
        mean_incremental_projection,
        *incremental_projection_ci,
        macro_class_mean_projection,
        car_mean_projection,
        truck_mean_projection,
        observed_final_delta_pp,
        observed_hard_mean_delta_pp,
        min_observed_final_delta_pp,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate inputs must be finite")
    signal_checks = {
        "input_contract_valid": bool(input_contract_valid),
        "active_conflict_count_matches_proxy_log": bool(active_conflict_count_matches),
        "changed_top1_count_matches_proxy_log": bool(changed_top1_count_matches),
        "mean_incremental_oracle_projection_positive": (
            mean_incremental_projection > 0.0
        ),
        "incremental_projection_ci_lower_positive": (
            incremental_projection_ci[0] > 0.0
        ),
        "macro_class_mean_projection_positive": macro_class_mean_projection > 0.0,
        "car_mean_projection_nonnegative": car_mean_projection >= 0.0,
        "truck_mean_projection_nonnegative": truck_mean_projection >= 0.0,
    }
    translation_checks = {
        "observed_proxy_final_gain_at_least_0.20pp": (
            observed_final_delta_pp >= min_observed_final_delta_pp
        ),
        "observed_proxy_hard_classes_noninferior": (observed_hard_mean_delta_pp >= 0.0),
    }
    signal_passed = all(signal_checks.values())
    translation_passed = all(translation_checks.values())
    if not signal_passed:
        diagnosis = "incremental_kl_direction_is_not_class_safe"
    elif not translation_passed:
        diagnosis = "directional_signal_did_not_translate_to_proxy_accuracy"
    else:
        diagnosis = "direction_and_translation_checks_passed"
    return {
        "decision": (
            "PASS_DIAGNOSTIC_ONLY"
            if signal_passed and translation_passed
            else "REJECT_ATTRIBUTE_BRANCH"
        ),
        "diagnosis": diagnosis,
        "signal_checks": signal_checks,
        "translation_checks": translation_checks,
        "training_authorized": False,
    }
