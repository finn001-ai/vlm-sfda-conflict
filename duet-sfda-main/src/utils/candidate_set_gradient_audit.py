"""Pure NumPy helpers for the candidate-set logit-gradient audit."""

from __future__ import annotations

from typing import Any

import numpy as np


def _probability_matrix(probability: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(f"{name} must have shape [sample, class]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError(f"{name} rows must sum to one")
    return values


def kl_logit_descent(
    student_probability: np.ndarray,
    target_probability: np.ndarray,
) -> np.ndarray:
    """Return ``-d KL(target || student) / d student_logits``.

    This is the exact logit-space descent used by DUET's CLIP KL term, apart
    from its positive scalar loss weight and batch reduction.
    """
    student = _probability_matrix(student_probability, name="student_probability")
    target = _probability_matrix(target_probability, name="target_probability")
    if student.shape != target.shape:
        raise ValueError("student and target probability shapes must match")
    return target - student


def set_mass_logit_descent(
    student_probability: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """Return the exact descent for ``-log(sum(student[c] for c in S))``."""
    student = _probability_matrix(student_probability, name="student_probability")
    mask = np.asarray(candidate_mask, dtype=bool)
    if mask.shape != student.shape:
        raise ValueError("candidate mask and student probability shapes must match")
    if not np.all(mask.any(axis=1)):
        raise ValueError("every candidate set must be non-empty")
    mass = np.where(mask, student, 0.0).sum(axis=1)
    if not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise ValueError("candidate set must contain positive student mass")
    descent = np.where(mask, student / mass[:, None], 0.0) - student
    if not np.isfinite(descent).all():
        raise ValueError("set-mass descent must be finite")
    if not np.allclose(descent.sum(axis=1), 0.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("set-mass descent must sum to zero per row")
    return descent


def oracle_ce_logit_descent(
    student_probability: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return oracle CE logit descent for diagnostic use only."""
    student = _probability_matrix(student_probability, name="student_probability")
    target = np.asarray(labels, dtype=np.int64)
    if target.shape != (student.shape[0],):
        raise ValueError("labels must contain one class per sample")
    if np.any(target < 0) or np.any(target >= student.shape[1]):
        raise ValueError("label is outside the class range")
    descent = -student.copy()
    descent[np.arange(target.size), target] += 1.0
    return descent


def rowwise_oracle_alignment(
    candidate_descent: np.ndarray,
    oracle_descent: np.ndarray,
    *,
    epsilon: float = 1e-15,
) -> dict[str, np.ndarray]:
    """Measure first-order oracle benefit and direction-only cosine alignment.

    ``first_order`` is the directional increase in oracle log-probability for
    an infinitesimal logit step. ``oracle_unit_projection`` retains candidate
    gradient magnitude while normalizing only oracle difficulty. Cosine is set
    to zero when either direction has effectively zero norm; those rows are
    separately exposed by ``joint_nonzero``.
    """
    candidate = np.asarray(candidate_descent, dtype=np.float64)
    oracle = np.asarray(oracle_descent, dtype=np.float64)
    if (
        candidate.ndim != 2
        or candidate.shape != oracle.shape
        or candidate.shape[0] == 0
    ):
        raise ValueError("candidate and oracle descents must be same-shaped 2-D arrays")
    if not np.isfinite(candidate).all() or not np.isfinite(oracle).all():
        raise ValueError("descent arrays must be finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    first_order = np.einsum("ij,ij->i", candidate, oracle)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    oracle_norm = np.linalg.norm(oracle, axis=1)
    joint_nonzero = (candidate_norm > epsilon) & (oracle_norm > epsilon)
    cosine = np.zeros(candidate.shape[0], dtype=np.float64)
    cosine[joint_nonzero] = first_order[joint_nonzero] / (
        candidate_norm[joint_nonzero] * oracle_norm[joint_nonzero]
    )
    oracle_unit_projection = np.zeros(candidate.shape[0], dtype=np.float64)
    oracle_nonzero = oracle_norm > epsilon
    oracle_unit_projection[oracle_nonzero] = (
        first_order[oracle_nonzero] / oracle_norm[oracle_nonzero]
    )
    return {
        "first_order": first_order,
        "oracle_unit_projection": oracle_unit_projection,
        "cosine": np.clip(cosine, -1.0, 1.0),
        "candidate_norm": candidate_norm,
        "oracle_norm": oracle_norm,
        "joint_nonzero": joint_nonzero,
    }


def paired_mean_bootstrap_ci(
    difference: np.ndarray,
    *,
    repeats: int = 2_000,
    seed: int = 2_020,
    batch_size: int = 50,
) -> tuple[float, float]:
    """Return a paired-bootstrap 95% CI for a mean per-sample difference."""
    values = np.asarray(difference, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("difference must be a finite non-empty 1-D array")
    if repeats <= 0 or batch_size <= 0:
        raise ValueError("repeats and batch_size must be positive")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats, dtype=np.float64)
    written = 0
    while written < repeats:
        current = min(batch_size, repeats - written)
        indices = rng.integers(0, values.size, size=(current, values.size))
        bootstrap[written : written + current] = values[indices].mean(axis=1)
        written += current
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def evaluate_candidate_gradient_gate(
    *,
    input_contract_valid: bool,
    comparisons: dict[str, dict[str, dict[str, Any]]],
    macro_first_order_delta_vs_clip: float,
    hard_class_first_order_delta_vs_clip: dict[str, float],
    top2_harmful_pct: float,
    clip_harmful_pct: float,
) -> dict[str, Any]:
    """Apply the predeclared gradient gate; passing never starts training."""
    required = ("versus_clip_kl", "versus_top1_set")
    metrics = ("cosine", "oracle_unit_projection", "first_order")
    for comparison in required:
        if comparison not in comparisons:
            raise ValueError(f"missing comparison: {comparison}")
        for metric in metrics:
            result = comparisons[comparison].get(metric, {})
            mean = result.get("mean_difference")
            interval = result.get("paired_bootstrap_95_ci")
            if (
                not isinstance(interval, (list, tuple))
                or len(interval) != 2
                or not all(np.isfinite(value) for value in (mean, *interval))
            ):
                raise ValueError(f"invalid {comparison} {metric} result")
    hard_classes = ("car", "person", "truck")
    if set(hard_class_first_order_delta_vs_clip) != set(hard_classes):
        raise ValueError("hard-class results must contain car, person, and truck")
    numeric = (
        macro_first_order_delta_vs_clip,
        top2_harmful_pct,
        clip_harmful_pct,
        *hard_class_first_order_delta_vs_clip.values(),
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate values must be finite")

    def positive_with_ci(comparison: str, metric: str) -> bool:
        result = comparisons[comparison][metric]
        return bool(
            result["mean_difference"] > 0.0
            and result["paired_bootstrap_95_ci"][0] > 0.0
        )

    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "top2_cosine_beats_clip_with_positive_ci": positive_with_ci(
            "versus_clip_kl", "cosine"
        ),
        "top2_projection_beats_clip_with_positive_ci": positive_with_ci(
            "versus_clip_kl", "oracle_unit_projection"
        ),
        "top2_first_order_beats_clip_with_positive_ci": positive_with_ci(
            "versus_clip_kl", "first_order"
        ),
        "top2_cosine_beats_top1_with_positive_ci": positive_with_ci(
            "versus_top1_set", "cosine"
        ),
        "top2_projection_beats_top1_with_positive_ci": positive_with_ci(
            "versus_top1_set", "oracle_unit_projection"
        ),
        "top2_first_order_beats_top1_with_positive_ci": positive_with_ci(
            "versus_top1_set", "first_order"
        ),
        "class_macro_first_order_delta_vs_clip_positive": (
            macro_first_order_delta_vs_clip > 0.0
        ),
        "car_first_order_delta_vs_clip_nonnegative": (
            hard_class_first_order_delta_vs_clip["car"] >= 0.0
        ),
        "person_first_order_delta_vs_clip_nonnegative": (
            hard_class_first_order_delta_vs_clip["person"] >= 0.0
        ),
        "truck_first_order_delta_vs_clip_nonnegative": (
            hard_class_first_order_delta_vs_clip["truck"] >= 0.0
        ),
        "harmful_fraction_not_above_clip": top2_harmful_pct <= clip_harmful_pct,
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_SET_GRADIENT_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "paired_mean_improvements": "> 0",
            "paired_bootstrap_95_ci_lower_bounds": "> 0",
            "class_macro_first_order_delta_vs_clip": "> 0",
            "car_person_truck_first_order_delta_vs_clip": ">= 0",
            "top2_harmful_fraction_minus_clip": "<= 0",
        },
        "training_authorized": False,
    }
