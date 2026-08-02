"""Pure helpers for a CPU-only VSFOT alignment-direction audit."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logsumexp


def normalize_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must have shape [sample, class]")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    total = matrix.sum(axis=1)
    if np.any(total <= 0.0):
        raise ValueError(f"{name} rows must have positive mass")
    return matrix / total[:, None]


def log_sinkhorn(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    cost: np.ndarray,
    *,
    regularization: float = 0.2,
    iterations: int = 1_000,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compute a stable entropic transport plan in the log domain."""
    source = np.asarray(source_mass, dtype=np.float64)
    target = np.asarray(target_mass, dtype=np.float64)
    values = np.asarray(cost, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("transport marginals must be vectors")
    if values.shape != (source.size, target.size):
        raise ValueError("cost shape must match transport marginals")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or np.any(source <= 0.0)
        or np.any(target <= 0.0)
    ):
        raise ValueError("transport inputs must be finite with positive marginals")
    if regularization <= 0.0 or iterations <= 0 or tolerance <= 0.0:
        raise ValueError("invalid Sinkhorn numerical contract")
    source = source / source.sum()
    target = target / target.sum()
    log_kernel = -values / float(regularization)
    log_source = np.log(source)
    log_target = np.log(target)
    log_u = np.zeros_like(source)
    log_v = np.zeros_like(target)
    converged_at = iterations
    for iteration in range(iterations):
        log_u = log_source - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_target - logsumexp(log_kernel + log_u[:, None], axis=0)
        if iteration % 5 == 4 or iteration == iterations - 1:
            log_plan = log_u[:, None] + log_kernel + log_v[None, :]
            plan = np.exp(log_plan)
            error = max(
                float(np.max(np.abs(plan.sum(axis=1) - source))),
                float(np.max(np.abs(plan.sum(axis=0) - target))),
            )
            if error <= tolerance:
                converged_at = iteration + 1
                break
    else:  # pragma: no cover - loop always exits through finite iteration count
        plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
        error = max(
            float(np.max(np.abs(plan.sum(axis=1) - source))),
            float(np.max(np.abs(plan.sum(axis=0) - target))),
        )
    if not np.isfinite(plan).all():
        raise RuntimeError("Sinkhorn plan is not finite")
    return {
        "plan": plan,
        "max_marginal_error": error,
        "iterations": converged_at,
    }


def clip_kl_feature_descent(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
    classifier_weight: np.ndarray,
) -> np.ndarray:
    """Feature descent of KL(CLIP || task), up to a positive scalar."""
    task = normalize_rows(task_probability, name="task_probability")
    clip = normalize_rows(clip_probability, name="clip_probability")
    weight = np.asarray(classifier_weight, dtype=np.float64)
    if task.shape != clip.shape or weight.shape[0] != task.shape[1]:
        raise ValueError("CLIP-KL feature dimensions do not match")
    result = (clip - task) @ weight
    if not np.isfinite(result).all():
        raise RuntimeError("CLIP-KL feature descent is not finite")
    return result


def vsfot_alignment_feature_descent(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
    task_feature: np.ndarray,
    classifier_weight: np.ndarray,
    class_weight: np.ndarray,
    batch_order: np.ndarray,
    *,
    batch_size: int = 64,
    regularization: float = 0.2,
) -> dict[str, Any]:
    """Replay the public VSFOT CLIP-coupled source-prototype alignment.

    This isolates ``loss_align`` from the public implementation. SimSiam,
    information maximization, CLIP adapters, and every optimizer update are
    deliberately excluded.
    """
    task = normalize_rows(task_probability, name="task_probability")
    clip = normalize_rows(clip_probability, name="clip_probability")
    feature = np.asarray(task_feature, dtype=np.float64)
    weight = np.asarray(classifier_weight, dtype=np.float64)
    class_scale = np.asarray(class_weight, dtype=np.float64)
    order = np.asarray(batch_order, dtype=np.int64)
    sample_count, class_count = task.shape
    if clip.shape != task.shape or feature.shape[0] != sample_count:
        raise ValueError("VSFOT inputs must share the sample dimension")
    if weight.shape != (class_count, feature.shape[1]):
        raise ValueError("classifier weight does not match VSFOT features")
    if class_scale.shape != (class_count,) or np.any(class_scale <= 0.0):
        raise ValueError("class_weight must be a positive class vector")
    if order.shape != (sample_count,) or not np.array_equal(
        np.sort(order), np.arange(sample_count)
    ):
        raise ValueError("batch_order must be a sample permutation")
    if batch_size <= 1:
        raise ValueError("batch_size must exceed one")
    feature_norm = np.linalg.norm(feature, axis=1)
    weight_norm = np.linalg.norm(weight, axis=1)
    if np.any(feature_norm <= 0.0) or np.any(weight_norm <= 0.0):
        raise ValueError("feature and classifier rows must have positive norm")
    feature_unit = feature / feature_norm[:, None]
    weight_unit = weight / weight_norm[:, None]
    cosine = feature_unit @ weight_unit.T

    classification = np.zeros_like(feature)
    prototype = np.zeros_like(feature)
    max_marginal_error = 0.0
    max_sinkhorn_iterations = 0
    for start in range(0, sample_count, batch_size):
        index = order[start : start + batch_size]
        batch_clip = clip[index]
        batch_task = task[index]
        batch_cosine = cosine[index]
        clip_distance = 1.0 - batch_clip.T
        if float(np.max(clip_distance)) <= 0.0:
            raise ValueError("CLIP distance must contain a positive value")
        clip_alpha = 1.0 / float(np.max(clip_distance))
        clip_cost = clip_alpha * clip_distance - np.log(batch_clip.T + 1e-6)
        source_mass = np.maximum(batch_clip.mean(axis=0), 1e-12)
        target_mass = np.full(index.size, 1.0 / index.size, dtype=np.float64)
        sinkhorn = log_sinkhorn(
            source_mass,
            target_mass,
            clip_cost,
            regularization=regularization,
        )
        gamma = sinkhorn["plan"].T
        weighted_gamma = gamma * class_scale[None, :]
        # Exact negative-log-probability derivative with the public code's
        # +1e-6 probability floor kept inside the logarithm.
        adjusted_gamma = weighted_gamma * batch_task / (batch_task + 1e-6)
        adjusted_mass = adjusted_gamma.sum(axis=1)
        logit_descent = adjusted_gamma - adjusted_mass[:, None] * batch_task
        classification[index] = logit_descent @ weight

        feature_distance = 1.0 - batch_cosine
        if float(np.max(feature_distance)) <= 0.0:
            raise ValueError("prototype distance must contain a positive value")
        feature_alpha = 1.0 / float(np.max(feature_distance))
        derivative = (
            weight_unit[None, :, :]
            - batch_cosine[:, :, None] * feature_unit[index, None, :]
        ) / feature_norm[index, None, None]
        prototype[index] = feature_alpha * np.einsum(
            "bc,bcf->bf", weighted_gamma, derivative
        )
        # The public implementation does not detach ``alpha`` in the task
        # feature cost. Reproduce the derivative of 1 / max(1 - cosine),
        # including PyTorch's single argmax subgradient for the maximum.
        max_flat = int(np.argmax(feature_distance))
        max_row, max_class = np.unravel_index(max_flat, feature_distance.shape)
        weighted_distance_sum = float(np.sum(weighted_gamma * feature_distance))
        prototype[index[max_row]] -= (
            weighted_distance_sum
            / float(np.max(feature_distance)) ** 2
            * derivative[max_row, max_class]
        )
        max_marginal_error = max(max_marginal_error, sinkhorn["max_marginal_error"])
        max_sinkhorn_iterations = max(max_sinkhorn_iterations, sinkhorn["iterations"])
    combined = classification + prototype
    if not all(
        np.isfinite(value).all() for value in (classification, prototype, combined)
    ):
        raise RuntimeError("VSFOT alignment descent is not finite")
    return {
        "classification_descent": classification,
        "prototype_descent": prototype,
        "combined_descent": combined,
        "max_sinkhorn_marginal_error": max_marginal_error,
        "max_sinkhorn_iterations": max_sinkhorn_iterations,
    }


def row_unit(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("values must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("direction matrix must be finite")
    norm = np.linalg.norm(matrix, axis=1)
    result = np.zeros_like(matrix)
    nonzero = norm > 0.0
    result[nonzero] = matrix[nonzero] / norm[nonzero, None]
    return result


def row_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_unit = row_unit(first)
    second_unit = row_unit(second)
    if first_unit.shape != second_unit.shape:
        raise ValueError("direction matrices must have matching shape")
    return np.clip(np.einsum("ij,ij->i", first_unit, second_unit), -1.0, 1.0)


def evaluate_vsfot_alignment_gate(
    *,
    input_contract_valid: bool,
    max_sinkhorn_marginal_error: float,
    minimum_replay_median_cosine: float,
    comparisons: dict[str, dict[str, dict[str, Any]]],
    every_replay_conflict_gain_vs_clip_positive: bool,
    candidate_negative_burden: float,
    clip_negative_burden: float,
    group_delta_vs_clip: dict[str, float],
) -> dict[str, Any]:
    """Gate one isolated VSFOT alignment component; never authorize training."""
    if set(comparisons) != {"clip_kl", "transport_classification_only"}:
        raise ValueError("VSFOT comparisons must contain both matched baselines")
    for baseline in comparisons.values():
        if set(baseline) != {"overall", "conflict"}:
            raise ValueError("each VSFOT baseline needs overall and conflict scopes")
        for result in baseline.values():
            interval = result.get("paired_bootstrap_95_ci")
            values = (result.get("mean_difference"), *(interval or ()))
            if len(values) != 3 or not all(np.isfinite(value) for value in values):
                raise ValueError("invalid VSFOT paired comparison")
    if set(group_delta_vs_clip) != {"car", "person", "truck", "other_nine"}:
        raise ValueError("VSFOT group contract changed")

    def positive(result: dict[str, Any]) -> bool:
        return bool(
            result["mean_difference"] > 0.0
            and result["paired_bootstrap_95_ci"][0] > 0.0
        )

    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "sinkhorn_marginal_error_at_most_1e_6": (max_sinkhorn_marginal_error <= 1e-6),
        "minimum_replay_median_direction_cosine_at_least_0_90": (
            minimum_replay_median_cosine >= 0.90
        ),
        "overall_gain_vs_duet_clip_kl_ci_lower_positive": positive(
            comparisons["clip_kl"]["overall"]
        ),
        "conflict_gain_vs_duet_clip_kl_ci_lower_positive": positive(
            comparisons["clip_kl"]["conflict"]
        ),
        "overall_gain_vs_transport_classification_ci_lower_positive": positive(
            comparisons["transport_classification_only"]["overall"]
        ),
        "conflict_gain_vs_transport_classification_ci_lower_positive": positive(
            comparisons["transport_classification_only"]["conflict"]
        ),
        "every_replay_conflict_gain_vs_clip_positive": bool(
            every_replay_conflict_gain_vs_clip_positive
        ),
        "candidate_negative_burden_not_worse": (
            candidate_negative_burden >= clip_negative_burden
        ),
        "car_delta_vs_clip_nonnegative": group_delta_vs_clip["car"] >= 0.0,
        "person_delta_vs_clip_nonnegative": group_delta_vs_clip["person"] >= 0.0,
        "truck_delta_vs_clip_nonnegative": group_delta_vs_clip["truck"] >= 0.0,
        "other_nine_delta_vs_clip_nonnegative": (
            group_delta_vs_clip["other_nine"] >= 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_VSFOT_ALIGNMENT_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "max_sinkhorn_marginal_error": 1e-6,
            "min_replay_median_direction_cosine": 0.90,
            "paired_mean_and_ci_lower": "> 0",
            "candidate_minus_clip_negative_burden": ">= 0",
            "car_person_truck_other_nine_delta_vs_clip": ">= 0",
        },
        "training_authorized": False,
        "proxy_authorized": False,
        "gpu_authorized": False,
    }
