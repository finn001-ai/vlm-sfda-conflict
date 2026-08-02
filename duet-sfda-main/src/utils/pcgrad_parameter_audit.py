"""Pure helpers for the locked VisDA PCGrad parameter preflight."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


AUDIT_BATCH_COUNT = 10
AUDIT_BATCH_SIZE = 64
CONFLICTS_PER_BATCH = 10
AUDITED_CONFLICTS = AUDIT_BATCH_COUNT * CONFLICTS_PER_BATCH


def _evenly_spaced(pool: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(pool, dtype=np.int64)
    if values.ndim != 1 or values.size < count or count <= 0:
        raise ValueError("pool must be one-dimensional and large enough")
    if np.unique(values).size != values.size:
        raise ValueError("pool indices must be unique")
    positions = np.floor(
        (np.arange(count, dtype=np.float64) + 0.5) * values.size / count
    ).astype(np.int64)
    selected = values[positions]
    if np.unique(selected).size != count:
        raise RuntimeError("evenly spaced selection produced duplicates")
    return selected


def build_locked_parameter_audit_batches(
    conflict_indices: np.ndarray,
    admitted_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ten fixed 64-row batches without labels or fitted thresholds."""
    conflict = np.sort(np.asarray(conflict_indices, dtype=np.int64))
    admitted = np.sort(np.asarray(admitted_indices, dtype=np.int64))
    context_per_batch = AUDIT_BATCH_SIZE - CONFLICTS_PER_BATCH
    selected_conflict = _evenly_spaced(conflict, AUDITED_CONFLICTS)
    selected_admitted = _evenly_spaced(
        admitted, AUDIT_BATCH_COUNT * context_per_batch
    )
    conflict_batches = selected_conflict.reshape(
        AUDIT_BATCH_COUNT, CONFLICTS_PER_BATCH
    )
    admitted_batches = selected_admitted.reshape(
        AUDIT_BATCH_COUNT, context_per_batch
    )
    batches = np.concatenate((conflict_batches, admitted_batches), axis=1)
    conflict_position = np.zeros_like(batches, dtype=bool)
    conflict_position[:, :CONFLICTS_PER_BATCH] = True
    if np.unique(batches).size != batches.size:
        raise RuntimeError("audit batches must not reuse target samples")
    return batches, conflict_position


def symmetric_pcgrad_output_correction(
    consistency_descent: torch.Tensor,
    clip_descent: torch.Tensor,
    conflict_mask: torch.Tensor,
    *,
    epsilon: float = 1e-15,
) -> dict[str, torch.Tensor]:
    """Return row-wise symmetric PCGrad correction in joint logit space."""
    first = consistency_descent
    second = clip_descent
    mask = conflict_mask.bool()
    if first.ndim != 2 or first.shape != second.shape or first.shape[0] == 0:
        raise ValueError("descent tensors must be non-empty same-shaped matrices")
    if mask.shape != (first.shape[0],):
        raise ValueError("conflict_mask must contain one value per row")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("descent tensors must be finite")
    dot = torch.sum(first * second, dim=1)
    first_norm_sq = torch.sum(first.square(), dim=1)
    second_norm_sq = torch.sum(second.square(), dim=1)
    active = (
        mask
        & (dot < 0.0)
        & (first_norm_sq > epsilon)
        & (second_norm_sq > epsilon)
    )
    first_projected = first.clone()
    second_projected = second.clone()
    if bool(active.any()):
        first_projected[active] -= (
            dot[active] / second_norm_sq[active]
        ).unsqueeze(1) * second[active]
        second_projected[active] -= (
            dot[active] / first_norm_sq[active]
        ).unsqueeze(1) * first[active]
    baseline = first + second
    candidate = first_projected + second_projected
    denominator = torch.sqrt(first_norm_sq * second_norm_sq)
    cosine = torch.zeros_like(dot)
    nonzero = denominator > epsilon
    cosine[nonzero] = dot[nonzero] / denominator[nonzero]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "correction": candidate - baseline,
        "active": active,
        "component_dot": dot,
        "component_cosine": cosine.clamp(-1.0, 1.0),
    }


def paired_mean_bootstrap_ci(
    values: np.ndarray,
    *,
    repeats: int = 2_000,
    seed: int = 2_020,
) -> tuple[float, float]:
    differences = np.asarray(values, dtype=np.float64)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("values must be a non-empty vector")
    if not np.isfinite(differences).all() or repeats <= 0:
        raise ValueError("bootstrap inputs must be finite and valid")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, differences.size, size=(repeats, differences.size)
    )
    bootstrap = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return float(low), float(high)


def paired_metric_summary(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if candidate_values.shape != baseline_values.shape or candidate_values.ndim != 1:
        raise ValueError("paired metrics must be same-shaped vectors")
    difference = candidate_values - baseline_values
    return {
        "samples": int(difference.size),
        "baseline_mean": float(baseline_values.mean()),
        "candidate_mean": float(candidate_values.mean()),
        "mean_difference": float(difference.mean()),
        "paired_bootstrap_95_ci": list(
            paired_mean_bootstrap_ci(difference, seed=seed)
        ),
    }


def negative_burden(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a finite non-empty vector")
    return float(np.minimum(array, 0.0).mean())


def evaluate_exact_parameter_gate(
    *,
    input_contract_valid: bool,
    cycle1_max_accuracy_error_pp: float,
    audited_conflict_coverage_pct: float,
    output_active_coverage_pct: float,
    comparisons: dict[str, dict[str, Any]],
    baseline_negative_burden: float,
    candidate_negative_burden: float,
    helpful_retention_pct: float,
    mean_norm_ratio: float,
    positive_batch_fraction_pct: float,
    group_first_order_delta: dict[str, float],
) -> dict[str, Any]:
    """Gate one exact matched proxy; passing never authorizes a full run."""
    required_metrics = ("cosine", "oracle_unit_projection", "first_order")
    for name in required_metrics:
        result = comparisons.get(name, {})
        interval = result.get("paired_bootstrap_95_ci")
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or not all(
                np.isfinite(value)
                for value in (result.get("mean_difference"), *interval)
            )
        ):
            raise ValueError(f"invalid paired parameter metric: {name}")
    if set(group_first_order_delta) != {"car", "person", "truck", "other_nine"}:
        raise ValueError("group deltas must contain car, person, truck, other_nine")
    numeric = (
        cycle1_max_accuracy_error_pp,
        audited_conflict_coverage_pct,
        output_active_coverage_pct,
        baseline_negative_burden,
        candidate_negative_burden,
        helpful_retention_pct,
        mean_norm_ratio,
        positive_batch_fraction_pct,
        *group_first_order_delta.values(),
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("parameter gate values must be finite")

    def positive_with_ci(name: str) -> bool:
        result = comparisons[name]
        return bool(
            result["mean_difference"] > 0.0
            and result["paired_bootstrap_95_ci"][0] > 0.0
        )

    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "pure_duet_cycle1_replay_error_at_most_0_10pp": (
            cycle1_max_accuracy_error_pp <= 0.10
        ),
        "audits_at_least_5pct_of_unresolved_conflicts": (
            audited_conflict_coverage_pct >= 5.0
        ),
        "output_pcgrad_active_coverage_at_least_5pct": (
            output_active_coverage_pct >= 5.0
        ),
        "parameter_cosine_gain_ci_lower_positive": positive_with_ci("cosine"),
        "parameter_projection_gain_ci_lower_positive": positive_with_ci(
            "oracle_unit_projection"
        ),
        "parameter_first_order_gain_ci_lower_positive": positive_with_ci(
            "first_order"
        ),
        "parameter_negative_burden_not_worse": (
            candidate_negative_burden >= baseline_negative_burden
        ),
        "parameter_helpful_retention_at_least_99pct": (
            helpful_retention_pct >= 99.0
        ),
        "parameter_mean_norm_inflation_at_most_1_5x": mean_norm_ratio <= 1.5,
        "at_least_80pct_batches_have_positive_first_order_delta": (
            positive_batch_fraction_pct >= 80.0
        ),
        "car_parameter_first_order_delta_nonnegative": (
            group_first_order_delta["car"] >= 0.0
        ),
        "person_parameter_first_order_delta_nonnegative": (
            group_first_order_delta["person"] >= 0.0
        ),
        "truck_parameter_first_order_delta_nonnegative": (
            group_first_order_delta["truck"] >= 0.0
        ),
        "other_nine_parameter_first_order_delta_nonnegative": (
            group_first_order_delta["other_nine"] >= 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_EXACT_PARAMETER_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "cycle1_max_accuracy_error_pp": "<= 0.10",
            "audited_conflict_coverage_pct": ">= 5",
            "output_active_coverage_pct": ">= 5",
            "paired_cosine_projection_first_order_mean_and_ci_lower": "> 0",
            "candidate_negative_burden_minus_baseline": ">= 0",
            "helpful_retention_pct": ">= 99",
            "candidate_to_baseline_mean_norm_ratio": "<= 1.5",
            "positive_batch_fraction_pct": ">= 80",
            "car_person_truck_other_nine_first_order_delta": ">= 0",
        },
        "matched_proxy_authorized": bool(passed),
        "full_training_authorized": False,
        "seed_sweep_authorized": False,
    }
