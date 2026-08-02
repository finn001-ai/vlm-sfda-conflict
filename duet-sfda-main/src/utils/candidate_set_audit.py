"""Label-free candidate-set construction and oracle-only coverage gates."""

from __future__ import annotations

from typing import Any

import numpy as np


def stable_topk(probability: np.ndarray, k: int) -> np.ndarray:
    """Return deterministic top-k class indices, breaking ties by class index."""
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probability must have shape [sample, class]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("probability must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("probability rows must sum to one")
    if not 1 <= k <= values.shape[1]:
        raise ValueError("k must be inside the class range")
    return np.argsort(-values, axis=1, kind="stable")[:, :k]


def union_candidate_mask(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
    *,
    k: int,
) -> dict[str, np.ndarray]:
    """Build the union of task and CLIP top-k classes for every sample."""
    task = np.asarray(task_probability, dtype=np.float64)
    clip = np.asarray(clip_probability, dtype=np.float64)
    if task.shape != clip.shape:
        raise ValueError("task and CLIP probability shapes must match")
    task_topk = stable_topk(task, k)
    clip_topk = stable_topk(clip, k)
    mask = np.zeros(task.shape, dtype=bool)
    row = np.arange(task.shape[0])[:, None]
    mask[row, task_topk] = True
    mask[row, clip_topk] = True
    return {
        "task_topk": task_topk,
        "clip_topk": clip_topk,
        "union_mask": mask,
        "set_size": mask.sum(axis=1),
    }


def candidate_coverage(mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return per-sample oracle coverage for a previously locked set mask."""
    candidate = np.asarray(mask, dtype=bool)
    label = np.asarray(labels, dtype=np.int64)
    if candidate.ndim != 2 or candidate.shape[0] == 0:
        raise ValueError("mask must have shape [sample, class]")
    if label.shape != (candidate.shape[0],):
        raise ValueError("labels must contain one class per sample")
    if np.any(label < 0) or np.any(label >= candidate.shape[1]):
        raise ValueError("label is outside the class range")
    return candidate[np.arange(label.size), label]


def evaluate_candidate_set_gate(
    *,
    input_contract_valid: bool,
    top2_coverage_pct: float,
    recovered_top1_misses_pct: float,
    minimum_class_coverage_pct: float,
    car_coverage_pct: float,
    truck_coverage_pct: float,
    mean_set_size: float,
    min_top2_coverage_pct: float = 90.0,
    min_recovered_top1_misses_pct: float = 60.0,
    min_per_class_coverage_pct: float = 85.0,
    min_car_truck_coverage_pct: float = 90.0,
    max_mean_set_size: float = 3.5,
) -> dict[str, Any]:
    """Apply predeclared gates; passing never authorizes training."""
    numeric = (
        top2_coverage_pct,
        recovered_top1_misses_pct,
        minimum_class_coverage_pct,
        car_coverage_pct,
        truck_coverage_pct,
        mean_set_size,
        min_top2_coverage_pct,
        min_recovered_top1_misses_pct,
        min_per_class_coverage_pct,
        min_car_truck_coverage_pct,
        max_mean_set_size,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate values must be finite")
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "top2_union_coverage_at_least_90pct": (
            top2_coverage_pct >= min_top2_coverage_pct
        ),
        "top2_recovers_at_least_60pct_of_top1_misses": (
            recovered_top1_misses_pct >= min_recovered_top1_misses_pct
        ),
        "every_class_top2_coverage_at_least_85pct": (
            minimum_class_coverage_pct >= min_per_class_coverage_pct
        ),
        "car_top2_coverage_at_least_90pct": (
            car_coverage_pct >= min_car_truck_coverage_pct
        ),
        "truck_top2_coverage_at_least_90pct": (
            truck_coverage_pct >= min_car_truck_coverage_pct
        ),
        "mean_candidate_set_size_at_most_3_5": (mean_set_size <= max_mean_set_size),
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_CANDIDATE_SET_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "min_top2_union_coverage_pct": min_top2_coverage_pct,
            "min_recovered_top1_misses_pct": min_recovered_top1_misses_pct,
            "min_per_class_top2_coverage_pct": min_per_class_coverage_pct,
            "min_car_truck_top2_coverage_pct": min_car_truck_coverage_pct,
            "max_mean_candidate_set_size": max_mean_set_size,
        },
        "training_authorized": False,
    }
