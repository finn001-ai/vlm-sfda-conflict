"""Label-free helpers for auditing hidden errors inside DUET agreements."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.candidate_set_audit import stable_topk


def agreement_rank_residual(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    """Measure bidirectional disagreement between task/CLIP runner-up classes.

    The score is non-zero only when both models share top-1 but choose different
    runner-up classes.  It is the geometric mean of their opposing pairwise
    probability margins, so one model cannot create a large residual alone.
    """
    task = np.asarray(task_probability, dtype=np.float64)
    clip = np.asarray(clip_probability, dtype=np.float64)
    if task.shape != clip.shape:
        raise ValueError("task and CLIP probability shapes must match")
    task_top2 = stable_topk(task, 2)
    clip_top2 = stable_topk(clip, 2)
    common_top1 = task_top2[:, 0] == clip_top2[:, 0]
    runner_up_disagreement = common_top1 & (task_top2[:, 1] != clip_top2[:, 1])
    row = np.arange(task.shape[0])
    task_margin = (
        task[row, task_top2[:, 1]] - task[row, clip_top2[:, 1]]
    )
    clip_margin = (
        clip[row, clip_top2[:, 1]] - clip[row, task_top2[:, 1]]
    )
    task_margin = np.maximum(task_margin, 0.0)
    clip_margin = np.maximum(clip_margin, 0.0)
    score = np.sqrt(task_margin * clip_margin)
    score[~runner_up_disagreement] = 0.0
    return {
        "task_top2": task_top2,
        "clip_top2": clip_top2,
        "common_top1": common_top1,
        "runner_up_disagreement": runner_up_disagreement,
        "task_opposing_margin": task_margin,
        "clip_opposing_margin": clip_margin,
        "rank_residual": score,
    }


def select_class_balanced_fraction(
    values: np.ndarray,
    eligible: np.ndarray,
    group: np.ndarray,
    *,
    fraction: float,
    largest: bool,
) -> dict[str, Any]:
    """Select an exact rank fraction independently inside every eligible group."""
    score = np.asarray(values, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    labels = np.asarray(group, dtype=np.int64)
    if score.ndim != 1 or mask.shape != score.shape or labels.shape != score.shape:
        raise ValueError("values, eligible, and group must be same-shaped vectors")
    if not np.isfinite(score).all():
        raise ValueError("selection values must be finite")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    if np.any(labels < 0):
        raise ValueError("group indices must be non-negative")

    selected = np.zeros(score.size, dtype=bool)
    counts: dict[int, int] = {}
    for class_index in np.unique(labels[mask]):
        indices = np.flatnonzero(mask & (labels == class_index))
        count = max(1, int(np.ceil(indices.size * fraction)))
        # lexsort makes sample position the deterministic secondary key.
        primary = -score[indices] if largest else score[indices]
        order = np.lexsort((indices, primary))
        chosen = indices[order[:count]]
        selected[chosen] = True
        counts[int(class_index)] = int(count)
    return {"selected": selected, "counts_by_group": counts}


def select_matched_counts(
    values: np.ndarray,
    eligible: np.ndarray,
    group: np.ndarray,
    counts_by_group: dict[int, int],
    *,
    largest: bool,
) -> np.ndarray:
    """Select fixed per-group counts for a label-free matched comparator."""
    score = np.asarray(values, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    labels = np.asarray(group, dtype=np.int64)
    if score.ndim != 1 or mask.shape != score.shape or labels.shape != score.shape:
        raise ValueError("values, eligible, and group must be same-shaped vectors")
    if not np.isfinite(score).all():
        raise ValueError("selection values must be finite")
    selected = np.zeros(score.size, dtype=bool)
    for class_index, count in counts_by_group.items():
        indices = np.flatnonzero(mask & (labels == int(class_index)))
        if not 0 <= int(count) <= indices.size:
            raise ValueError("matched group count is outside the eligible group")
        primary = -score[indices] if largest else score[indices]
        order = np.lexsort((indices, primary))
        selected[indices[order[: int(count)]]] = True
    return selected


def paired_selection_precision_bootstrap_ci(
    candidate_selected: np.ndarray,
    baseline_selected: np.ndarray,
    wrong: np.ndarray,
    eligible: np.ndarray,
    *,
    repeats: int = 2_000,
    seed: int = 2_020,
    batch_size: int = 50,
) -> tuple[float, float]:
    """CI for candidate-minus-baseline error precision at matched coverage."""
    candidate = np.asarray(candidate_selected, dtype=bool)
    baseline = np.asarray(baseline_selected, dtype=bool)
    error = np.asarray(wrong, dtype=bool)
    admitted = np.asarray(eligible, dtype=bool)
    if not (candidate.shape == baseline.shape == error.shape == admitted.shape):
        raise ValueError("paired selection arrays must have matching shapes")
    candidate = candidate[admitted]
    baseline = baseline[admitted]
    error = error[admitted]
    if candidate.size == 0 or candidate.sum() == 0:
        raise ValueError("candidate selection must be non-empty")
    if candidate.sum() != baseline.sum():
        raise ValueError("candidate and baseline coverage must match")
    if repeats <= 0 or batch_size <= 0:
        raise ValueError("bootstrap settings must be positive")

    difference = (candidate.astype(np.float64) - baseline.astype(np.float64)) * error
    scale = float(candidate.size / candidate.sum() * 100.0)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats, dtype=np.float64)
    written = 0
    while written < repeats:
        current = min(batch_size, repeats - written)
        indices = rng.integers(0, difference.size, size=(current, difference.size))
        bootstrap[written : written + current] = (
            difference[indices].mean(axis=1) * scale
        )
        written += current
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def evaluate_agreement_rank_residual_gate(
    *,
    input_contract_valid: bool,
    baseline_reproduced: bool,
    selected_fraction_pct: float,
    error_enrichment: float,
    retained_accuracy_gain_pp: float,
    comparisons: dict[str, dict[str, Any]],
    car_wrong_captures: dict[str, int],
    truck_wrong_captures: dict[str, int],
    noncar_wrong_captures: dict[str, int],
) -> dict[str, Any]:
    """Apply the predeclared offline gate; passing never authorizes training."""
    required = {
        "task_confidence",
        "clip_confidence",
        "arithmetic_confidence",
        "rms_confidence",
    }
    if set(comparisons) != required:
        raise ValueError("all three matched confidence comparators are required")
    for name in required:
        interval = comparisons[name].get("paired_bootstrap_95_ci_pp")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"missing paired interval for {name}")
    for captures in (car_wrong_captures, truck_wrong_captures, noncar_wrong_captures):
        if set(captures) != {"candidate", *required}:
            raise ValueError("capture dictionaries must contain candidate and baselines")

    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "cycle1_duet_agreement_baseline_reproduced": bool(baseline_reproduced),
        "selection_coverage_between_5_and_10pct": (
            5.0 <= selected_fraction_pct <= 10.1
        ),
        "selected_error_enrichment_at_least_2x": error_enrichment >= 2.0,
        "beats_all_matched_confidence_error_captures": all(
            comparisons[name]["captured_error_gain"] > 0 for name in required
        ),
        "all_paired_precision_ci_lowers_positive": all(
            comparisons[name]["paired_bootstrap_95_ci_pp"][0] > 0.0
            for name in required
        ),
        "retained_pseudo_label_accuracy_gain_at_least_0_25pp": (
            retained_accuracy_gain_pp >= 0.25
        ),
        "car_wrong_capture_nonworse_than_all_confidence_baselines": all(
            car_wrong_captures["candidate"] >= car_wrong_captures[name]
            for name in required
        ),
        "truck_wrong_capture_nonworse_than_all_confidence_baselines": all(
            truck_wrong_captures["candidate"] >= truck_wrong_captures[name]
            for name in required
        ),
        "noncar_wrong_capture_nonworse_than_all_confidence_baselines": all(
            noncar_wrong_captures["candidate"] >= noncar_wrong_captures[name]
            for name in required
        ),
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_AGREEMENT_RANK_RESIDUAL_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "selection_fraction_by_pseudo_class": 0.10,
            "min_selected_fraction_pct": 5.0,
            "max_selected_fraction_pct": 10.1,
            "min_error_enrichment": 2.0,
            "min_retained_accuracy_gain_pp": 0.25,
            "paired_precision_ci_lower": "> 0",
        },
        "training_authorized": False,
    }
