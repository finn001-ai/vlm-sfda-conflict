"""Label-free helpers for auditing a candidate-set target in DUET DVO/TMI."""

from __future__ import annotations

import sys
from typing import Any, Iterable

import numpy as np
import torch


def _probability_matrix(probability: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(f"{name} must have shape [sample, class]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    row_sum = values.sum(axis=1)
    if np.any(row_sum <= 0.0) or not np.allclose(row_sum, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} rows must sum to one")
    return values / row_sum[:, None]


def support_conditioned_mixed_target(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
    active_conflict: np.ndarray,
) -> dict[str, np.ndarray]:
    """Condition DUET's mixed DVO target on a top-2 union.

    Only active task/CLIP top-1 conflicts are changed.  The original mixed
    top-1 is explicitly retained in the support, so the intervention changes
    distributional supervision but never changes the target's hard class.
    """
    task = _probability_matrix(task_probability, name="task_probability")
    clip = _probability_matrix(clip_probability, name="clip_probability")
    if task.shape != clip.shape:
        raise ValueError("task and CLIP probabilities must have matching shapes")
    active = np.asarray(active_conflict, dtype=bool)
    if active.shape != (task.shape[0],):
        raise ValueError("active_conflict must contain one value per sample")
    if not np.any(active):
        raise ValueError("active_conflict must select at least one sample")

    mixed = 0.5 * (task + clip)
    mixed /= mixed.sum(axis=1, keepdims=True)
    support = np.zeros(task.shape, dtype=bool)
    row = np.arange(task.shape[0])[:, None]
    task_top2 = np.argsort(-task, axis=1, kind="stable")[:, :2]
    clip_top2 = np.argsort(-clip, axis=1, kind="stable")[:, :2]
    support[row, task_top2] = True
    support[row, clip_top2] = True
    mixed_top1 = mixed.argmax(axis=1)
    support[np.arange(task.shape[0]), mixed_top1] = True

    candidate = mixed.copy()
    active_supported = mixed[active] * support[active]
    retained_mass = active_supported.sum(axis=1)
    if np.any(retained_mass <= 0.0):
        raise RuntimeError("candidate support retained no mixed-target mass")
    candidate[active] = active_supported / retained_mass[:, None]
    if not np.array_equal(candidate.argmax(axis=1), mixed_top1):
        raise RuntimeError("support conditioning changed the mixed target top-1")
    if not np.allclose(candidate.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("candidate target rows must sum to one")
    if not np.array_equal(candidate[~active], mixed[~active]):
        raise RuntimeError("candidate changed a non-conflict target")
    return {
        "baseline_probability": mixed,
        "candidate_probability": candidate,
        "support": support,
        "support_size": support.sum(axis=1),
        "retained_mass": retained_mass,
        "mixed_top1": mixed_top1,
        "task_top2": task_top2,
        "clip_top2": clip_top2,
    }


def _dynamic_q_entropy(probability: torch.Tensor) -> float:
    marginal = probability.mean(dim=0)
    entropy = -(marginal * torch.log(marginal + 1e-9)).sum()
    # Match ``src/utils/IID_losses.py``: its denominator is explicitly a
    # float32 tensor even when an audit uses float64 proxy logits.
    maximum = torch.log(
        torch.tensor(marginal.numel(), dtype=torch.float32, device=probability.device)
    )
    return float((entropy / maximum).detach().cpu().item())


def _tsallis_log(probability: torch.Tensor, q_value: float) -> torch.Tensor:
    epsilon = sys.float_info.epsilon
    values = torch.clamp(probability, min=epsilon)
    if abs(q_value - 1.0) < 1e-12:
        return torch.log(values)
    return (values ** (1.0 - q_value) - 1.0) / (1.0 - q_value)


def _fixed_q_tmi_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    q_value: float,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be same-shaped matrices")
    joint = prediction.unsqueeze(2) * target.unsqueeze(1)
    joint = joint.sum(dim=0)
    joint = 0.5 * (joint + joint.t())
    joint = joint / joint.sum()
    epsilon = sys.float_info.epsilon
    joint = torch.clamp(joint, min=epsilon)
    marginal_i = torch.clamp(joint.sum(dim=1).view(-1, 1).expand_as(joint), min=epsilon)
    marginal_j = torch.clamp(joint.sum(dim=0).view(1, -1).expand_as(joint), min=epsilon)
    loss = -joint * (
        _tsallis_log(joint, q_value)
        - _tsallis_log(marginal_j, q_value)
        - _tsallis_log(marginal_i, q_value)
    )
    return loss.sum()


def tmi_logit_descent_replays(
    clip_probability: np.ndarray,
    target_probability: np.ndarray,
    *,
    permutation_seeds: Iterable[int],
    batch_size: int = 64,
    initial_q: float = 1.05,
    beta: float = 0.99,
) -> dict[str, np.ndarray]:
    """Replay exact output-level TMI gradients across fixed batch orders.

    No image or model is loaded.  The supplied CLIP probabilities are treated
    as detached logits through ``log(p)``; only those proxy logits receive
    autograd gradients.  This reproduces the released TMI joint objective and
    q update at the output level.
    """
    clip = _probability_matrix(clip_probability, name="clip_probability")
    target = _probability_matrix(target_probability, name="target_probability")
    if clip.shape != target.shape:
        raise ValueError("CLIP and target probabilities must have matching shapes")
    seeds = tuple(int(seed) for seed in permutation_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("permutation_seeds must be non-empty and unique")
    if batch_size <= 1:
        raise ValueError("batch_size must be greater than one")
    if not np.isfinite(initial_q) or not np.isfinite(beta) or not 0.0 <= beta < 1.0:
        raise ValueError("initial_q and beta must be finite; beta must be in [0, 1)")

    replay = np.zeros((len(seeds), *clip.shape), dtype=np.float64)
    final_q = np.zeros(len(seeds), dtype=np.float64)
    for replay_index, seed in enumerate(seeds):
        order = np.random.default_rng(seed).permutation(clip.shape[0])
        q_value = float(initial_q)
        for start in range(0, order.size, batch_size):
            batch = order[start : start + batch_size]
            clip_batch = torch.as_tensor(clip[batch], dtype=torch.float64)
            target_batch = torch.as_tensor(target[batch], dtype=torch.float64)
            logits = (
                torch.log(torch.clamp(clip_batch, min=torch.finfo(torch.float64).tiny))
                .detach()
                .requires_grad_(True)
            )
            prediction = torch.softmax(logits, dim=1)
            q_value = beta * q_value + (1.0 - beta) * _dynamic_q_entropy(prediction)
            loss = _fixed_q_tmi_loss(prediction, target_batch, q_value)
            gradient = torch.autograd.grad(loss, logits, create_graph=False)[0]
            descent = -gradient.detach().cpu().numpy()
            if not np.isfinite(descent).all():
                raise RuntimeError("TMI replay produced non-finite logit descent")
            if not np.allclose(descent.sum(axis=1), 0.0, atol=1e-10, rtol=1e-10):
                raise RuntimeError("TMI logit descent must sum to zero per row")
            replay[replay_index, batch] = descent
        final_q[replay_index] = q_value
    return {
        "descent_by_replay": replay,
        "mean_descent": replay.mean(axis=0),
        "final_q": final_q,
        "permutation_seeds": np.asarray(seeds, dtype=np.int64),
    }


def evaluate_dvo_candidate_target_gate(
    *,
    input_contract_valid: bool,
    target_top1_unchanged: bool,
    mean_retained_mass: float,
    mean_support_size: float,
    oracle_candidate_coverage_pct: float,
    comparisons: dict[str, dict[str, Any]],
    minimum_replay_first_order_delta: float,
    macro_first_order_delta: float,
    hard_class_first_order_delta: dict[str, float],
    other_nine_first_order_delta: float,
    candidate_negative_burden: float,
    baseline_negative_burden: float,
    max_class_mass_shift_pp: float,
) -> dict[str, Any]:
    """Apply the fixed no-training gate for the DVO candidate target."""
    required_metrics = ("cosine", "oracle_unit_projection", "first_order")
    hard_classes = ("car", "person", "truck")
    if set(comparisons) != set(required_metrics):
        raise ValueError("comparisons must contain cosine, projection, and first-order")
    if set(hard_class_first_order_delta) != set(hard_classes):
        raise ValueError("hard-class deltas must contain car, person, and truck")
    numeric = (
        mean_retained_mass,
        mean_support_size,
        oracle_candidate_coverage_pct,
        minimum_replay_first_order_delta,
        macro_first_order_delta,
        other_nine_first_order_delta,
        candidate_negative_burden,
        baseline_negative_burden,
        max_class_mass_shift_pp,
        *hard_class_first_order_delta.values(),
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("gate metrics must be finite")
    for metric in required_metrics:
        result = comparisons[metric]
        interval = result.get("paired_bootstrap_95_ci")
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or not all(
                np.isfinite(value)
                for value in (result.get("mean_difference"), *interval)
            )
        ):
            raise ValueError(f"invalid comparison for {metric}")

    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "mixed_target_top1_unchanged": bool(target_top1_unchanged),
        "mean_retained_mass_at_least_90pct": mean_retained_mass >= 0.90,
        "mean_support_size_at_most_4": mean_support_size <= 4.0,
        "oracle_candidate_coverage_at_least_90pct": (
            oracle_candidate_coverage_pct >= 90.0
        ),
        "cosine_gain_ci_lower_positive": (
            comparisons["cosine"]["paired_bootstrap_95_ci"][0] > 0.0
        ),
        "projection_gain_ci_lower_positive": (
            comparisons["oracle_unit_projection"]["paired_bootstrap_95_ci"][0] > 0.0
        ),
        "first_order_gain_ci_lower_positive": (
            comparisons["first_order"]["paired_bootstrap_95_ci"][0] > 0.0
        ),
        "every_replay_first_order_delta_positive": (
            minimum_replay_first_order_delta > 0.0
        ),
        "class_macro_first_order_delta_positive": macro_first_order_delta > 0.0,
        "car_first_order_delta_nonnegative": (
            hard_class_first_order_delta["car"] >= 0.0
        ),
        "person_first_order_delta_nonnegative": (
            hard_class_first_order_delta["person"] >= 0.0
        ),
        "truck_first_order_delta_nonnegative": (
            hard_class_first_order_delta["truck"] >= 0.0
        ),
        "other_nine_first_order_delta_nonnegative": (
            other_nine_first_order_delta >= 0.0
        ),
        "negative_burden_not_worse": (
            candidate_negative_burden <= baseline_negative_burden
        ),
        "max_class_mass_shift_at_most_1pp": max_class_mass_shift_pp <= 1.0,
    }
    passed = all(checks.values())
    return {
        "decision": "PASS_DVO_CANDIDATE_TARGET_PREFLIGHT" if passed else "REJECT",
        "checks": checks,
        "thresholds": {
            "min_mean_retained_mass": 0.90,
            "max_mean_support_size": 4.0,
            "min_oracle_candidate_coverage_pct": 90.0,
            "paired_metric_ci_lower": "> 0",
            "every_replay_first_order_delta": "> 0",
            "class_macro_first_order_delta": "> 0",
            "car_person_truck_other9_first_order_delta": ">= 0",
            "candidate_negative_burden": "<= baseline",
            "max_class_mass_shift_pp": 1.0,
        },
        "training_authorized": False,
    }
