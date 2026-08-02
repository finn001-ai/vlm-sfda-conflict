"""Pure helpers for the agreement-only transport full-gradient preflight."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.conflict_pcgrad_audit import duet_output_descent_components


def agreement_transport_joint_descents(
    weak_probability: np.ndarray,
    strong_probability: np.ndarray,
    clip_probability: np.ndarray,
    task_label: np.ndarray,
    agreement_mask: np.ndarray,
    transport_probability: np.ndarray,
    batch_order: np.ndarray,
    *,
    batch_size: int = 64,
    consistency_weight: float = 0.2,
    clip_weight: float = 0.4,
    pseudo_ce_weight: float = 0.4,
    probability_floor: float = float(np.finfo(np.float32).tiny),
) -> dict[str, np.ndarray]:
    """Replay complete DUET joint logit descents under three KL targets."""
    weak = np.asarray(weak_probability, dtype=np.float64)
    strong = np.asarray(strong_probability, dtype=np.float64)
    clip = np.asarray(clip_probability, dtype=np.float64)
    transport = np.asarray(transport_probability, dtype=np.float64)
    label = np.asarray(task_label, dtype=np.int64)
    agreement = np.asarray(agreement_mask, dtype=bool)
    order = np.asarray(batch_order, dtype=np.int64)
    if not (weak.shape == strong.shape == clip.shape == transport.shape):
        raise ValueError("all probability matrices must have matching shape")
    sample_count, class_count = weak.shape
    if label.shape != (sample_count,) or agreement.shape != (sample_count,):
        raise ValueError("labels and agreement mask must match sample count")
    if order.shape != (sample_count,) or not np.array_equal(
        np.sort(order), np.arange(sample_count)
    ):
        raise ValueError("batch_order must be a sample permutation")
    if np.any(label < 0) or np.any(label >= class_count):
        raise ValueError("task label outside class range")
    if batch_size <= 1:
        raise ValueError("batch_size must exceed one")

    baseline = np.zeros((sample_count, 2 * class_count), dtype=np.float64)
    candidate = np.zeros_like(baseline)
    duplicate_ce = np.zeros_like(baseline)
    one_hot = np.eye(class_count, dtype=np.float64)[label]
    for start in range(0, sample_count, batch_size):
        index = order[start : start + batch_size]
        current_size = index.size
        admitted = agreement[index]
        components = duet_output_descent_components(
            weak[index],
            strong[index],
            clip[index],
            consistency_weight=consistency_weight,
            clip_weight=clip_weight,
            probability_floor=probability_floor,
        )
        consistency_weak = components["consistency_weak"] / current_size
        consistency_strong = components["consistency_strong"] / current_size
        clip_weak = components["clip_weak"] / current_size
        ce_weak = np.zeros_like(clip_weak)
        if admitted.any():
            ce_weak[admitted] = (
                float(pseudo_ce_weight)
                / int(admitted.sum())
                * (one_hot[index][admitted] - weak[index][admitted])
            )

        baseline_weak = consistency_weak + clip_weak + ce_weak
        routed_transport = clip[index].copy()
        routed_transport[admitted] = transport[index][admitted]
        candidate_weak = (
            consistency_weak
            + float(clip_weight) / current_size * (routed_transport - weak[index])
            + ce_weak
        )
        routed_hard = clip[index].copy()
        routed_hard[admitted] = one_hot[index][admitted]
        duplicate_weak = (
            consistency_weak
            + float(clip_weight) / current_size * (routed_hard - weak[index])
            + ce_weak
        )
        baseline[index] = np.concatenate((baseline_weak, consistency_strong), axis=1)
        candidate[index] = np.concatenate((candidate_weak, consistency_strong), axis=1)
        duplicate_ce[index] = np.concatenate(
            (duplicate_weak, consistency_strong), axis=1
        )
    if not all(
        np.isfinite(value).all() for value in (baseline, candidate, duplicate_ce)
    ):
        raise RuntimeError("full-gradient descents must be finite")
    return {
        "duet_joint": baseline,
        "agreement_transport_joint": candidate,
        "duplicate_hard_ce_joint": duplicate_ce,
    }


def evaluate_agreement_transport_gate(
    *,
    input_contract_valid: bool,
    max_sinkhorn_marginal_error: float,
    minimum_target_replay_median_cosine: float,
    mean_transport_ce_component_cosine: float,
    comparisons: dict[str, dict[str, dict[str, Any]]],
    every_replay_agreement_first_order_gain_positive: dict[str, bool],
    candidate_negative_burden: float,
    strongest_control_negative_burden: float,
    candidate_to_strongest_mean_norm_ratio: float,
    group_first_order_delta_vs_strongest: dict[str, float],
) -> dict[str, Any]:
    required_controls = {"original_duet", "duplicate_hard_ce"}
    required_scopes = {"overall", "agreement"}
    required_metrics = {"first_order", "cosine"}
    if set(comparisons) != required_controls:
        raise ValueError("agreement transport controls changed")
    for control in comparisons.values():
        if set(control) != required_scopes:
            raise ValueError("agreement transport scopes changed")
        for scope in control.values():
            if set(scope) != required_metrics:
                raise ValueError("agreement transport metrics changed")
            for result in scope.values():
                interval = result.get("paired_bootstrap_95_ci")
                if not (
                    isinstance(interval, (list, tuple))
                    and len(interval) == 2
                    and all(
                        np.isfinite(value)
                        for value in (result.get("mean_difference"), *interval)
                    )
                ):
                    raise ValueError("invalid agreement transport comparison")
    if set(every_replay_agreement_first_order_gain_positive) != required_controls:
        raise ValueError("replay control contract changed")
    if set(group_first_order_delta_vs_strongest) != {
        "car",
        "person",
        "truck",
        "other_nine",
    }:
        raise ValueError("group contract changed")

    def positive(control: str, scope: str, metric: str) -> bool:
        result = comparisons[control][scope][metric]
        return bool(
            result["mean_difference"] > 0.0
            and result["paired_bootstrap_95_ci"][0] > 0.0
        )

    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "sinkhorn_marginal_error_at_most_1e_6": max_sinkhorn_marginal_error <= 1e-6,
        "minimum_target_replay_median_cosine_at_least_0_90": (
            minimum_target_replay_median_cosine >= 0.90
        ),
        "transport_component_mean_cosine_with_existing_ce_nonnegative": (
            mean_transport_ce_component_cosine >= 0.0
        ),
        "agreement_first_order_gain_vs_original_duet_ci_lower_positive": positive(
            "original_duet", "agreement", "first_order"
        ),
        "agreement_cosine_gain_vs_original_duet_ci_lower_positive": positive(
            "original_duet", "agreement", "cosine"
        ),
        "agreement_first_order_gain_vs_duplicate_ce_ci_lower_positive": positive(
            "duplicate_hard_ce", "agreement", "first_order"
        ),
        "agreement_cosine_gain_vs_duplicate_ce_ci_lower_positive": positive(
            "duplicate_hard_ce", "agreement", "cosine"
        ),
        "overall_first_order_gain_vs_both_controls_ci_lower_positive": all(
            positive(control, "overall", "first_order") for control in required_controls
        ),
        "every_replay_agreement_first_order_gain_vs_both_controls_positive": all(
            every_replay_agreement_first_order_gain_positive.values()
        ),
        "candidate_negative_burden_not_worse": (
            candidate_negative_burden >= strongest_control_negative_burden
        ),
        "candidate_mean_norm_inflation_at_most_1_5x": (
            candidate_to_strongest_mean_norm_ratio <= 1.5
        ),
        "car_first_order_delta_nonnegative": group_first_order_delta_vs_strongest["car"]
        >= 0.0,
        "person_first_order_delta_nonnegative": group_first_order_delta_vs_strongest[
            "person"
        ]
        >= 0.0,
        "truck_first_order_delta_nonnegative": group_first_order_delta_vs_strongest[
            "truck"
        ]
        >= 0.0,
        "other_nine_first_order_delta_nonnegative": (
            group_first_order_delta_vs_strongest["other_nine"] >= 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "decision": ("NEEDS_EXACT_PARAMETER_AUDIT" if passed else "REJECT"),
        "checks": checks,
        "training_authorized": False,
        "proxy_authorized": False,
        "gpu_authorized": False,
    }
