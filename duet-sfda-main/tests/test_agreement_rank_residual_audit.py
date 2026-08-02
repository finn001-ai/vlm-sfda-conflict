from __future__ import annotations

import numpy as np

from src.utils.agreement_rank_residual_audit import (
    agreement_rank_residual,
    evaluate_agreement_rank_residual_gate,
    paired_selection_precision_bootstrap_ci,
    select_class_balanced_fraction,
    select_matched_counts,
)


def test_rank_residual_requires_common_top1_and_opposing_runner_up() -> None:
    task = np.array([
        [0.60, 0.30, 0.10],
        [0.60, 0.30, 0.10],
        [0.60, 0.30, 0.10],
    ])
    clip = np.array([
        [0.55, 0.10, 0.35],
        [0.55, 0.35, 0.10],
        [0.10, 0.55, 0.35],
    ])
    result = agreement_rank_residual(task, clip)
    np.testing.assert_array_equal(result["common_top1"], [True, True, False])
    np.testing.assert_array_equal(
        result["runner_up_disagreement"], [True, False, False]
    )
    assert result["rank_residual"][0] > 0.0
    assert result["rank_residual"][1] == 0.0
    assert result["rank_residual"][2] == 0.0


def test_selector_and_baseline_have_identical_per_class_counts() -> None:
    values = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.3, 0.0])
    eligible = np.ones(8, dtype=bool)
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    candidate = select_class_balanced_fraction(
        values, eligible, group, fraction=0.5, largest=True
    )
    np.testing.assert_array_equal(
        candidate["selected"], [True, True, False, False, True, True, False, False]
    )
    baseline = select_matched_counts(
        values, eligible, group, candidate["counts_by_group"], largest=False
    )
    np.testing.assert_array_equal(
        baseline, [False, False, True, True, False, False, True, True]
    )
    for class_index in (0, 1):
        assert (candidate["selected"] & (group == class_index)).sum() == 2
        assert (baseline & (group == class_index)).sum() == 2


def test_paired_precision_interval_is_positive_for_better_selector() -> None:
    wrong = np.zeros(1_000, dtype=bool)
    wrong[:100] = True
    candidate = np.zeros(1_000, dtype=bool)
    baseline = np.zeros(1_000, dtype=bool)
    candidate[:100] = True
    baseline[100:200] = True
    interval = paired_selection_precision_bootstrap_ci(
        candidate, baseline, wrong, np.ones(1_000, dtype=bool), repeats=500
    )
    assert interval[0] > 0.0


def _comparison(gain: int = 10, low: float = 1.0) -> dict:
    return {
        "captured_error_gain": gain,
        "selection_precision_gain_pp": 2.0,
        "paired_bootstrap_95_ci_pp": [low, 3.0],
    }


def test_gate_requires_car_truck_and_noncar_noninferiority() -> None:
    comparisons = {
        "task_confidence": _comparison(),
        "clip_confidence": _comparison(),
        "arithmetic_confidence": _comparison(),
        "rms_confidence": _comparison(),
    }
    captures = {
        "candidate": 5,
        "task_confidence": 4,
        "clip_confidence": 5,
        "arithmetic_confidence": 3,
        "rms_confidence": 4,
    }
    passing = evaluate_agreement_rank_residual_gate(
        input_contract_valid=True,
        baseline_reproduced=True,
        selected_fraction_pct=10.0,
        error_enrichment=2.5,
        retained_accuracy_gain_pp=0.5,
        comparisons=comparisons,
        car_wrong_captures=captures,
        truck_wrong_captures=captures,
        noncar_wrong_captures=captures,
    )
    assert passing["decision"] == "PASS_AGREEMENT_RANK_RESIDUAL_PREFLIGHT"

    failing_truck = {**captures, "candidate": 2}
    failing = evaluate_agreement_rank_residual_gate(
        input_contract_valid=True,
        baseline_reproduced=True,
        selected_fraction_pct=10.0,
        error_enrichment=2.5,
        retained_accuracy_gain_pp=0.5,
        comparisons=comparisons,
        car_wrong_captures=captures,
        truck_wrong_captures=failing_truck,
        noncar_wrong_captures=captures,
    )
    assert failing["decision"] == "REJECT"
    assert not failing["checks"][
        "truck_wrong_capture_nonworse_than_all_confidence_baselines"
    ]
