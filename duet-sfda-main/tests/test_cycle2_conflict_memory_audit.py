from pathlib import Path

import numpy as np

from src.utils.cycle2_conflict_memory_audit import (
    build_cycle2_conflict_memory_target,
    evaluate_cycle2_conflict_memory_gate,
    stable_top2_union_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cycle2_target_preserves_clip_top1_and_relative_mass() -> None:
    task = np.array([[0.60, 0.20, 0.15, 0.05]], dtype=np.float64)
    clip = np.array([[0.10, 0.30, 0.20, 0.40]], dtype=np.float64)
    result = build_cycle2_conflict_memory_target(task, clip)
    expected_support = np.array([[True, True, False, True]])
    np.testing.assert_array_equal(result["support"], expected_support)
    np.testing.assert_array_equal(result["probability"].argmax(1), clip.argmax(1))
    np.testing.assert_allclose(
        result["probability"][0, 3] / result["probability"][0, 1], 4.0 / 3.0
    )


def test_top2_union_requires_matching_probabilities() -> None:
    task = np.array([[0.7, 0.3]])
    clip = np.array([[0.2, 0.3, 0.5]])
    try:
        stable_top2_union_mask(task, clip)
    except ValueError as error:
        assert "matching shapes" in str(error)
    else:
        raise AssertionError("mismatched probabilities were accepted")


def _metric(mean=0.02, low=0.01, high=0.03):
    return {
        "mean_difference": mean,
        "paired_bootstrap_95_ci": [low, high],
    }


def test_gate_requires_both_resolved_and_still_conflict_gain() -> None:
    comparison = {
        "cosine": _metric(),
        "oracle_unit_projection": _metric(),
        "first_order": _metric(),
    }
    passing = evaluate_cycle2_conflict_memory_gate(
        input_contract_valid=True,
        candidate_top1_matches_clip=True,
        overall_comparison=comparison,
        resolved_first_order=_metric(),
        still_conflict_first_order=_metric(),
        minimum_class_first_order_delta=0.001,
        candidate_negative_burden=-0.01,
        clip_negative_burden=-0.02,
        top2_union_oracle_coverage_pct=93.0,
        max_full_mass_shift_pp=0.5,
    )
    assert passing["decision"] == "PASS_CYCLE2_CONFLICT_MEMORY_PREFLIGHT"

    failing = evaluate_cycle2_conflict_memory_gate(
        input_contract_valid=True,
        candidate_top1_matches_clip=True,
        overall_comparison=comparison,
        resolved_first_order=_metric(),
        still_conflict_first_order=_metric(mean=-0.01, low=-0.02, high=0.0),
        minimum_class_first_order_delta=0.001,
        candidate_negative_burden=-0.01,
        clip_negative_burden=-0.02,
        top2_union_oracle_coverage_pct=93.0,
        max_full_mass_shift_pp=0.5,
    )
    assert failing["decision"] == "REJECT"
    assert not failing["checks"]["still_conflict_first_order_gain_ci_lower_positive"]


def test_runner_stops_before_cycle2_optimization() -> None:
    plmatch = (REPO_ROOT / "src/methods/oh/plmatch.py").read_text()
    runner = (
        REPO_ROOT / "tools/run_visda_cycle2_conflict_memory_audit.sh"
    ).read_text()
    assert "stop_after_pre_cycle == curr_cycle + 1" in plmatch
    assert "return netF, netB, netC" in plmatch
    assert "FAILURE_AUDIT.STOP_AFTER_PRE_CYCLE 2" in runner
    assert "ACTIVE.CYCLE 2" in runner
    assert "optimizer_steps_in_cycle=0" in runner
    assert "Reusing completed cycle-2 snapshots; GPU will not be started" in runner
    assert "if maximum_error > 0.10" in runner
    assert "no proxy or full training was started" in runner
