from pathlib import Path

import numpy as np

from src.utils.patch_cls_pair_neutralization_audit import (
    evaluate_patch_pair_neutralization_gate,
    neutralize_candidate_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_neutralize_candidate_pair_preserves_only_pair_mass() -> None:
    probability = np.array(
        [[0.10, 0.60, 0.20, 0.10], [0.55, 0.05, 0.30, 0.10]],
        dtype=np.float64,
    )
    task = np.array([2, 2])
    clip = np.array([1, 0])
    result, transferred = neutralize_candidate_pair(probability, task, clip)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert np.allclose(result[0], [0.10, 0.40, 0.40, 0.10])
    assert np.allclose(result[1], [0.425, 0.05, 0.425, 0.10])
    assert np.allclose(transferred, [0.20, 0.125])


def test_neutralize_candidate_pair_rejects_agreement_rows() -> None:
    probability = np.array([[0.6, 0.4]])
    with np.testing.assert_raises(ValueError):
        neutralize_candidate_pair(probability, np.array([0]), np.array([0]))


def _paired(mean: float = 0.1, low: float = 0.01) -> dict:
    return {
        "mean_difference": mean,
        "paired_bootstrap_95_ci": [low, mean + 0.1],
    }


def _gate(**overrides):
    arguments = {
        "input_contract_valid": True,
        "source_suppression_reject_preserved": True,
        "heldout_selector_passed": True,
        "selected_coverage_pct": 2.9,
        "target_replay_max_abs_error": 1e-8,
        "nonpair_target_max_abs_error": 0.0,
        "pair_mass_max_abs_error": 0.0,
        "baseline_output_first_order": _paired(),
        "baseline_feature_first_order": _paired(),
        "suppression_output_first_order": _paired(),
        "suppression_feature_first_order": _paired(),
        "output_negative_burden_baseline": -0.2,
        "output_negative_burden_candidate": -0.1,
        "feature_negative_burden_baseline": -0.2,
        "feature_negative_burden_candidate": -0.1,
        "feature_helpful_retention_pct": 101.0,
        "feature_mean_norm_ratio": 1.0,
        "max_full_target_class_mass_shift_pp": 0.5,
        "class_macro_feature_first_order_delta": 0.01,
        "car_feature_first_order_delta": 0.01,
        "person_feature_first_order_delta": 0.01,
        "truck_feature_first_order_delta": 0.01,
        "other_nine_feature_first_order_delta": 0.01,
    }
    arguments.update(overrides)
    return evaluate_patch_pair_neutralization_gate(**arguments)


def test_gate_pass_only_authorizes_exact_parameter_audit() -> None:
    gate = _gate()
    assert gate["decision"] == "NEEDS_EXACT_PARAMETER_AUDIT"
    assert gate["exact_parameter_audit_authorized"] is True
    assert gate["proxy_authorized"] is False
    assert gate["training_authorized"] is False


def test_gate_rejects_nonpair_mutation_or_person_exchange() -> None:
    gate = _gate(nonpair_target_max_abs_error=1e-5)
    assert gate["decision"] == "REJECT"
    assert not gate["checks"]["nonpair_target_probability_unchanged"]
    gate = _gate(person_feature_first_order_delta=-1e-6)
    assert gate["decision"] == "REJECT"
    assert not gate["checks"]["person_feature_first_order_delta_nonnegative"]
    gate = _gate(max_full_target_class_mass_shift_pp=1.01)
    assert gate["decision"] == "REJECT"
    assert not gate["checks"]["max_full_target_class_mass_shift_at_most_1pp"]


def test_entrypoint_is_cpu_only_and_locks_before_oracle() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_patch_cls_pair_neutralization_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_patch_cls_pair_neutralization.py"
    ).read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "netF" not in audit
    assert "netB" not in audit
    assert "clip.load" not in audit
    assert ".backward(" not in audit
    assert "optimizer.step" not in audit
    assert audit.index("lock_path.write_text") < audit.index(
        "labels = _read_labels_after_lock"
    )
    assert audit.index("lock_path.write_text") < audit.index(
        "source_summary = json.loads"
    )
