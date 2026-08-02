from pathlib import Path

import numpy as np
import pytest

from src.utils.candidate_set_gradient_audit import (
    evaluate_candidate_gradient_gate,
    kl_logit_descent,
    oracle_ce_logit_descent,
    paired_mean_bootstrap_ci,
    rowwise_oracle_alignment,
    set_mass_logit_descent,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


def test_set_mass_descent_matches_finite_difference_logit_gradient() -> None:
    logits = np.array([1.2, -0.3, 0.7, -1.0], dtype=np.float64)
    probability = _softmax(logits)[None, :]
    mask = np.array([[True, False, True, False]])
    descent = set_mass_logit_descent(probability, mask)[0]

    epsilon = 1e-6
    numerical_gradient = np.empty_like(logits)
    for index in range(logits.size):
        plus = logits.copy()
        minus = logits.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss = -np.log(_softmax(plus)[mask[0]].sum())
        minus_loss = -np.log(_softmax(minus)[mask[0]].sum())
        numerical_gradient[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
    np.testing.assert_allclose(descent, -numerical_gradient, atol=1e-9, rtol=1e-7)


def test_kl_and_oracle_ce_descents_have_expected_closed_forms() -> None:
    student = np.array([[0.6, 0.3, 0.1]])
    target = np.array([[0.1, 0.2, 0.7]])
    np.testing.assert_allclose(kl_logit_descent(student, target), target - student)
    np.testing.assert_allclose(
        oracle_ce_logit_descent(student, np.array([2])),
        np.array([[-0.6, -0.3, 0.9]]),
    )


def test_alignment_exposes_magnitude_direction_and_zero_rows() -> None:
    oracle = np.array([[1.0, -1.0], [0.5, -0.5]])
    candidate = np.array([[2.0, -2.0], [0.0, 0.0]])
    result = rowwise_oracle_alignment(candidate, oracle)
    np.testing.assert_allclose(result["first_order"], [4.0, 0.0])
    np.testing.assert_allclose(result["cosine"], [1.0, 0.0])
    np.testing.assert_array_equal(result["joint_nonzero"], [True, False])


def test_paired_bootstrap_detects_uniform_positive_difference() -> None:
    low, high = paired_mean_bootstrap_ci(
        np.linspace(0.1, 0.3, 200), repeats=300, seed=2020, batch_size=25
    )
    assert low > 0.0
    assert high > low


def _passing_comparisons() -> dict:
    metric = {"mean_difference": 0.1, "paired_bootstrap_95_ci": [0.05, 0.15]}
    return {
        "versus_clip_kl": {
            "cosine": dict(metric),
            "oracle_unit_projection": dict(metric),
            "first_order": dict(metric),
        },
        "versus_top1_set": {
            "cosine": dict(metric),
            "oracle_unit_projection": dict(metric),
            "first_order": dict(metric),
        },
    }


def test_gate_requires_paired_direction_magnitude_and_hard_class_safety() -> None:
    passing = evaluate_candidate_gradient_gate(
        input_contract_valid=True,
        comparisons=_passing_comparisons(),
        macro_first_order_delta_vs_clip=0.01,
        hard_class_first_order_delta_vs_clip={
            "car": 0.01,
            "person": 0.02,
            "truck": 0.0,
        },
        top2_harmful_pct=20.0,
        clip_harmful_pct=21.0,
    )
    assert passing["decision"] == "PASS_SET_GRADIENT_PREFLIGHT"
    assert passing["training_authorized"] is False

    failed_comparisons = _passing_comparisons()
    failed_comparisons["versus_top1_set"]["first_order"] = {
        "mean_difference": 0.1,
        "paired_bootstrap_95_ci": [-0.01, 0.2],
    }
    failing = evaluate_candidate_gradient_gate(
        input_contract_valid=True,
        comparisons=failed_comparisons,
        macro_first_order_delta_vs_clip=0.01,
        hard_class_first_order_delta_vs_clip={
            "car": 0.01,
            "person": 0.02,
            "truck": -1e-6,
        },
        top2_harmful_pct=22.0,
        clip_harmful_pct=21.0,
    )
    assert failing["decision"] == "REJECT"
    assert not failing["checks"]["top2_first_order_beats_top1_with_positive_ci"]
    assert not failing["checks"]["truck_first_order_delta_vs_clip_nonnegative"]
    assert not failing["checks"]["harmful_fraction_not_above_clip"]


def test_probability_and_mask_validation_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        kl_logit_descent(np.array([[0.8, 0.8]]), np.array([[0.5, 0.5]]))
    with pytest.raises(ValueError, match="non-empty"):
        set_mass_logit_descent(np.array([[0.5, 0.5]]), np.array([[False, False]]))


def test_cloud_entrypoint_is_cpu_only_and_locks_before_oracle_labels() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_candidate_set_gradient_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_candidate_set_gradient.py"
    ).read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "image_target_of_oh_vs.py" not in runner
    assert "import torch" not in audit
    assert "import clip" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert '"target_images_loaded": False' in audit
    assert '"model_forward_calls": 0' in audit
    assert '"training_code_modified": False' in audit
    assert '"training_authorized": False' in audit
    assert audit.index("lock_path.write_text") < audit.index(
        "target_list_hash = _sha256(args.target_list)"
    )
    assert audit.index("lock_path.write_text") < audit.index(
        "_parse_labels_after_lock(args.target_list)"
    )
