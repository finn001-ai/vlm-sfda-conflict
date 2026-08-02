import numpy as np

from src.utils.agreement_confidence_weight_audit import (
    ce_logit_descent,
    class_balanced_bottom_fraction_reference_weight,
    class_mean_normalized_confidence_weight,
    evaluate_agreement_confidence_weight_gate,
    paired_mean_bootstrap_ci,
    weighted_logit_alignment,
)


def test_confidence_weight_preserves_each_eligible_group_mean():
    confidence = np.array([0.2, 0.4, 0.8, 0.3, 0.6, 0.9])
    eligible = np.array([True, True, True, True, True, False])
    group = np.array([0, 0, 0, 1, 1, 1])
    weight = class_mean_normalized_confidence_weight(confidence, eligible, group)
    assert np.isclose(weight[:3].mean(), 1.0)
    assert np.isclose(weight[3:5].mean(), 1.0)
    assert weight[5] == 0.0


def test_bottom_fraction_reference_preserves_group_mean():
    confidence = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    eligible = np.ones(6, dtype=bool)
    group = np.array([0, 0, 0, 1, 1, 1])
    result = class_balanced_bottom_fraction_reference_weight(
        confidence, eligible, group, fraction=0.25
    )
    assert result["delayed"].tolist() == [True, False, False, True, False, False]
    assert np.isclose(result["weight"][:3].mean(), 1.0)
    assert np.isclose(result["weight"][3:].mean(), 1.0)


def test_logit_descent_and_weighted_alignment():
    probability = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]])
    pseudo_label = np.array([0, 1])
    oracle_label = np.array([0, 2])
    pseudo = ce_logit_descent(probability, pseudo_label)
    oracle = ce_logit_descent(probability, oracle_label)
    summary = weighted_logit_alignment(
        pseudo, oracle, np.array([1.0, 0.5]), np.array([True, True])
    )
    assert np.allclose(pseudo.sum(axis=1), 0.0)
    assert np.isclose(summary["mean_first_order"], -0.02)
    assert 0.0 < summary["effective_sample_size_pct"] <= 100.0


def test_paired_bootstrap_identifies_constant_positive_delta():
    candidate = np.array([2.0, 3.0, 4.0, 5.0])
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    interval = paired_mean_bootstrap_ci(
        candidate, reference, np.ones(4, dtype=bool), repeats=100, batch_size=10
    )
    assert interval == (1.0, 1.0)


def test_gate_passes_only_when_all_safeguards_pass():
    gate = evaluate_agreement_confidence_weight_gate(
        input_contract_valid=True,
        max_pseudo_class_mean_weight_error=1e-12,
        effective_sample_size_pct=95.0,
        delta_vs_unweighted_ci=(0.01, 0.02),
        delta_vs_hard_delay_ci=(0.005, 0.015),
        candidate_negative_burden=0.1,
        baseline_negative_burden=0.2,
        candidate_positive_support=0.5,
        hard_delay_positive_support=0.49,
        car_first_order_delta=0.01,
        truck_first_order_delta=0.01,
        noncar_first_order_delta=0.01,
    )
    assert gate["decision"] == "PASS_AGREEMENT_CONFIDENCE_WEIGHT_PREFLIGHT"
    assert gate["training_authorized"] is False
