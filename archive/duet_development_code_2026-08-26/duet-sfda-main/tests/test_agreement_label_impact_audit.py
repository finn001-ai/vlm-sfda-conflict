from pathlib import Path

import numpy as np
import pytest

from src.utils.agreement_label_impact_audit import (
    evaluate_agreement_label_impact_gate,
    fit_agreement_label_impact,
    label_impact_score,
    select_candidate_by_label_impact,
    stratified_alternating_reference_masks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARATORS = {
    "fixed_task",
    "fixed_clip",
    "confidence_choice",
    "arithmetic",
    "rms",
}


def _toy_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.75, 0.20, 0.05],
            [0.80, 0.15, 0.05],
            [0.10, 0.80, 0.10],
            [0.15, 0.75, 0.10],
            [0.05, 0.85, 0.10],
            [0.05, 0.10, 0.85],
            [0.10, 0.15, 0.75],
            [0.10, 0.05, 0.85],
        ],
        dtype=np.float64,
    )
    feature = np.array(
        [
            [2.0, 0.1],
            [1.8, 0.2],
            [2.1, -0.1],
            [0.1, 2.0],
            [0.2, 1.8],
            [-0.1, 2.1],
            [-1.8, -1.8],
            [-2.0, -1.7],
            [-1.7, -2.0],
        ],
        dtype=np.float64,
    )
    label = np.repeat(np.arange(3), 3)
    reference = np.ones(9, dtype=bool)
    return probability, feature, label, reference


def test_label_impact_score_matches_explicit_head_gradient_dot_product() -> None:
    probability, feature, label, reference = _toy_reference()
    model = fit_agreement_label_impact(
        probability, feature, label, reference, class_count=3
    )
    query_probability = np.array([[0.4, 0.35, 0.25]])
    query_feature = np.array([[0.9, 0.4]])
    score = label_impact_score(query_probability, query_feature, model)

    augmented = np.append(query_feature[0], 1.0)
    for candidate in range(3):
        residual = query_probability[0].copy()
        residual[candidate] -= 1.0
        gradient = residual[:, None] * augmented[None, :]
        expected = np.sum(model["preconditioned_mean_gradient"][candidate] * gradient)
        np.testing.assert_allclose(score[0, candidate], expected)


def test_candidate_selection_never_leaves_declared_set() -> None:
    score = np.array([[5.0, 4.0, 9.0], [3.0, 8.0, 7.0]])
    candidates = np.array([[0, 1, -1], [0, 2, -1]])
    result = select_candidate_by_label_impact(score, candidates)
    assert result["prediction"].tolist() == [0, 2]
    assert np.all((candidates == result["prediction"][:, None]).any(axis=1))


def test_fit_rejects_missing_class_references() -> None:
    probability, feature, label, reference = _toy_reference()
    reference[label == 2] = False
    with pytest.raises(ValueError, match="at least two"):
        fit_agreement_label_impact(
            probability, feature, label, reference, class_count=3
        )


def test_stratified_split_partitions_every_class() -> None:
    label = np.repeat(np.arange(3), 6)
    reference = np.ones(label.size, dtype=bool)
    first, second = stratified_alternating_reference_masks(
        label, reference, np.arange(label.size), class_count=3
    )
    assert not np.any(first & second)
    assert np.array_equal(first | second, reference)
    for class_index in range(3):
        assert int((first & (label == class_index)).sum()) == 3
        assert int((second & (label == class_index)).sum()) == 3


def _comparisons(gain: float = 1.2, low: float = 0.4) -> dict:
    return {
        name: {
            "gain_pp": gain,
            "paired_bootstrap_95_ci_pp": [low, 1.8],
        }
        for name in COMPARATORS
    }


def test_gate_requires_accuracy_stability_and_class_safety() -> None:
    kwargs = {
        "input_contract_valid": True,
        "agreement_reference_accuracy_pct": 94.0,
        "minimum_split_decision_stability_pct": 93.0,
        "candidate_set_coverage_pct": 93.0,
        "minimum_class_candidate_coverage_pct": 87.0,
        "comparisons": _comparisons(),
        "best_baseline_name": "fixed_clip",
        "car_delta_pp": 0.2,
        "truck_delta_pp": 0.1,
        "car_truck_mean_delta_pp": 0.15,
        "other_ten_mean_delta_pp": 0.3,
        "max_class_mass_shift_pp": 0.8,
    }
    passing = evaluate_agreement_label_impact_gate(**kwargs)
    assert passing["decision"] == "PASS_AGREEMENT_LABEL_IMPACT_PREFLIGHT"
    assert passing["training_authorized"] is False
    assert passing["proxy_authorized"] is False
    assert passing["gpu_authorized"] is False

    kwargs["truck_delta_pp"] = -0.8
    rejected = evaluate_agreement_label_impact_gate(**kwargs)
    assert rejected["decision"] == "REJECT"
    assert not rejected["checks"]["truck_regression_at_most_0_5pp"]


def test_entrypoint_is_cpu_only_and_locks_before_oracle() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_agreement_label_impact_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_agreement_label_impact.py"
    ).read_text()
    helper = (REPO_ROOT / "src/utils/agreement_label_impact_audit.py").read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "source_C.pt" not in runner
    assert "source_F.pt" not in runner
    assert "import torch" not in audit
    assert "import clip" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert '"training_authorized": False' in helper
    assert audit.index("lock_path.write_text") < audit.rindex(
        "labels = _parse_labels_after_lock"
    )
    assert audit.index("lock_path.write_text") < audit.rindex(
        'snapshot["target_label"]'
    )
