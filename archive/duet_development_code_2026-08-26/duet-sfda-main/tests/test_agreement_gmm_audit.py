from pathlib import Path

import numpy as np
import pytest

from src.utils.agreement_gmm_audit import (
    diagonal_gaussian_log_likelihood,
    evaluate_agreement_gmm_gate,
    fit_diagonal_class_gaussians,
    joint_centered_log_probability,
    select_candidate_by_log_likelihood,
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


def test_joint_centered_log_probability_preserves_log_odds() -> None:
    task = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    clip = np.array([[0.6, 0.1, 0.3], [0.1, 0.7, 0.2]])
    evidence = joint_centered_log_probability(task, clip)
    assert evidence.shape == (2, 6)
    np.testing.assert_allclose(evidence[:, :3].sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(evidence[:, 3:].sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        evidence[:, 0] - evidence[:, 1], np.log(task[:, 0] / task[:, 1])
    )


def test_diagonal_gaussians_recover_separated_classes() -> None:
    evidence = np.array(
        [
            [-2.0, -1.8],
            [-1.7, -2.1],
            [2.0, 1.8],
            [1.7, 2.1],
            [-1.9, -2.0],
            [1.9, 2.0],
        ]
    )
    label = np.array([0, 0, 1, 1, 0, 1])
    reference = np.ones(6, dtype=bool)
    model = fit_diagonal_class_gaussians(
        evidence, label, reference, class_count=2
    )
    score = diagonal_gaussian_log_likelihood(
        evidence, model["mean"], model["variance"]
    )
    assert np.array_equal(score.argmax(axis=1), label)
    assert model["reference_count"].tolist() == [3, 3]
    assert np.all(model["variance"] > 0.0)


def test_gaussian_fit_requires_two_references_per_class() -> None:
    with pytest.raises(ValueError, match="at least two"):
        fit_diagonal_class_gaussians(
            np.eye(3),
            np.array([0, 0, 1]),
            np.ones(3, dtype=bool),
            class_count=2,
        )


def test_candidate_selection_never_leaves_declared_set() -> None:
    score = np.array([[5.0, 4.0, 3.0], [0.0, 2.0, 8.0]])
    candidates = np.array([[1, 2, -1], [0, 1, -1]])
    result = select_candidate_by_log_likelihood(score, candidates)
    assert result["prediction"].tolist() == [1, 1]
    assert np.all((candidates == result["prediction"][:, None]).any(axis=1))


def test_stratified_alternating_split_avoids_index_class_correlation() -> None:
    label = np.repeat(np.arange(3), 6)
    reference = np.ones(label.size, dtype=bool)
    sample_index = np.arange(label.size)
    first, second = stratified_alternating_reference_masks(
        label, reference, sample_index, class_count=3
    )
    assert not np.any(first & second)
    assert np.array_equal(first | second, reference)
    for class_index in range(3):
        assert int((first & (label == class_index)).sum()) == 3
        assert int((second & (label == class_index)).sum()) == 3


def _comparisons(gain: float = 1.2, low: float = 0.5) -> dict:
    return {
        name: {
            "gain_pp": gain,
            "paired_bootstrap_95_ci_pp": [low, 1.8],
        }
        for name in COMPARATORS
    }


def test_gate_requires_gain_stability_and_class_safety() -> None:
    kwargs = {
        "input_contract_valid": True,
        "reference_crossfit_accuracy_pct": 95.0,
        "minimum_split_decision_stability_pct": 94.0,
        "candidate_set_coverage_pct": 93.0,
        "minimum_class_candidate_coverage_pct": 87.0,
        "comparisons": _comparisons(),
        "best_baseline_name": "fixed_clip",
        "car_delta_pp": 0.2,
        "truck_delta_pp": 0.3,
        "car_truck_mean_delta_pp": 0.25,
        "other_ten_mean_delta_pp": 0.4,
        "max_class_mass_shift_pp": 0.8,
    }
    passing = evaluate_agreement_gmm_gate(**kwargs)
    assert passing["decision"] == "PASS_AGREEMENT_GMM_PREFLIGHT"
    assert passing["training_authorized"] is False
    assert passing["proxy_authorized"] is False
    assert passing["gpu_authorized"] is False

    kwargs["truck_delta_pp"] = -0.8
    rejected = evaluate_agreement_gmm_gate(**kwargs)
    assert rejected["decision"] == "REJECT"
    assert not rejected["checks"]["truck_regression_at_most_0_5pp"]


def test_entrypoint_is_cpu_only_and_locks_before_oracle() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_agreement_gmm_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_agreement_gmm.py"
    ).read_text()
    helper = (REPO_ROOT / "src/utils/agreement_gmm_audit.py").read_text()
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
