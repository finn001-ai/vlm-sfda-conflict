from pathlib import Path

import numpy as np
import pytest

from src.utils.agreement_neighbor_clip_audit import (
    agreement_neighbor_clip_posterior,
    evaluate_agreement_neighbor_clip_gate,
    select_from_candidate_set,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARATORS = {
    "fixed_task",
    "fixed_clip",
    "confidence_choice",
    "arithmetic",
    "rms",
}


def test_agreement_neighbors_are_exact_and_average_clip_probabilities():
    feature = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.95, 0.05],
        ]
    )
    clip = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.85, 0.15],
        ]
    )
    agreement = np.array([True, True, True, True, True, True, False])
    query = ~agreement
    result = agreement_neighbor_clip_posterior(
        feature, clip, agreement, query, neighbors=5, chunk_size=1
    )
    assert result["query_index"].tolist() == [6]
    assert np.all(agreement[result["neighbor_index"]])
    expected = clip[result["neighbor_index"][0]].mean(axis=0)
    assert np.allclose(result["posterior"][0], expected)
    similarity = result["neighbor_similarity"][0]
    assert np.all(similarity[:-1] >= similarity[1:])


def test_neighbor_contract_rejects_overlapping_reference_and_query_masks():
    feature = np.eye(3)
    probability = np.full((3, 3), 1.0 / 3.0)
    with pytest.raises(ValueError, match="disjoint"):
        agreement_neighbor_clip_posterior(
            feature,
            probability,
            np.array([True, True, False]),
            np.array([False, True, True]),
            neighbors=2,
        )


def test_candidate_selection_never_leaves_the_candidate_set():
    posterior = np.array([[0.1, 0.6, 0.3], [0.7, 0.2, 0.1]])
    candidates = np.array([[0, 2, -1], [1, 2, -1]])
    selected = select_from_candidate_set(posterior, candidates)
    assert selected["prediction"].tolist() == [2, 1]
    assert np.all((candidates == selected["prediction"][:, None]).any(axis=1))


def test_gate_requires_matched_gain_stability_and_hard_class_safety():
    comparisons = {
        name: {"gain_pp": 1.2, "paired_bootstrap_95_ci_pp": (0.5, 1.8)}
        for name in COMPARATORS
    }
    passing = evaluate_agreement_neighbor_clip_gate(
        input_contract_valid=True,
        neighbors=5,
        decision_stability_pct=95.0,
        candidate_set_coverage_pct=93.0,
        minimum_class_candidate_coverage_pct=87.0,
        neighbor_label_match_pct=70.0,
        comparisons=comparisons,
        best_baseline_name="fixed_clip",
        car_delta_pp=0.2,
        truck_delta_pp=0.3,
        car_truck_mean_delta_pp=0.25,
        other_ten_mean_delta_pp=0.5,
        max_class_mass_shift_pp=0.4,
    )
    assert passing["decision"] == "PASS_AGREEMENT_NEIGHBOR_CLIP_PREFLIGHT"
    assert passing["training_authorized"] is False

    failing = evaluate_agreement_neighbor_clip_gate(
        input_contract_valid=True,
        neighbors=5,
        decision_stability_pct=95.0,
        candidate_set_coverage_pct=93.0,
        minimum_class_candidate_coverage_pct=87.0,
        neighbor_label_match_pct=70.0,
        comparisons=comparisons,
        best_baseline_name="fixed_clip",
        car_delta_pp=-0.8,
        truck_delta_pp=0.3,
        car_truck_mean_delta_pp=-0.25,
        other_ten_mean_delta_pp=0.5,
        max_class_mass_shift_pp=0.4,
    )
    assert failing["decision"] == "REJECT"
    assert not failing["checks"]["car_regression_at_most_0_5pp"]
    assert not failing["checks"]["car_truck_mean_nonnegative"]


def test_cloud_entrypoint_is_cpu_only_and_locks_before_oracle_labels():
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_agreement_neighbor_clip_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_agreement_neighbor_clip.py"
    ).read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "import torch" not in audit
    assert "import clip" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert audit.index("lock_path.write_text") < audit.index(
        "_parse_labels_after_lock(args.target_list)"
    )
    assert audit.index("lock_path.write_text") < audit.index('snapshot["target_label"]')
