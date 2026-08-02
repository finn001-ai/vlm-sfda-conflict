from pathlib import Path

import numpy as np
import pytest

from src.utils.attribute_kl_influence_audit import (
    evaluate_attribute_kl_influence,
    kl_logit_descent_directions,
    oracle_logit_influence,
    paired_bootstrap_mean_ci,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_kl_increment_is_exact_candidate_minus_control_target() -> None:
    student = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    control = np.array([[0.3, 0.6, 0.1], [0.2, 0.7, 0.1]])
    candidate = np.array([[0.5, 0.4, 0.1], [0.4, 0.5, 0.1]])
    result = kl_logit_descent_directions(student, control, candidate, kl_weight=0.4)
    assert np.allclose(result["control_direction"], 0.4 * (control - student))
    assert np.allclose(result["candidate_direction"], 0.4 * (candidate - student))
    assert np.allclose(result["incremental_direction"], 0.4 * (candidate - control))


def test_oracle_projection_separates_task_clip_and_neither_labels() -> None:
    student = np.repeat(np.array([[0.7, 0.2, 0.1]]), 3, axis=0)
    control = np.repeat(np.array([[0.3, 0.6, 0.1]]), 3, axis=0)
    candidate = np.repeat(np.array([[0.5, 0.4, 0.1]]), 3, axis=0)
    directions = kl_logit_descent_directions(student, control, candidate, kl_weight=0.4)
    result = oracle_logit_influence(
        student,
        directions["control_direction"],
        directions["candidate_direction"],
        np.array([0, 1, 2]),
    )
    assert result["incremental_projection"][0] > 0.0
    assert result["incremental_projection"][1] < 0.0
    assert result["incremental_projection"][2] < 0.0
    assert np.all(np.isfinite(result["candidate_cosine"]))


def test_bootstrap_ci_is_deterministic() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4])
    first = paired_bootstrap_mean_ci(values, seed=2020, repeats=200)
    second = paired_bootstrap_mean_ci(values, seed=2020, repeats=200)
    assert first == second
    assert first[0] > 0.0


def test_gate_rejects_positive_direction_that_failed_proxy_translation() -> None:
    report = evaluate_attribute_kl_influence(
        input_contract_valid=True,
        active_conflict_count_matches=True,
        changed_top1_count_matches=True,
        mean_incremental_projection=0.01,
        incremental_projection_ci=(0.005, 0.015),
        macro_class_mean_projection=0.008,
        car_mean_projection=0.004,
        truck_mean_projection=0.002,
        observed_final_delta_pp=0.01,
        observed_hard_mean_delta_pp=-0.067,
    )
    assert report["decision"] == "REJECT_ATTRIBUTE_BRANCH"
    assert (
        report["diagnosis"] == "directional_signal_did_not_translate_to_proxy_accuracy"
    )
    assert report["training_authorized"] is False


def test_gate_rejects_class_unsafe_incremental_direction() -> None:
    report = evaluate_attribute_kl_influence(
        input_contract_valid=True,
        active_conflict_count_matches=True,
        changed_top1_count_matches=True,
        mean_incremental_projection=0.01,
        incremental_projection_ci=(0.005, 0.015),
        macro_class_mean_projection=0.008,
        car_mean_projection=0.004,
        truck_mean_projection=-0.002,
        observed_final_delta_pp=0.3,
        observed_hard_mean_delta_pp=0.1,
    )
    assert report["decision"] == "REJECT_ATTRIBUTE_BRANCH"
    assert report["diagnosis"] == "incremental_kl_direction_is_not_class_safe"


def test_probability_validation_rejects_non_normalized_rows() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        kl_logit_descent_directions(
            np.array([[0.8, 0.8]]),
            np.array([[0.5, 0.5]]),
            np.array([[0.5, 0.5]]),
            kl_weight=0.4,
        )


def test_wrapper_declares_cpu_only_no_training_contract() -> None:
    wrapper = (
        REPO_ROOT / "tools/run_visda_conflict_attribute_kl_influence_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_attribute_kl_influence.py"
    ).read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in wrapper
    assert "image_target_of_oh_vs.py" not in wrapper
    assert "optimizer.step" not in audit
    assert '"training_authorized": False' in audit
    assert '"network_jacobian_included": False' in audit
