from pathlib import Path

import numpy as np

from src.utils.attribute_mass_audit import (
    evaluate_attribute_mass_gate,
    paired_mean_bootstrap_ci,
    redistribute_pairwise_attribute_mass,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pairwise_mass_redistribution_is_exact_and_conservative() -> None:
    probability = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.4, 0.1, 0.5],
        ],
        dtype=np.float64,
    )
    margin = np.zeros((2, 2, 4), dtype=np.float64)
    margin[1] = np.log(3.0) / 2.0
    result = redistribute_pairwise_attribute_mass(
        probability,
        task_prediction=np.array([0, 1]),
        clip_prediction=np.array([2, 0]),
        attribute_margin=margin,
        clip_logit_scale=2.0,
    )

    np.testing.assert_allclose(result["task_fraction"], [0.5, 0.75])
    np.testing.assert_allclose(
        result["probability"],
        [[0.35, 0.3, 0.35], [0.125, 0.375, 0.5]],
    )
    np.testing.assert_allclose(result["probability"].sum(axis=1), 1.0)
    assert result["probability"][0, 1] == probability[0, 1]
    assert result["probability"][1, 2] == probability[1, 2]


def test_mass_redistribution_rejects_agreement_rows_and_bad_scale() -> None:
    probability = np.array([[0.6, 0.4]], dtype=np.float64)
    margin = np.zeros((1, 2, 4), dtype=np.float64)

    try:
        redistribute_pairwise_attribute_mass(
            probability,
            task_prediction=np.array([0]),
            clip_prediction=np.array([0]),
            attribute_margin=margin,
            clip_logit_scale=1.0,
        )
    except ValueError as error:
        assert "conflict rows only" in str(error)
    else:
        raise AssertionError("agreement row was not rejected")

    try:
        redistribute_pairwise_attribute_mass(
            probability,
            task_prediction=np.array([0]),
            clip_prediction=np.array([1]),
            attribute_margin=margin,
            clip_logit_scale=0.0,
        )
    except ValueError as error:
        assert "finite and positive" in str(error)
    else:
        raise AssertionError("non-positive scale was not rejected")


def test_paired_mean_bootstrap_detects_uniform_improvement() -> None:
    low, high = paired_mean_bootstrap_ci(
        np.full(20, 0.25), repeats=200, seed=2020, batch_size=20
    )
    assert low == 0.25
    assert high == 0.25


def test_gate_requires_all_comparators_class_safety_and_mass_safety() -> None:
    comparisons = {
        name: {
            "nll_ci_lower_positive": True,
            "brier_ci_lower_positive": True,
            "true_probability_ci_lower_positive": True,
        }
        for name in ("fixed_clip", "arithmetic", "rms")
    }
    passing = evaluate_attribute_mass_gate(
        input_contract_valid=True,
        comparison_checks=comparisons,
        car_nll_improvement=0.1,
        truck_nll_improvement=0.1,
        car_brier_improvement=0.1,
        truck_brier_improvement=0.1,
        max_abs_class_mass_shift_pp=0.9,
    )
    truck_failure = evaluate_attribute_mass_gate(
        input_contract_valid=True,
        comparison_checks=comparisons,
        car_nll_improvement=0.1,
        truck_nll_improvement=-0.01,
        car_brier_improvement=0.1,
        truck_brier_improvement=0.1,
        max_abs_class_mass_shift_pp=0.9,
    )
    mass_failure = evaluate_attribute_mass_gate(
        input_contract_valid=True,
        comparison_checks=comparisons,
        car_nll_improvement=0.1,
        truck_nll_improvement=0.1,
        car_brier_improvement=0.1,
        truck_brier_improvement=0.1,
        max_abs_class_mass_shift_pp=1.01,
    )

    assert passing["decision"] == "PASS_OFFLINE_GATE"
    assert truck_failure["decision"] == "REJECT"
    assert not truck_failure["checks"]["truck_nll_nonworse"]
    assert mass_failure["decision"] == "REJECT"
    assert not mass_failure["checks"]["class_mass_shift_at_most_1pp"]


def test_cloud_entrypoint_is_cpu_only_and_locks_before_labels() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_attribute_mass_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_attribute_mass.py"
    ).read_text()

    assert "image_target_of_oh_vs.py" not in runner
    assert 'CUDA_VISIBLE_DEVICES=""' in runner
    assert "encode_image" not in audit
    assert "encode_text" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert '"target_images_loaded": False' in audit
    assert '"model_forward_calls": 0' in audit
    assert '"training_code_modified": False' in audit
    assert '"training_authorized": False' in audit
    assert '"contains_target_paths": False' in audit
    assert '"source_signal_csv_read": False' in audit
    assert audit.index("lock_path.write_text") < audit.index(
        "target_list_sha256 = _sha256(args.target_list)"
    )
    assert audit.index("lock_path.write_text") < audit.index(
        "_parse_labels_after_lock(args.target_list)"
    )
