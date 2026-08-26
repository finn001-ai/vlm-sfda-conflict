from pathlib import Path

import numpy as np

from src.utils.attribute_mass_audit import (
    entropy_anchored_attribute_mass,
    evaluate_attribute_reliability_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _margin_for_log_odds(log_odds: np.ndarray) -> np.ndarray:
    margin = np.zeros((log_odds.size, 2, 4), dtype=np.float64)
    margin[:] = (log_odds / 2.0)[:, None, None]
    return margin


def test_entropy_anchor_uses_attribute_only_when_clip_uncertain_and_task_certain() -> (
    None
):
    clip_probability = np.array(
        [
            [0.999, 0.0005, 0.0005],
            [0.45, 0.45, 0.10],
        ],
        dtype=np.float64,
    )
    task_probability = np.array(
        [
            [0.999, 0.0005, 0.0005],
            [0.89, 0.01, 0.10],
        ],
        dtype=np.float64,
    )
    task_prediction = np.array([0, 0])
    clip_prediction = np.array([1, 1])
    result = entropy_anchored_attribute_mass(
        task_probability,
        clip_probability,
        task_prediction,
        clip_prediction,
        _margin_for_log_odds(np.array([-4.0, 4.0])),
        clip_logit_scale=2.0,
    )

    assert result["attribute_weight"][0] < 0.01
    assert result["anchored_fraction"][0] > 0.99
    assert result["attribute_weight"][1] > 0.9
    assert result["anchored_fraction"][1] > 0.97


def test_entropy_anchor_conserves_pair_mass_and_every_outside_probability() -> None:
    clip_probability = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
    task_probability = np.array([[0.7, 0.2, 0.1]], dtype=np.float64)
    result = entropy_anchored_attribute_mass(
        task_probability,
        clip_probability,
        task_prediction=np.array([0]),
        clip_prediction=np.array([2]),
        attribute_margin=np.ones((1, 2, 4), dtype=np.float64) * 0.1,
        clip_logit_scale=2.0,
    )

    np.testing.assert_allclose(result["probability"].sum(axis=1), 1.0)
    np.testing.assert_allclose(result["probability"][0, [0, 2]].sum(), 0.7)
    assert result["probability"][0, 1] == clip_probability[0, 1]


def test_reliability_gate_rejects_truck_or_noncar_exchange() -> None:
    fixed_checks = {
        "nll_ci_lower_positive": True,
        "brier_ci_lower_positive": True,
        "true_probability_ci_lower_positive": True,
    }
    safe_class = {
        "nll_improvement": 0.01,
        "brier_improvement": 0.01,
        "accuracy_gain_pp": 0.1,
    }
    passing = evaluate_attribute_reliability_gate(
        input_contract_valid=True,
        fixed_clip_checks=fixed_checks,
        accuracy_gain_pp=1.1,
        accuracy_ci_pp=(0.5, 1.7),
        car_metrics=safe_class,
        truck_metrics=safe_class,
        noncar_net_corrections=1,
        max_abs_class_mass_shift_pp=0.8,
    )
    unsafe_truck = dict(safe_class, accuracy_gain_pp=-0.1)
    failing = evaluate_attribute_reliability_gate(
        input_contract_valid=True,
        fixed_clip_checks=fixed_checks,
        accuracy_gain_pp=1.1,
        accuracy_ci_pp=(0.5, 1.7),
        car_metrics=safe_class,
        truck_metrics=unsafe_truck,
        noncar_net_corrections=-1,
        max_abs_class_mass_shift_pp=0.8,
    )

    assert passing["decision"] == "PASS_OFFLINE_GATE"
    assert failing["decision"] == "REJECT"
    assert not failing["checks"]["truck_accuracy_nonworse"]
    assert not failing["checks"]["noncar_net_corrections_nonnegative"]


def test_entrypoint_is_zero_training_and_locks_before_oracle_labels() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_attribute_reliability_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_attribute_reliability.py"
    ).read_text()

    assert "image_target_of_oh_vs.py" not in runner
    assert 'CUDA_VISIBLE_DEVICES=""' in runner
    assert "clip.load" not in audit
    assert "encode_image" not in audit
    assert "encode_text" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert '"target_images_loaded": False' in audit
    assert '"model_checkpoint_loads": 0' in audit
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
