from pathlib import Path

import numpy as np

from src.utils.agreement_shared_runner_up_audit import (
    evaluate_shared_runner_up_gate,
    shared_runner_up_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shared_runner_up_requires_common_top1_and_second_class():
    task = np.array([
        [0.7, 0.2, 0.1],
        [0.7, 0.1, 0.2],
        [0.6, 0.3, 0.1],
    ])
    clip = np.array([
        [0.6, 0.3, 0.1],
        [0.6, 0.3, 0.1],
        [0.3, 0.6, 0.1],
    ])
    result = shared_runner_up_candidate(task, clip)
    np.testing.assert_array_equal(result["agreement"], [True, True, False])
    np.testing.assert_array_equal(result["selected"], [True, False, False])
    assert result["candidate_mask"][0].tolist() == [True, True, False]
    assert result["candidate_mask"][1].tolist() == [True, False, False]


def test_gate_passes_only_with_coverage_gradient_and_class_safety():
    passing = evaluate_shared_runner_up_gate(
        input_contract_valid=True,
        selected_fraction_pct=20.0,
        selected_candidate_coverage_pct=99.0,
        selected_top1_miss_recovery_pct=60.0,
        delta_vs_top1_ci=(0.01, 0.03),
        delta_vs_zero_delay_ci=(0.02, 0.04),
        car_first_order_delta=0.01,
        person_first_order_delta=0.01,
        truck_first_order_delta=0.01,
        nonhard_first_order_delta=0.01,
        max_full_mass_shift_pp=0.5,
    )
    assert passing["decision"] == "PASS_SHARED_RUNNER_UP_PREFLIGHT"
    assert passing["training_authorized"] is False

    failing = evaluate_shared_runner_up_gate(
        input_contract_valid=True,
        selected_fraction_pct=20.0,
        selected_candidate_coverage_pct=99.0,
        selected_top1_miss_recovery_pct=60.0,
        delta_vs_top1_ci=(-0.01, 0.03),
        delta_vs_zero_delay_ci=(0.02, 0.04),
        car_first_order_delta=0.01,
        person_first_order_delta=0.01,
        truck_first_order_delta=-0.01,
        nonhard_first_order_delta=0.01,
        max_full_mass_shift_pp=0.5,
    )
    assert failing["decision"] == "REJECT"
    assert not failing["checks"]["first_order_gain_vs_top1_ci_lower_positive"]
    assert not failing["checks"]["truck_first_order_delta_nonnegative"]


def test_cloud_entrypoint_is_cpu_only_and_locks_before_oracle_labels():
    runner = (REPO_ROOT / "tools/run_visda_agreement_shared_runner_up_audit.sh").read_text()
    audit = (REPO_ROOT / "tools/audit_visda_agreement_shared_runner_up.py").read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "import torch" not in audit
    assert "import clip" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert audit.index("lock_path.write_text") < audit.index(
        "target_hash_matches ="
    )
    assert audit.index("lock_path.write_text") < audit.index(
        "_parse_labels_after_lock(args.target_list)"
    )
