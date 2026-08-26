from pathlib import Path

import numpy as np

from src.utils.agreement_revocation_audit import (
    evaluate_agreement_revocation_gate,
    normalized_mask_weight,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_NAMES = {
    "task_confidence",
    "clip_confidence",
    "arithmetic_confidence",
    "rms_confidence",
}


def test_normalized_mask_weight_has_population_mean_one():
    mask = np.array([True, False, True, True, False])
    weight = normalized_mask_weight(mask)
    assert np.isclose(weight.mean(), 1.0)
    assert np.all(weight[~mask] == 0.0)
    assert np.all(weight[mask] > 0.0)


def test_gate_passes_only_with_matched_and_classwise_evidence():
    intervals = {name: (0.01, 0.03) for name in CONFIDENCE_NAMES}
    passing = evaluate_agreement_revocation_gate(
        input_contract_valid=True,
        stale_fraction_of_admitted_pct=10.0,
        stale_error_enrichment=3.0,
        captured_error_gains={name: 3 for name in CONFIDENCE_NAMES},
        precision_gain_cis=intervals,
        retained_accuracy_gain_pp=0.4,
        first_order_delta_vs_baseline_ci=(0.01, 0.03),
        first_order_delta_vs_confidence_cis=intervals,
        car_first_order_delta=0.01,
        person_first_order_delta=0.01,
        truck_first_order_delta=0.01,
        nonhard_first_order_delta=0.01,
        max_class_mass_shift_pp=0.5,
    )
    assert passing["decision"] == "PASS_AGREEMENT_REVOCATION_PREFLIGHT"
    assert passing["training_authorized"] is False

    failing = evaluate_agreement_revocation_gate(
        input_contract_valid=True,
        stale_fraction_of_admitted_pct=10.0,
        stale_error_enrichment=3.0,
        captured_error_gains={name: 3 for name in CONFIDENCE_NAMES},
        precision_gain_cis=intervals,
        retained_accuracy_gain_pp=0.4,
        first_order_delta_vs_baseline_ci=(-0.01, 0.03),
        first_order_delta_vs_confidence_cis=intervals,
        car_first_order_delta=0.01,
        person_first_order_delta=0.01,
        truck_first_order_delta=-0.01,
        nonhard_first_order_delta=0.01,
        max_class_mass_shift_pp=0.5,
    )
    assert failing["decision"] == "REJECT"
    assert not failing["checks"][
        "first_order_gain_vs_monotonic_mask_ci_lower_positive"
    ]
    assert not failing["checks"]["truck_first_order_delta_nonnegative"]


def test_cloud_entrypoint_is_cpu_only_and_locks_before_oracle_labels():
    runner = (REPO_ROOT / "tools/run_visda_agreement_revocation_audit.sh").read_text()
    audit = (REPO_ROOT / "tools/audit_visda_agreement_revocation.py").read_text()
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
