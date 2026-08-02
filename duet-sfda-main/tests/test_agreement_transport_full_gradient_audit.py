from pathlib import Path

import numpy as np

from src.utils.agreement_transport_full_gradient_audit import (
    agreement_transport_joint_descents,
    evaluate_agreement_transport_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_gradient_keeps_conflict_kl_and_routes_agreements() -> None:
    weak = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    strong = np.array([[0.7, 0.3], [0.4, 0.6], [0.55, 0.45]])
    clip = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
    transport = np.array([[0.6, 0.4], [0.1, 0.9], [0.95, 0.05]])
    label = np.array([0, 1, 0])
    agreement = np.array([True, True, False])
    result = agreement_transport_joint_descents(
        weak,
        strong,
        clip,
        label,
        agreement,
        transport,
        np.arange(3),
        batch_size=3,
    )
    class_count = 2
    baseline = result["duet_joint"][:, :class_count]
    candidate = result["agreement_transport_joint"][:, :class_count]
    duplicate = result["duplicate_hard_ce_joint"][:, :class_count]
    np.testing.assert_allclose(candidate[~agreement], baseline[~agreement])
    np.testing.assert_allclose(duplicate[~agreement], baseline[~agreement])
    assert not np.allclose(candidate[agreement], baseline[agreement])
    assert not np.allclose(candidate[agreement], duplicate[agreement])
    for value in result.values():
        np.testing.assert_allclose(value[:, :class_count].sum(axis=1), 0.0, atol=1e-15)
        np.testing.assert_allclose(value[:, class_count:].sum(axis=1), 0.0, atol=1e-15)


def _comparisons(low: float = 0.01) -> dict:
    result = {"mean_difference": 0.02, "paired_bootstrap_95_ci": [low, 0.03]}
    return {
        control: {
            scope: {metric: dict(result) for metric in ("first_order", "cosine")}
            for scope in ("overall", "agreement")
        }
        for control in ("original_duet", "duplicate_hard_ce")
    }


def test_gate_never_authorizes_training_or_gpu() -> None:
    kwargs = {
        "input_contract_valid": True,
        "max_sinkhorn_marginal_error": 1e-8,
        "minimum_target_replay_median_cosine": 0.95,
        "mean_transport_ce_component_cosine": 0.2,
        "comparisons": _comparisons(),
        "every_replay_agreement_first_order_gain_positive": {
            "original_duet": True,
            "duplicate_hard_ce": True,
        },
        "candidate_negative_burden": -0.01,
        "strongest_control_negative_burden": -0.02,
        "candidate_to_strongest_mean_norm_ratio": 1.1,
        "group_first_order_delta_vs_strongest": {
            "car": 0.1,
            "person": 0.1,
            "truck": 0.1,
            "other_nine": 0.1,
        },
    }
    gate = evaluate_agreement_transport_gate(**kwargs)
    assert gate["decision"] == "NEEDS_EXACT_PARAMETER_AUDIT"
    assert gate["training_authorized"] is False
    assert gate["proxy_authorized"] is False
    assert gate["gpu_authorized"] is False

    kwargs["comparisons"] = _comparisons(low=-0.01)
    rejected = evaluate_agreement_transport_gate(**kwargs)
    assert rejected["decision"] == "REJECT"


def test_entrypoint_is_cpu_only_and_locks_before_oracle() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_agreement_transport_full_gradient_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_agreement_transport_full_gradient.py"
    ).read_text()
    helper = (
        REPO_ROOT / "src/utils/agreement_transport_full_gradient_audit.py"
    ).read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "source_C.pt" in runner
    assert "source_F.pt" not in runner
    assert "source_B.pt" not in runner
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert '"training_authorized": False' in helper
    assert audit.index("lock_path.write_text") < audit.rindex(
        "labels = _parse_labels_after_lock"
    )
    assert audit.index("lock_path.write_text") < audit.rindex(
        'embedded_labels = np.asarray(snapshot["target_label"]'
    )
