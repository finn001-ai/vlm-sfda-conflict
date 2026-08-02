from pathlib import Path

import numpy as np
import torch

from src.utils.pcgrad_feature_jacobian_audit import (
    classifier_probability,
    effective_weight_normalized_linear,
    evaluate_feature_jacobian_gate,
    map_joint_logit_descent_to_feature,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_effective_weight_normalization_recovers_row_magnitudes() -> None:
    vector = np.array([[3.0, 4.0], [0.0, 2.0]])
    magnitude = np.array([[10.0], [3.0]])
    weight = effective_weight_normalized_linear(vector, magnitude)
    np.testing.assert_allclose(np.linalg.norm(weight, axis=1), [10.0, 3.0])
    np.testing.assert_allclose(weight[0], [6.0, 8.0])


def test_effective_weight_matches_legacy_torch_weight_norm() -> None:
    layer = torch.nn.utils.weight_norm(torch.nn.Linear(3, 2), dim=0)
    vector = torch.tensor([[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]])
    magnitude = torch.tensor([[10.0], [3.0]])
    with torch.no_grad():
        layer.weight_v.copy_(vector)
        layer.weight_g.copy_(magnitude)
    layer(torch.zeros(1, 3))
    recovered = effective_weight_normalized_linear(
        vector.numpy(), magnitude.numpy()
    )
    np.testing.assert_allclose(recovered, layer.weight.detach().numpy())


def test_joint_logit_descent_maps_weak_and_strong_separately() -> None:
    weight = np.array([[1.0, 2.0], [-1.0, 3.0]])
    joint = np.array([[2.0, -1.0, 0.5, -0.5]])
    mapped = map_joint_logit_descent_to_feature(joint, weight)
    np.testing.assert_allclose(mapped, [[3.0, 1.0, 1.0, -0.5]])


def test_classifier_probability_replays_linear_softmax() -> None:
    feature = np.array([[1.0, 2.0]])
    weight = np.array([[1.0, 0.0], [0.0, 1.0]])
    bias = np.array([0.5, -0.5])
    probability = classifier_probability(feature, weight, bias)
    expected = np.exp([1.5, 1.5]) / np.exp([1.5, 1.5]).sum()
    np.testing.assert_allclose(probability[0], expected)


def _comparison(low: float = 0.01) -> dict:
    return {"mean_difference": 0.02, "paired_bootstrap_95_ci": [low, 0.03]}


def test_gate_only_requests_exact_control_audit() -> None:
    group = {"car": 0.01, "person": 0.01, "truck": 0.02, "other_nine": 0.01}
    gate = evaluate_feature_jacobian_gate(
        input_contract_valid=True,
        classifier_top1_reproduced=True,
        max_probability_replay_error=3e-4,
        overall_first_order=_comparison(),
        active_first_order=_comparison(),
        baseline_negative_burden=-0.2,
        candidate_negative_burden=-0.1,
        helpful_retention_pct=101.0,
        candidate_to_baseline_mean_norm_ratio=1.1,
        group_first_order_delta=group,
    )
    assert gate["decision"] == "NEEDS_EXACT_CONTROL_PARAMETER_AUDIT"
    assert gate["training_authorized"] is False
    assert gate["gpu_authorized"] is False

    failed = evaluate_feature_jacobian_gate(
        input_contract_valid=True,
        classifier_top1_reproduced=True,
        max_probability_replay_error=6e-4,
        overall_first_order=_comparison(),
        active_first_order=_comparison(-0.01),
        baseline_negative_burden=-0.2,
        candidate_negative_burden=-0.1,
        helpful_retention_pct=101.0,
        candidate_to_baseline_mean_norm_ratio=1.1,
        group_first_order_delta={**group, "truck": -0.01},
    )
    assert failed["decision"] == "REJECT"
    assert not failed["checks"]["max_probability_replay_error_at_most_5e_4"]
    assert not failed["checks"]["feature_active_first_order_gain_ci_lower_positive"]
    assert not failed["checks"]["truck_feature_first_order_delta_nonnegative"]


def test_cloud_entrypoint_loads_only_frozen_classifier_and_never_trains() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_pcgrad_feature_jacobian_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_pcgrad_feature_jacobian.py"
    ).read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "source_C.pt" in runner
    assert "source_F.pt" not in runner
    assert "source_B.pt" not in runner
    assert "image_target_of_oh_vs.py" not in runner
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert '"training_authorized": False' in audit
    assert '"gpu_authorized": False' in audit
    assert audit.index("lock_path.write_text") < audit.rindex(
        "_parse_labels_after_lock"
    )
