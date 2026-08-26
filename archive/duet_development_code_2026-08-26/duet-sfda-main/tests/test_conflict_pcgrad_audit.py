from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.conflict_pcgrad_audit import (
    decision_stability,
    direction_stability,
    duet_output_descent_components,
    evaluate_conflict_pcgrad_gate,
    symmetric_pcgrad,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_duet_components_match_autograd_on_finite_probabilities() -> None:
    weak_logits = torch.tensor(
        [[1.2, -0.1, 0.4], [0.2, 0.8, -0.3]], dtype=torch.float64,
        requires_grad=True,
    )
    strong_logits = torch.tensor(
        [[0.9, 0.2, 0.1], [-0.2, 1.0, 0.1]], dtype=torch.float64,
        requires_grad=True,
    )
    clip_logits = torch.tensor(
        [[0.1, 0.7, -0.2], [0.8, -0.1, 0.3]], dtype=torch.float64
    )
    weak = torch.softmax(weak_logits, dim=1)
    strong = torch.softmax(strong_logits, dim=1)
    clip = torch.softmax(clip_logits, dim=1)
    consistency = 0.2 * F.kl_div(
        strong.log(), weak, reduction="sum"
    )
    clip_kl = 0.4 * F.kl_div(weak.log(), clip, reduction="sum")
    con_gradient = torch.autograd.grad(
        consistency, (weak_logits, strong_logits), retain_graph=True
    )
    clip_gradient = torch.autograd.grad(clip_kl, weak_logits)[0]

    result = duet_output_descent_components(
        weak.detach().numpy(), strong.detach().numpy(), clip.numpy()
    )
    np.testing.assert_allclose(result["consistency_weak"], -con_gradient[0].numpy())
    np.testing.assert_allclose(
        result["consistency_strong"], -con_gradient[1].numpy()
    )
    np.testing.assert_allclose(result["clip_weak"], -clip_gradient.numpy())


def test_symmetric_pcgrad_projects_only_negative_dot_rows() -> None:
    first = np.array([[1.0, 0.0], [1.0, 0.0]])
    second = np.array([[-1.0, 1.0], [1.0, 1.0]])
    result = symmetric_pcgrad(first, second)

    np.testing.assert_array_equal(result["gradient_conflict"], [True, False])
    first_dot = np.einsum(
        "ij,ij->i", result["first_projected"], second
    )
    second_dot = np.einsum(
        "ij,ij->i", result["second_projected"], first
    )
    assert abs(first_dot[0]) < 1e-12
    assert abs(second_dot[0]) < 1e-12
    np.testing.assert_allclose(result["candidate_joint"][1], first[1] + second[1])


def _comparison(low: float = 0.01) -> dict:
    return {"mean_difference": 0.02, "paired_bootstrap_95_ci": [low, 0.03]}


def test_gate_can_only_request_parameter_audit_and_checks_hard_groups() -> None:
    gate = evaluate_conflict_pcgrad_gate(
        input_contract_valid=True,
        conflict_coverage_pct=20.0,
        floor_decision_stability_pct=99.0,
        floor_direction_stability_pct=99.9,
        floor_mean_norm_ratio_max_deviation=0.01,
        overall_first_order=_comparison(),
        conflict_first_order=_comparison(),
        baseline_negative_burden=-0.2,
        candidate_negative_burden=-0.1,
        helpful_retention_pct=101.0,
        candidate_to_baseline_mean_norm_ratio=1.1,
        group_first_order_delta={
            "car": 0.01,
            "person": 0.01,
            "truck": 0.0,
            "other_nine": 0.02,
        },
    )
    assert gate["decision"] == "NEEDS_PARAMETER_AUDIT"
    assert gate["training_authorized"] is False
    assert gate["proxy_authorized"] is False

    failed = evaluate_conflict_pcgrad_gate(
        input_contract_valid=True,
        conflict_coverage_pct=20.0,
        floor_decision_stability_pct=99.0,
        floor_direction_stability_pct=99.9,
        floor_mean_norm_ratio_max_deviation=0.01,
        overall_first_order=_comparison(),
        conflict_first_order=_comparison(-0.01),
        baseline_negative_burden=-0.2,
        candidate_negative_burden=-0.1,
        helpful_retention_pct=101.0,
        candidate_to_baseline_mean_norm_ratio=1.1,
        group_first_order_delta={
            "car": 0.01,
            "person": 0.01,
            "truck": -0.01,
            "other_nine": 0.02,
        },
    )
    assert failed["decision"] == "REJECT"
    assert not failed["checks"]["conflict_subset_first_order_gain_ci_lower_positive"]
    assert not failed["checks"]["truck_first_order_delta_nonnegative"]


def test_underflow_floor_stability_is_explicit() -> None:
    weak = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    strong = np.array([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    clip = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
    masks = []
    for floor in (np.nextafter(0.0, 1.0), np.finfo(np.float32).tiny, 1e-30):
        components = duet_output_descent_components(
            weak, strong, clip, probability_floor=float(floor)
        )
        masks.append(
            symmetric_pcgrad(
                components["consistency_joint"], components["clip_joint"]
            )["gradient_conflict"]
        )
    assert decision_stability(masks[0], masks[1]) == 100.0
    assert decision_stability(masks[0], masks[2]) == 100.0
    assert direction_stability(np.eye(2), np.eye(2)) == 100.0


def test_cloud_entrypoint_is_cpu_only_and_never_authorizes_training() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_pcgrad_audit.sh"
    ).read_text()
    audit = (REPO_ROOT / "tools/audit_visda_conflict_pcgrad.py").read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "image_target_of_oh_vs.py" not in runner
    assert "import torch" not in audit
    assert "import clip" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert '"training_authorized": False' in audit
    assert '"proxy_authorized": False' in audit
    assert audit.index("lock_path.write_text") < audit.rindex(
        "_parse_labels_after_lock"
    )
