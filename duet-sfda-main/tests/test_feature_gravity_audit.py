from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.feature_gravity_audit import (
    binary_auroc,
    classwise_gradient_mass,
    duet_logit_descent_components,
    evaluate_preflight_gate,
    fixed_tail_masks,
    gradient_projection_summary,
    stratified_bootstrap_auc_difference,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_detached_logit_components_remain_finite_for_extreme_logits() -> None:
    weak = torch.tensor(
        [[10_000.0, 0.0, -10_000.0], [1_000.0, 999.0, -1_000.0]]
    )
    strong = torch.tensor(
        [[9_999.0, 1.0, -9_999.0], [999.0, 1_000.0, -999.0]]
    )
    clip = torch.tensor(
        [[0.0, 10_000.0, -10_000.0], [1_000.0, -1_000.0, 999.0]]
    )

    components = duet_logit_descent_components(
        weak,
        strong,
        clip,
        con_weight=0.1,
        clip_weight=0.5,
        batch_size=32,
    )

    assert all(torch.isfinite(value).all() for value in components.values())
    torch.testing.assert_close(
        components["weak_prob"].sum(dim=1),
        torch.ones(2, dtype=torch.float64),
    )


def test_stabilized_components_match_released_kl_on_finite_inputs() -> None:
    weak = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.2, -0.4]], dtype=torch.float64)
    strong = torch.tensor([[1.5, 0.7, -0.5], [0.3, 0.9, -0.2]], dtype=torch.float64)
    clip = torch.tensor([[0.2, 1.1, -0.1], [0.8, -0.2, 0.4]], dtype=torch.float64)
    con_weight, clip_weight, batch_size = 0.1, 0.5, 32
    components = duet_logit_descent_components(
        weak,
        strong,
        clip,
        con_weight=con_weight,
        clip_weight=clip_weight,
        batch_size=batch_size,
    )

    weak_proxy = weak.detach().requires_grad_(True)
    strong_proxy = strong.detach().requires_grad_(True)
    weak_prob = torch.softmax(weak_proxy, dim=1)
    strong_prob = torch.softmax(strong_proxy, dim=1)
    consistency = F.kl_div(
        strong_prob.log(), weak_prob, reduction="none"
    ).sum(dim=1)
    clip_kl = F.kl_div(
        weak_prob.log(), torch.softmax(clip, dim=1), reduction="none"
    ).sum(dim=1)
    consistency_grad = torch.autograd.grad(
        con_weight * consistency.sum() / batch_size,
        (weak_proxy, strong_proxy),
        retain_graph=True,
    )
    clip_grad = torch.autograd.grad(
        clip_weight * clip_kl.sum() / batch_size, weak_proxy
    )[0]

    torch.testing.assert_close(components["consistency_per_sample"], consistency)
    torch.testing.assert_close(components["clip_per_sample"], clip_kl)
    torch.testing.assert_close(components["consistency_descent_weak"], -consistency_grad[0])
    torch.testing.assert_close(
        components["consistency_descent_strong"], -consistency_grad[1]
    )
    torch.testing.assert_close(components["clip_descent_weak"], -clip_grad)


def test_binary_auroc_and_fixed_tails_are_label_independent() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
    target = np.array([False, False, True, True, True])
    bottom, top = fixed_tail_masks(scores, fraction=0.2)

    assert binary_auroc(scores, target) == 1.0
    assert bottom.tolist() == [True, False, False, False, False]
    assert top.tolist() == [False, False, False, True, False]


def test_bootstrap_auc_difference_detects_stronger_feature_score() -> None:
    target = np.array([False] * 10 + [True] * 10)
    feature = np.arange(20, dtype=np.float64)
    confidence = np.zeros(20, dtype=np.float64)
    low, high = stratified_bootstrap_auc_difference(
        feature, confidence, target, repeats=100, seed=2020
    )

    assert low == 0.5
    assert high == 0.5


def test_gradient_projection_and_hard_class_gate() -> None:
    oracle = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    current = np.array([[-2.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
    candidate = np.array([[-1.0, 0.0], [1.9, 0.0], [-0.5, 0.0]])
    current_summary = gradient_projection_summary(current, oracle)
    candidate_summary = gradient_projection_summary(candidate, oracle)
    labels = np.array([0, 1, 2])
    classwise = classwise_gradient_mass(
        labels,
        ["car", "person", "truck"],
        current_summary["projection"],
        candidate_summary["projection"],
    )
    harmful_reduction = 100.0 * (
        current_summary["harmful_mass"] - candidate_summary["harmful_mass"]
    ) / current_summary["harmful_mass"]
    helpful_retention = (
        100.0 * candidate_summary["helpful_mass"] / current_summary["helpful_mass"]
    )
    gate = evaluate_preflight_gate(
        reproduction_passed=True,
        auc_gain=0.03,
        auc_ci=(0.01, 0.05),
        quintile_accuracy_gap_pp=6.0,
        harmful_reduction_percent=harmful_reduction,
        helpful_retention_percent=helpful_retention,
        classwise=classwise,
    )

    assert gate["decision"] == "PASS_OFFLINE_GATE"
    assert all(gate["checks"].values())


def test_gate_rejects_car_harmful_mass_increase() -> None:
    classwise = [
        {
            "class": name,
            "samples": 10,
            "current_harmful_mass": 1.0,
            "candidate_harmful_mass": 1.1 if name == "car" else 0.9,
        }
        for name in ("car", "person", "truck")
    ]
    gate = evaluate_preflight_gate(
        reproduction_passed=True,
        auc_gain=0.03,
        auc_ci=(0.01, 0.05),
        quintile_accuracy_gap_pp=6.0,
        harmful_reduction_percent=20.0,
        helpful_retention_percent=96.0,
        classwise=classwise,
    )

    assert gate["decision"] == "REJECT"
    assert not gate["checks"]["car_harmful_mass_nonincreasing"]


def test_cloud_entrypoint_cannot_start_training() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_feature_gravity_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_feature_gravity.py"
    ).read_text()

    assert "image_target_of_oh_vs.py" not in runner
    assert "optimizer.step" not in audit
    assert '"optimizer_steps": 0' in audit
    assert '"training_authorized": False' in audit
    assert audit.index("lock_path.write_text") < audit.rindex("_parse_labels_after_lock(")
