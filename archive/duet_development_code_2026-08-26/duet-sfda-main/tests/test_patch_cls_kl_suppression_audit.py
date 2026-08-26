from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.patch_cls_kl_suppression_audit import (
    consistency_logit_descent,
    evaluate_patch_cls_kl_suppression_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_consistency_descent_matches_exact_torch_autograd() -> None:
    weak_logits = torch.tensor(
        [[0.2, -0.4, 1.1], [1.5, -0.2, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    strong_logits = torch.tensor(
        [[-0.1, 0.5, 0.7], [1.0, 0.1, -0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    weak = torch.softmax(weak_logits, dim=1)
    strong = torch.softmax(strong_logits, dim=1)
    per_sample = F.kl_div(
        torch.log_softmax(strong_logits, dim=1),
        torch.log_softmax(weak_logits, dim=1),
        reduction="none",
        log_target=True,
    ).sum(dim=1)
    row_batch_size = torch.tensor([64.0, 28.0], dtype=torch.float64)
    loss = torch.sum(0.2 * per_sample / row_batch_size)
    weak_grad, strong_grad = torch.autograd.grad(loss, (weak_logits, strong_logits))
    observed_weak, observed_strong = consistency_logit_descent(
        weak.detach().numpy(),
        strong.detach().numpy(),
        row_batch_size.numpy(),
        weight=0.2,
    )
    assert np.allclose(observed_weak, -weak_grad.detach().numpy(), atol=1e-14)
    assert np.allclose(observed_strong, -strong_grad.detach().numpy(), atol=1e-14)


def _paired(mean: float = 0.1, low: float = 0.01) -> dict:
    return {
        "mean_difference": mean,
        "paired_bootstrap_95_ci": [low, mean + 0.1],
    }


def _gate(**overrides):
    arguments = {
        "input_contract_valid": True,
        "heldout_selector_passed": True,
        "selected_coverage_pct": 2.9,
        "strong_replay_max_abs_error": 1e-10,
        "output_first_order": _paired(),
        "feature_first_order": _paired(),
        "output_negative_burden_baseline": -0.2,
        "output_negative_burden_candidate": -0.1,
        "feature_negative_burden_baseline": -0.2,
        "feature_negative_burden_candidate": -0.1,
        "feature_helpful_retention_pct": 105.0,
        "feature_mean_norm_ratio": 0.9,
        "class_macro_feature_first_order_delta": 0.01,
        "heldout_accuracy_gain_vs_clip_pp": 1.6,
        "heldout_accuracy_ci_lower_pp": 1.4,
        "car_accuracy_delta_pp": 2.4,
        "truck_accuracy_delta_pp": -0.19,
    }
    arguments.update(overrides)
    return evaluate_patch_cls_kl_suppression_gate(**arguments)


def test_gate_pass_only_authorizes_exact_parameter_audit() -> None:
    gate = _gate()
    assert gate["decision"] == "NEEDS_EXACT_PARAMETER_AUDIT"
    assert gate["exact_parameter_audit_authorized"] is True
    assert gate["proxy_authorized"] is False
    assert gate["training_authorized"] is False


def test_gate_rejects_negative_feature_ci_or_truck_exchange() -> None:
    gate = _gate(feature_first_order=_paired(mean=0.1, low=-0.01))
    assert gate["decision"] == "REJECT"
    assert not gate["checks"]["feature_first_order_gain_ci_lower_positive"]
    gate = _gate(truck_accuracy_delta_pp=-0.6)
    assert gate["decision"] == "REJECT"
    assert not gate["checks"]["truck_accuracy_regression_at_most_0_5pp"]


def test_entrypoint_is_cpu_only_and_locks_before_oracle() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_patch_cls_kl_suppression_impact_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_patch_cls_kl_suppression_impact.py"
    ).read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "netF" not in audit
    assert "netB" not in audit
    assert "clip.load" not in audit
    assert ".backward(" not in audit
    assert "optimizer.step" not in audit
    assert audit.index("lock_path.write_text") < audit.rindex(
        "labels = _read_selected_labels_after_lock"
    )
    assert audit.index("lock_path.write_text") < audit.rindex(
        "holdout_summary = json.loads"
    )
