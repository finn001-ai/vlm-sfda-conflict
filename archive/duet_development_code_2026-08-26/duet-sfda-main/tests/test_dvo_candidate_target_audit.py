from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils import IID_losses
from src.utils.dvo_candidate_target_audit import (
    _fixed_q_tmi_loss,
    evaluate_dvo_candidate_target_gate,
    support_conditioned_mixed_target,
    tmi_logit_descent_replays,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _probability(rows: list[list[float]]) -> np.ndarray:
    result = np.asarray(rows, dtype=np.float64)
    return result / result.sum(axis=1, keepdims=True)


def test_support_conditioning_is_conflict_only_and_top1_preserving():
    task = _probability(
        [
            [0.70, 0.20, 0.05, 0.05],
            [0.50, 0.35, 0.10, 0.05],
            [0.05, 0.65, 0.20, 0.10],
        ]
    )
    clip = _probability(
        [
            [0.60, 0.25, 0.10, 0.05],
            [0.20, 0.30, 0.45, 0.05],
            [0.10, 0.55, 0.25, 0.10],
        ]
    )
    active = np.array([False, True, False])
    result = support_conditioned_mixed_target(task, clip, active)

    assert np.array_equal(
        result["candidate_probability"][~active],
        result["baseline_probability"][~active],
    )
    assert np.array_equal(
        result["candidate_probability"].argmax(axis=1),
        result["baseline_probability"].argmax(axis=1),
    )
    assert np.allclose(result["candidate_probability"].sum(axis=1), 1.0)
    assert np.all(
        result["candidate_probability"][active][~result["support"][active]] == 0.0
    )
    assert 0.0 < result["retained_mass"].item() <= 1.0


def test_support_conditioning_requires_an_active_conflict():
    probability = _probability([[0.7, 0.3], [0.4, 0.6]])
    with pytest.raises(ValueError, match="at least one"):
        support_conditioned_mixed_target(
            probability, probability, np.zeros(2, dtype=bool)
        )


def test_fixed_q_loss_matches_released_tmi_objective_and_replay_is_stable():
    prediction = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=torch.float64)
    target = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], dtype=torch.float64)
    expected, returned_q = IID_losses.tsallis_mutual_info(
        prediction, target, q_value=1.05, beta=1.0
    )
    actual = _fixed_q_tmi_loss(prediction, target, q_value=1.05)
    assert returned_q == 1.05
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    clip = prediction.detach().numpy()
    target_probability = target.detach().numpy()
    replay = tmi_logit_descent_replays(
        clip,
        target_probability,
        permutation_seeds=(7, 11),
        batch_size=2,
    )
    assert replay["descent_by_replay"].shape == (2, 2, 3)
    assert np.isfinite(replay["descent_by_replay"]).all()
    assert np.allclose(replay["descent_by_replay"].sum(axis=2), 0.0)
    assert np.allclose(replay["descent_by_replay"][0], replay["descent_by_replay"][1])


def test_gate_requires_robust_and_classwise_positive_evidence():
    comparisons = {
        name: {
            "mean_difference": 0.02,
            "paired_bootstrap_95_ci": (0.01, 0.03),
        }
        for name in ("cosine", "oracle_unit_projection", "first_order")
    }
    passing = evaluate_dvo_candidate_target_gate(
        input_contract_valid=True,
        target_top1_unchanged=True,
        mean_retained_mass=0.95,
        mean_support_size=3.4,
        oracle_candidate_coverage_pct=94.0,
        comparisons=comparisons,
        minimum_replay_first_order_delta=0.01,
        macro_first_order_delta=0.01,
        hard_class_first_order_delta={"car": 0.01, "person": 0.01, "truck": 0.01},
        other_nine_first_order_delta=0.01,
        candidate_negative_burden=0.1,
        baseline_negative_burden=0.2,
        max_class_mass_shift_pp=0.5,
    )
    assert passing["decision"] == "PASS_DVO_CANDIDATE_TARGET_PREFLIGHT"
    assert passing["training_authorized"] is False

    failing = evaluate_dvo_candidate_target_gate(
        input_contract_valid=True,
        target_top1_unchanged=True,
        mean_retained_mass=0.95,
        mean_support_size=3.4,
        oracle_candidate_coverage_pct=94.0,
        comparisons=comparisons,
        minimum_replay_first_order_delta=-0.001,
        macro_first_order_delta=0.01,
        hard_class_first_order_delta={"car": -0.01, "person": 0.01, "truck": 0.01},
        other_nine_first_order_delta=0.01,
        candidate_negative_burden=0.1,
        baseline_negative_burden=0.2,
        max_class_mass_shift_pp=0.5,
    )
    assert failing["decision"] == "REJECT"
    assert not failing["checks"]["every_replay_first_order_delta_positive"]
    assert not failing["checks"]["car_first_order_delta_nonnegative"]


def test_cloud_entrypoint_is_cpu_only_and_locks_before_oracle_labels():
    runner = (REPO_ROOT / "tools/run_visda_dvo_candidate_target_audit.sh").read_text()
    audit = (REPO_ROOT / "tools/audit_visda_dvo_candidate_target.py").read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "import clip" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert audit.index("lock_path.write_text") < audit.index(
        "_parse_labels_after_lock(args.target_list)"
    )
    assert audit.index("lock_path.write_text") < audit.index('snapshot["target_label"]')
