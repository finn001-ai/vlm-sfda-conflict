from pathlib import Path

import numpy as np
import torch

from src.utils.vsfot_alignment_audit import (
    evaluate_vsfot_alignment_gate,
    log_sinkhorn,
    row_cosine,
    vsfot_alignment_feature_descent,
    vsfot_transport_probability,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_log_sinkhorn_matches_both_marginals() -> None:
    source = np.array([0.2, 0.3, 0.5])
    target = np.full(4, 0.25)
    cost = np.array([[0.1, 1.0, 2.0, 0.5], [0.3, 0.2, 0.9, 1.2], [1.0, 0.4, 0.1, 0.2]])
    result = log_sinkhorn(source, target, cost)
    np.testing.assert_allclose(result["plan"].sum(axis=1), source, atol=1e-8)
    np.testing.assert_allclose(result["plan"].sum(axis=0), target, atol=1e-8)
    assert result["max_marginal_error"] <= 1e-8


def test_log_sinkhorn_dual_refinement_handles_extreme_marginals() -> None:
    source = np.array([0.999998, 1e-6, 1e-6])
    target = np.full(8, 1.0 / 8.0)
    cost = np.array(
        [
            [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0],
            [140.0, 120.0, 100.0, 80.0, 60.0, 40.0, 20.0, 0.0],
            [70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0],
        ]
    )
    result = log_sinkhorn(source, target, cost, iterations=5)
    np.testing.assert_allclose(result["plan"].sum(axis=1), source, atol=1e-7)
    np.testing.assert_allclose(result["plan"].sum(axis=0), target, atol=1e-7)
    assert result["iterations"] > 5 or result["dual_refined"]


def test_zero_direction_has_zero_cosine_instead_of_crashing() -> None:
    result = row_cosine(np.array([[0.0, 0.0], [1.0, 0.0]]), np.eye(2))
    np.testing.assert_allclose(result, [0.0, 0.0])


def test_vsfot_alignment_is_finite_and_additive() -> None:
    rng = np.random.default_rng(5)
    sample_count, class_count, feature_count = 9, 3, 4
    task = rng.dirichlet(np.ones(class_count), sample_count)
    clip = rng.dirichlet(np.ones(class_count), sample_count)
    feature = rng.normal(size=(sample_count, feature_count))
    weight = rng.normal(size=(class_count, feature_count))
    result = vsfot_alignment_feature_descent(
        task,
        clip,
        feature,
        weight,
        np.array([1.0, 2.0, 3.0]),
        rng.permutation(sample_count),
        batch_size=4,
    )
    for key in ("classification_descent", "prototype_descent", "combined_descent"):
        assert result[key].shape == feature.shape
        assert np.isfinite(result[key]).all()
    np.testing.assert_allclose(
        result["combined_descent"],
        result["classification_descent"] + result["prototype_descent"],
    )
    assert not np.allclose(result["prototype_descent"], 0.0)
    assert result["max_sinkhorn_marginal_error"] <= 1e-6


def test_vsfot_alignment_matches_task_cost_autograd() -> None:
    rng = np.random.default_rng(17)
    sample_count, class_count, feature_count = 5, 3, 4
    feature = rng.normal(size=(sample_count, feature_count))
    weight = rng.normal(size=(class_count, feature_count))
    bias = rng.normal(size=class_count)
    logits = feature @ weight.T + bias
    task = np.exp(logits - logits.max(axis=1, keepdims=True))
    task /= task.sum(axis=1, keepdims=True)
    clip = rng.dirichlet(np.ones(class_count), sample_count)
    class_scale = np.array([1.0, 1.5, 2.0])
    result = vsfot_alignment_feature_descent(
        task,
        clip,
        feature,
        weight,
        class_scale,
        np.arange(sample_count),
        batch_size=sample_count,
    )

    clip_distance = 1.0 - clip.T
    clip_cost = clip_distance / clip_distance.max() - np.log(clip.T + 1e-6)
    coupling = log_sinkhorn(
        clip.mean(axis=0),
        np.full(sample_count, 1.0 / sample_count),
        clip_cost,
    )["plan"].T
    weighted_coupling = torch.tensor(coupling * class_scale[None, :])
    feature_tensor = torch.tensor(feature, dtype=torch.float64, requires_grad=True)
    weight_tensor = torch.tensor(weight, dtype=torch.float64)
    bias_tensor = torch.tensor(bias, dtype=torch.float64)
    probability = torch.softmax(feature_tensor @ weight_tensor.T + bias_tensor, dim=1)
    cosine = torch.nn.functional.cosine_similarity(
        feature_tensor[:, None, :], weight_tensor[None, :, :], dim=2
    )
    distance = 1.0 - cosine
    cost = distance / distance.max() - torch.log(probability + 1e-6)
    loss = torch.sum(weighted_coupling * cost)
    loss.backward()
    np.testing.assert_allclose(
        result["combined_descent"],
        -feature_tensor.grad.detach().numpy(),
        rtol=2e-5,
        atol=2e-7,
    )


def test_transport_probability_is_row_normalized_and_replay_complete() -> None:
    rng = np.random.default_rng(23)
    clip = rng.dirichlet(np.ones(3), 11)
    task = rng.dirichlet(np.ones(3), 11)
    result = vsfot_transport_probability(clip, task, rng.permutation(11), batch_size=4)
    assert result["probability"].shape == clip.shape
    np.testing.assert_allclose(result["probability"].sum(axis=1), 1.0)
    assert np.all(result["probability"] >= 0.0)
    assert result["max_sinkhorn_marginal_error"] <= 1e-6


def _comparisons(low: float = 0.01) -> dict:
    result = {
        "mean_difference": 0.02,
        "paired_bootstrap_95_ci": [low, 0.03],
    }
    return {
        name: {scope: dict(result) for scope in ("overall", "conflict")}
        for name in ("clip_kl", "transport_classification_only")
    }


def test_gate_passes_only_as_offline_preflight() -> None:
    kwargs = {
        "input_contract_valid": True,
        "max_sinkhorn_marginal_error": 1e-8,
        "minimum_replay_median_cosine": 0.95,
        "comparisons": _comparisons(),
        "every_replay_conflict_gain_vs_clip_positive": True,
        "candidate_negative_burden": -0.02,
        "clip_negative_burden": -0.03,
        "group_delta_vs_clip": {
            "car": 0.01,
            "person": 0.02,
            "truck": 0.01,
            "other_nine": 0.02,
        },
    }
    gate = evaluate_vsfot_alignment_gate(**kwargs)
    assert gate["decision"] == "PASS_VSFOT_ALIGNMENT_PREFLIGHT"
    assert gate["training_authorized"] is False
    assert gate["proxy_authorized"] is False
    assert gate["gpu_authorized"] is False

    kwargs["comparisons"] = _comparisons(low=-0.01)
    rejected = evaluate_vsfot_alignment_gate(**kwargs)
    assert rejected["decision"] == "REJECT"
    assert not rejected["checks"]["conflict_gain_vs_duet_clip_kl_ci_lower_positive"]


def test_cloud_entrypoint_is_cpu_only_and_locks_before_oracle() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_vsfot_alignment_audit.sh"
    ).read_text()
    audit = (REPO_ROOT / "tools/audit_visda_conflict_vsfot_alignment.py").read_text()
    helper = (REPO_ROOT / "src/utils/vsfot_alignment_audit.py").read_text()
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
