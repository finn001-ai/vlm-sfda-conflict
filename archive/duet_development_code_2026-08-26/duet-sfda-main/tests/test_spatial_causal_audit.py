from pathlib import Path

import numpy as np

from src.utils.spatial_causal_audit import (
    balanced_binary_masks,
    deterministic_hash_sample,
    evaluate_spatial_gate,
    spatial_consensus_selector,
    topk_union_candidates,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def test_balanced_mask_halves_are_deterministic_and_exact() -> None:
    masks = balanced_binary_masks(mask_count=64, grid_size=7, seed=2020)
    repeated = balanced_binary_masks(mask_count=64, grid_size=7, seed=2020)

    np.testing.assert_array_equal(masks, repeated)
    assert masks.shape == (64, 7, 7)
    np.testing.assert_allclose(masks[:32].mean(axis=0), 0.5)
    np.testing.assert_allclose(masks[32:].mean(axis=0), 0.5)


def test_hash_sample_uses_only_paths_and_prediction_mask() -> None:
    paths = [f"image_{index}.jpg" for index in range(10)]
    eligible = np.array([True, False, True, True, False, True, True, False, True, True])
    first = deterministic_hash_sample(
        paths, eligible, count=4, namespace="conflict"
    )
    second = deterministic_hash_sample(
        paths, eligible, count=4, namespace="conflict"
    )

    np.testing.assert_array_equal(first, second)
    assert eligible[first].all()
    assert len(np.unique(first)) == 4


def test_top2_union_is_stable_and_deduplicated() -> None:
    task = np.array([[0.6, 0.3, 0.1, 0.0], [0.1, 0.4, 0.3, 0.2]])
    clip = np.array([[0.1, 0.7, 0.2, 0.0], [0.5, 0.3, 0.1, 0.1]])
    candidates = topk_union_candidates(task, clip, top_k=2)

    assert candidates.tolist() == [[0, 1, 2, -1], [1, 2, 0, -1]]


def test_spatial_consensus_prefers_shared_class_specific_region() -> None:
    masks = balanced_binary_masks(mask_count=64, grid_size=2, seed=2020)
    flat = masks.reshape(64, -1) - 0.5
    task_logits = np.stack(
        (4.0 * flat[:, 0], 4.0 * flat[:, 2], np.zeros(64)), axis=1
    )
    clip_logits = np.stack(
        (4.0 * flat[:, 1], 4.0 * flat[:, 2], np.zeros(64)), axis=1
    )
    task_probability = _softmax(task_logits)[None]
    clip_probability = _softmax(clip_logits)[None]
    candidates = np.array([[0, 1, 2, -1]])
    result = spatial_consensus_selector(
        task_probability,
        clip_probability,
        candidates,
        masks,
        np.array([0]),
    )

    assert result["prediction"].tolist() == [1]
    assert result["candidate_score"][0, 1] > result["candidate_score"][0, 0]
    assert result["has_spatial_choice"].tolist() == [True]


def test_gate_passes_only_with_stable_gain_and_no_car_truck_exchange() -> None:
    passing = evaluate_spatial_gate(
        reproduction_passed=True,
        median_split_half_stability=0.9,
        balanced_gain_pp=2.0,
        balanced_ci=(0.1, 3.9),
        car_truck_gain_pp=3.2,
        eligible_rescue_rate=25.0,
        car_net_corrections=2,
        truck_net_corrections=1,
        changed_vs_clip_coverage=10.0,
    )
    exchange = evaluate_spatial_gate(
        reproduction_passed=True,
        median_split_half_stability=0.9,
        balanced_gain_pp=2.0,
        balanced_ci=(0.1, 3.9),
        car_truck_gain_pp=3.2,
        eligible_rescue_rate=25.0,
        car_net_corrections=2,
        truck_net_corrections=-1,
        changed_vs_clip_coverage=10.0,
    )

    assert passing["decision"] == "PASS_OFFLINE_GATE"
    assert exchange["decision"] == "REJECT"
    assert not exchange["checks"]["truck_net_corrections_nonnegative"]


def test_cloud_entrypoint_is_frozen_forward_only_and_locks_before_labels() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_spatial_causal_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_spatial_causal.py"
    ).read_text()

    assert "image_target_of_oh_vs.py" not in runner
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert 'str(cfg.ACTIVE.ARCH) != "ViT-B/32"' in audit
    assert '"optimizer_steps": 0' in audit
    assert '"training_authorized": False' in audit
    assert audit.index('replay_weak_x = inputs[1].to(device)') < audit.index(
        'weak_x = replay_weak_x[positions]'
    )
    assert '"complete_loader_batch_before_pilot_selection": True' in audit
    assert audit.index("lock_path.write_text") < audit.rindex(
        "_parse_labels_after_lock("
    )
