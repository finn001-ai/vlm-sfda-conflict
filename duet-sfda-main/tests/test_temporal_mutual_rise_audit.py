from pathlib import Path

import numpy as np

from src.utils.temporal_mutual_rise_audit import (
    MATCHED_BASELINES,
    centered_log_velocity,
    evaluate_temporal_mutual_rise_gate,
    route_mutual_rise,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_centered_log_velocity_preserves_log_odds_change() -> None:
    previous = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    current = np.array([[0.4, 0.5, 0.1], [0.1, 0.6, 0.3]])
    velocity = centered_log_velocity(previous, current)
    expected_pair_change = np.log(current[:, 0] / current[:, 1]) - np.log(
        previous[:, 0] / previous[:, 1]
    )
    np.testing.assert_allclose(velocity[:, 0] - velocity[:, 1], expected_pair_change)
    np.testing.assert_allclose(velocity.sum(axis=1), 0.0, atol=1e-12)


def test_mutual_rise_routes_and_swaps_only_two_clip_masses() -> None:
    previous_task = np.array([[0.80, 0.10, 0.10], [0.60, 0.30, 0.10]])
    current_task = np.array([[0.40, 0.50, 0.10], [0.65, 0.25, 0.10]])
    previous_clip = np.array([[0.90, 0.05, 0.05], [0.50, 0.40, 0.10]])
    current_clip = np.array([[0.55, 0.40, 0.05], [0.60, 0.30, 0.10]])
    candidates = np.array([[1, 0, 2, -1], [0, 1, 2, -1]])
    result = route_mutual_rise(
        previous_task,
        current_task,
        previous_clip,
        current_clip,
        candidates,
    )
    assert result["routed"].tolist() == [True, False]
    assert result["prediction"].tolist() == [1, 0]
    np.testing.assert_allclose(result["target_probability"][0], [0.40, 0.55, 0.05])
    np.testing.assert_allclose(result["target_probability"][1], current_clip[1])
    np.testing.assert_allclose(result["target_probability"].sum(axis=1), 1.0)
    assert result["selected_task_velocity"][0] > 0.0
    assert result["selected_clip_velocity"][0] > 0.0


def _comparisons(gain: float = 1.2, low: float = 0.4) -> dict:
    return {
        name: {
            "gain_pp": gain,
            "paired_bootstrap_95_ci_pp": [low, 1.9],
        }
        for name in MATCHED_BASELINES
    }


def test_gate_requires_gain_stability_and_hard_class_safety() -> None:
    kwargs = {
        "input_contract_valid": True,
        "route_coverage_pct": 20.0,
        "routed_union_decision_stability_pct": 95.0,
        "candidate_set_coverage_pct": 93.0,
        "minimum_class_candidate_coverage_pct": 87.0,
        "comparisons": _comparisons(),
        "best_baseline_name": "fixed_clip",
        "car_delta_pp": 0.2,
        "truck_delta_pp": 0.3,
        "car_truck_mean_delta_pp": 0.25,
        "other_ten_mean_delta_pp": 0.4,
        "max_full_target_mass_shift_pp": 0.7,
    }
    passing = evaluate_temporal_mutual_rise_gate(**kwargs)
    assert passing["decision"] == "PASS_TEMPORAL_MUTUAL_RISE_PREFLIGHT"
    assert passing["training_authorized"] is False
    assert passing["proxy_authorized"] is False
    assert passing["gpu_authorized"] is False

    kwargs["routed_union_decision_stability_pct"] = 85.0
    kwargs["truck_delta_pp"] = -0.8
    rejected = evaluate_temporal_mutual_rise_gate(**kwargs)
    assert rejected["decision"] == "REJECT"
    assert not rejected["checks"][
        "routed_union_decision_stability_at_least_90pct"
    ]
    assert not rejected["checks"]["truck_regression_at_most_0_5pp"]


def test_entrypoint_is_cpu_only_and_locks_before_oracle() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_temporal_mutual_rise_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_temporal_mutual_rise.py"
    ).read_text()
    helper = (REPO_ROOT / "src/utils/temporal_mutual_rise_audit.py").read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "source_C.pt" not in runner
    assert "source_F.pt" not in runner
    assert "import torch" not in audit
    assert "import clip" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert '"training_authorized": False' in helper
    assert audit.index("lock_path.write_text") < audit.rindex(
        "labels = _parse_labels_after_lock"
    )
    assert audit.index("lock_path.write_text") < audit.rindex(
        'snapshot1["target_label"]'
    )
