from pathlib import Path

import numpy as np

from src.utils.prototype_transport_audit import (
    capacity_preserving_transport,
    evaluate_prototype_transport_gate,
    prototype_cosine,
    row_ordinal_cost,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_row_ordinal_cost_is_parameter_free_and_deterministic() -> None:
    score = np.array([[0.2, 0.9, 0.5], [3.0, 1.0, 2.0]])
    np.testing.assert_array_equal(
        row_ordinal_cost(score),
        [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
    )


def test_prototype_cosine_normalizes_feature_and_class_rows() -> None:
    feature = np.array([[2.0, 0.0], [0.0, 3.0]])
    weight = np.array([[5.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(prototype_cosine(feature, weight), np.eye(2))


def test_transport_preserves_exact_integer_class_quota() -> None:
    cost = np.array(
        [
            [0.0, 10.0],
            [0.0, 5.0],
            [2.0, 0.0],
            [8.0, 0.0],
        ]
    )
    result = capacity_preserving_transport(cost, np.array([2, 2]))
    np.testing.assert_array_equal(result["prediction"], [0, 0, 1, 1])
    np.testing.assert_array_equal(result["plan"].sum(axis=0), [2.0, 2.0])
    assert result["integrality_max_error"] <= 1e-6


def _comparisons(gain: float = 1.2, low: float = 0.2) -> dict:
    return {
        name: {
            "gain_pp": gain,
            "paired_bootstrap_95_ci_pp": [low, 2.0],
        }
        for name in (
            "fixed_task",
            "fixed_clip",
            "confidence_choice",
            "arithmetic",
            "rms",
        )
    }


def test_gate_passes_only_as_offline_preflight() -> None:
    gate = evaluate_prototype_transport_gate(
        input_contract_valid=True,
        quota_exact=True,
        integrality_max_error=1e-9,
        changed_fraction_pct=12.0,
        comparisons=_comparisons(),
        best_baseline_name="fixed_clip",
        full_macro_delta_pp=0.3,
        car_delta_pp=0.1,
        truck_delta_pp=0.2,
        car_truck_mean_delta_pp=0.15,
        other_ten_mean_delta_pp=0.2,
    )
    assert gate["decision"] == "PASS_PROTOTYPE_TRANSPORT_PREFLIGHT"
    assert gate["training_authorized"] is False
    assert gate["proxy_authorized"] is False
    assert gate["gpu_authorized"] is False

    failed = evaluate_prototype_transport_gate(
        input_contract_valid=True,
        quota_exact=True,
        integrality_max_error=1e-9,
        changed_fraction_pct=12.0,
        comparisons=_comparisons(gain=0.8, low=-0.1),
        best_baseline_name="fixed_clip",
        full_macro_delta_pp=0.1,
        car_delta_pp=-0.6,
        truck_delta_pp=0.2,
        car_truck_mean_delta_pp=-0.2,
        other_ten_mean_delta_pp=0.2,
    )
    assert failed["decision"] == "REJECT"
    assert not failed["checks"]["gain_vs_best_baseline_at_least_1pp"]
    assert not failed["checks"]["gain_vs_best_baseline_ci_lower_positive"]
    assert not failed["checks"]["car_regression_at_most_0_5pp"]


def test_cloud_entrypoint_is_cpu_only_and_locks_before_oracle() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_prototype_transport_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_prototype_transport.py"
    ).read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "source_C.pt" in runner
    assert "source_F.pt" not in runner
    assert "source_B.pt" not in runner
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert (
        '"training_authorized": False'
        in (REPO_ROOT / "src/utils/prototype_transport_audit.py").read_text()
    )
    assert audit.index("lock_path.write_text") < audit.rindex(
        "labels = _parse_labels_after_lock"
    )
    assert audit.index("lock_path.write_text") < audit.rindex(
        'embedded_labels = np.asarray(snapshot["target_label"]'
    )
