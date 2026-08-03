from pathlib import Path

import numpy as np

from src.utils.patch_cls_risk_control_audit import (
    evaluate_patch_cls_holdout_gate,
    evaluate_patch_cls_risk_control_gate,
    select_upper_median_mass_capped_rescues,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_upper_median_and_mass_cap_are_label_free_and_deterministic() -> None:
    task = np.array([1, 1, 1, 2, 2, 2])
    clip = np.array([0, 0, 0, 0, 0, 0])
    margin = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    stable = np.ones(6, dtype=bool)
    result = select_upper_median_mass_capped_rescues(
        task,
        clip,
        margin,
        stable,
        full_sample_count=200,
        max_class_mass_shift_fraction=0.01,
        class_count=3,
    )

    assert np.isclose(result["threshold"], 0.65)
    assert result["upper_median"].tolist() == [True, True, True, False, False, False]
    # The 1% cap is two samples, so the third same-direction route is rejected.
    assert result["selected"].tolist() == [True, True, False, False, False, False]
    assert result["rejected_by_mass_cap"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert result["class_count_shift"].tolist() == [-2, 2, 0]
    assert result["prediction"].tolist() == [1, 1, 0, 0, 0, 0]


def test_class_mass_accounting_accepts_balancing_reverse_flow() -> None:
    task = np.array([1, 0, 1, 0])
    clip = np.array([0, 1, 0, 1])
    margin = np.array([0.9, 0.8, 0.7, 0.6])
    stable = np.ones(4, dtype=bool)
    result = select_upper_median_mass_capped_rescues(
        task,
        clip,
        margin,
        stable,
        full_sample_count=100,
        max_class_mass_shift_fraction=0.01,
        class_count=2,
    )
    assert result["upper_median"].tolist() == [True, True, False, False]
    assert result["selected"].tolist() == [True, True, False, False]
    assert result["class_count_shift"].tolist() == [0, 0]


def _comparisons(gain: float = 1.3, low: float = 0.4) -> dict:
    return {
        name: {
            "candidate_accuracy_pct": 73.6,
            "baseline_accuracy_pct": 72.3 - offset * 0.1,
            "gain_pp": gain + offset * 0.1,
            "paired_bootstrap_95_ci_pp": [low, 2.0],
        }
        for offset, name in enumerate(
            ("fixed_task", "fixed_clip", "confidence_choice", "arithmetic", "rms")
        )
    }


def test_gate_pass_is_exploratory_and_never_authorizes_training() -> None:
    gate = evaluate_patch_cls_risk_control_gate(
        input_contract_valid=True,
        source_reject_preserved=True,
        selected_coverage_pct=3.0,
        paired_adjudication_precision_pct=80.0,
        comparisons=_comparisons(),
        full_proxy_macro_gain_pp=0.5,
        car_delta_pp=0.2,
        truck_delta_pp=-0.2,
        car_truck_mean_delta_pp=0.0,
        other_ten_mean_delta_pp=0.3,
        max_class_mass_shift_pp=0.99,
    )
    assert gate["decision"] == "PASS_EXPLORATORY_PATCH_CLS_RISK_CONTROL"
    assert gate["heldout_full_audit_authorized"] is True
    assert gate["parameter_audit_authorized"] is False
    assert gate["proxy_authorized"] is False
    assert gate["training_authorized"] is False


def test_gate_rejects_class_exchange_or_nonpositive_ci() -> None:
    comparisons = _comparisons()
    comparisons["fixed_task"]["paired_bootstrap_95_ci_pp"] = [-0.1, 2.0]
    gate = evaluate_patch_cls_risk_control_gate(
        input_contract_valid=True,
        source_reject_preserved=True,
        selected_coverage_pct=3.0,
        paired_adjudication_precision_pct=80.0,
        comparisons=comparisons,
        full_proxy_macro_gain_pp=0.5,
        car_delta_pp=0.2,
        truck_delta_pp=-0.8,
        car_truck_mean_delta_pp=-0.3,
        other_ten_mean_delta_pp=0.3,
        max_class_mass_shift_pp=0.99,
    )
    assert gate["decision"] == "REJECT"
    assert not gate["checks"]["truck_regression_at_most_0_5pp"]


def test_entrypoint_is_cpu_only_and_locks_before_oracle() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_patch_cls_risk_control_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_patch_cls_risk_control.py"
    ).read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "import torch" not in audit
    assert "import clip" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert '"training_authorized": False' in audit
    assert audit.index("lock_path.write_text") < audit.rindex(
        "labels = _read_oracle_after_lock"
    )
    assert audit.index("lock_path.write_text") < audit.rindex(
        "source_summary = json.loads"
    )


def _heldout_comparisons(gain: float = 1.2, low: float = 0.3) -> dict:
    result = {}
    for offset, name in enumerate(("fixed_task", "fixed_clip", "confidence_choice")):
        current_gain = gain + 0.1 * offset
        result[name] = {
            "candidate_accuracy_pct": 73.5,
            "baseline_accuracy_pct": 73.5 - current_gain,
            "gain_pp": current_gain,
            "paired_bootstrap_95_ci_pp": [low, 2.0],
        }
    return result


def test_heldout_gate_pass_only_authorizes_parameter_audit() -> None:
    gate = evaluate_patch_cls_holdout_gate(
        input_contract_valid=True,
        exploratory_pass_preserved=True,
        heldout_is_disjoint=True,
        selected_coverage_pct=3.0,
        paired_adjudication_precision_pct=75.0,
        comparisons=_heldout_comparisons(),
        heldout_macro_gain_pp=0.4,
        car_delta_pp=0.1,
        truck_delta_pp=-0.2,
        car_truck_mean_delta_pp=-0.05,
        other_ten_mean_delta_pp=0.2,
        max_class_mass_shift_pp=0.99,
    )
    assert gate["decision"] == "REJECT"
    assert not gate["checks"]["car_truck_mean_nonnegative"]

    gate = evaluate_patch_cls_holdout_gate(
        input_contract_valid=True,
        exploratory_pass_preserved=True,
        heldout_is_disjoint=True,
        selected_coverage_pct=3.0,
        paired_adjudication_precision_pct=75.0,
        comparisons=_heldout_comparisons(),
        heldout_macro_gain_pp=0.4,
        car_delta_pp=0.2,
        truck_delta_pp=-0.2,
        car_truck_mean_delta_pp=0.0,
        other_ten_mean_delta_pp=0.2,
        max_class_mass_shift_pp=0.99,
    )
    assert gate["decision"] == "PASS_HELDOUT_PATCH_CLS_RISK_CONTROL"
    assert gate["parameter_audit_authorized"] is True
    assert gate["proxy_authorized"] is False
    assert gate["full_training_authorized"] is False


def test_heldout_gate_rejects_proxy_overlap_or_weak_clip_gain() -> None:
    comparisons = _heldout_comparisons(gain=0.8, low=-0.1)
    gate = evaluate_patch_cls_holdout_gate(
        input_contract_valid=True,
        exploratory_pass_preserved=True,
        heldout_is_disjoint=False,
        selected_coverage_pct=3.0,
        paired_adjudication_precision_pct=75.0,
        comparisons=comparisons,
        heldout_macro_gain_pp=0.4,
        car_delta_pp=0.2,
        truck_delta_pp=-0.2,
        car_truck_mean_delta_pp=0.0,
        other_ten_mean_delta_pp=0.2,
        max_class_mass_shift_pp=0.99,
    )
    assert gate["decision"] == "REJECT"
    assert not gate["checks"]["heldout_paths_disjoint_from_proxy25"]
    assert not gate["checks"]["gain_vs_fixed_clip_at_least_1pp"]
    assert not gate["checks"]["gain_vs_fixed_clip_ci_lower_positive"]


def test_heldout_entrypoint_excludes_proxy_and_locks_before_labels() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_patch_cls_risk_control_holdout.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_patch_cls_risk_control_holdout.py"
    ).read_text()
    assert "Excludes all 13,847 proxy25 design paths" in runner
    assert "source_F.pt" not in audit
    assert "source_B.pt" not in audit
    assert "source_C.pt" not in audit
    assert ".backward(" not in audit
    assert "optimizer.step" not in audit
    assert audit.index("lock_path.write_text") < audit.rindex(
        "labels = _parse_labels_after_lock"
    )
    assert audit.index("lock_path.write_text") < audit.rindex(
        "exploratory_summary = json.loads"
    )
