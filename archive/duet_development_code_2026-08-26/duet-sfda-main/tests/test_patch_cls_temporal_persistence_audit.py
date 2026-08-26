from pathlib import Path

import numpy as np

from src.utils.patch_cls_temporal_persistence_audit import (
    apply_frozen_patch_memory,
    evaluate_patch_temporal_persistence_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_patch_memory_changes_only_selected_differences() -> None:
    current = np.array([0, 1, 2, 3, 4])
    position = np.array([1, 3, 4])
    selected = np.array([True, True, False])
    memory = np.array([1, 2, 0])
    result = apply_frozen_patch_memory(current, position, selected, memory)
    assert result["prediction"].tolist() == [0, 1, 2, 2, 4]
    assert result["effective_correction"].tolist() == [False, True, False]


def _comparisons(gain: float = 4.0, low: float = 1.0) -> dict:
    names = (
        "cycle2_task", "cycle2_clip", "cycle2_confidence",
        "cycle2_arithmetic", "cycle2_rms", "cycle2_mix",
    )
    return {
        name: {
            "candidate_accuracy_pct": 80.0,
            "baseline_accuracy_pct": 80.0 - gain - offset * 0.1,
            "gain_pp": gain + offset * 0.1,
            "paired_bootstrap_95_ci_pp": [low, 6.0],
        }
        for offset, name in enumerate(names)
    }


def test_temporal_gate_pass_never_authorizes_training() -> None:
    gate = evaluate_patch_temporal_persistence_gate(
        input_contract_valid=True,
        exploratory_selector_pass_preserved=True,
        heldout_selector_pass_preserved=True,
        selected_coverage_pct=3.0,
        effective_corrections=50,
        selected_comparisons=_comparisons(),
        effective_task_comparison={"paired_bootstrap_95_ci_pp": [0.5, 5.0]},
        full_proxy_task_macro_gain_pp=0.3,
        car_delta_pp=0.1,
        truck_delta_pp=-0.1,
        car_truck_mean_delta_pp=0.0,
        other_ten_mean_delta_pp=0.2,
        max_class_mass_shift_pp=0.8,
    )
    assert gate["decision"] == "PASS_EXPLORATORY_PATCH_TEMPORAL_PERSISTENCE"
    assert gate["pure_duet_cycle2_snapshot_confirmation_authorized"] is True
    assert gate["proxy_training_authorized"] is False
    assert gate["full_training_authorized"] is False


def test_temporal_gate_rejects_best_control_or_class_exchange() -> None:
    comparisons = _comparisons()
    comparisons["cycle2_task"]["baseline_accuracy_pct"] = 79.5
    comparisons["cycle2_task"]["gain_pp"] = 0.5
    comparisons["cycle2_task"]["paired_bootstrap_95_ci_pp"] = [-0.2, 1.2]
    gate = evaluate_patch_temporal_persistence_gate(
        input_contract_valid=True,
        exploratory_selector_pass_preserved=True,
        heldout_selector_pass_preserved=True,
        selected_coverage_pct=3.0,
        effective_corrections=50,
        selected_comparisons=comparisons,
        effective_task_comparison={"paired_bootstrap_95_ci_pp": [-0.1, 2.0]},
        full_proxy_task_macro_gain_pp=0.3,
        car_delta_pp=0.1,
        truck_delta_pp=-0.8,
        car_truck_mean_delta_pp=-0.35,
        other_ten_mean_delta_pp=0.2,
        max_class_mass_shift_pp=0.8,
    )
    assert gate["decision"] == "REJECT"
    assert not gate["checks"]["memory_gain_vs_best_selected_baseline_at_least_1pp"]
    assert not gate["checks"]["truck_regression_at_most_0_5pp"]


def test_entrypoint_is_cpu_only_and_locks_before_oracle() -> None:
    runner = (REPO_ROOT / "tools/run_visda_patch_cls_temporal_persistence_audit.sh").read_text()
    audit = (REPO_ROOT / "tools/audit_visda_patch_cls_temporal_persistence.py").read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "import torch" not in audit
    assert "import clip" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert audit.index("lock_path.write_text") < audit.index(
        "risk_summary = json.loads"
    )
    assert audit.index("lock_path.write_text") < audit.index(
        "labels_by_dataset_index = _parse_labels_after_lock"
    )
