import os
from pathlib import Path

import numpy as np
import pytest

from src.utils.swap_conflict_selection import select_swap_labels
from src.utils.swap_intervention_audit import (
    SwapInterventionAuditor,
    build_cross_cycle_transition,
    build_cycle_confusion_matrices,
    build_swap_audit_payload,
    correction_stats,
)


CLASS_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]


def _minimal_diagnostics(n=6):
    """Diagnostics dict for a crafted 6-sample batch."""
    task_top1 = np.asarray([3, 3, 3, 2, 0, 1], dtype=np.int64)
    task_top2 = np.asarray([11, 11, 11, 11, 1, 0], dtype=np.int64)
    clip_top1 = np.asarray([11, 11, 11, 11, 1, 0], dtype=np.int64)
    clip_top2 = np.asarray([3, 3, 3, 2, 0, 1], dtype=np.int64)
    return {
        "is_conflict": np.asarray([True] * 4 + [False] * 2),
        "is_swap_candidate": np.asarray([True] * 4 + [False] * 2),
        "task_top1": task_top1,
        "task_top1_prob": np.asarray([0.7] * n),
        "task_top2": task_top2,
        "task_top2_prob": np.asarray([0.2] * n),
        "clip_top1": clip_top1,
        "clip_top1_prob": np.asarray([0.7] * n),
        "clip_top2": clip_top2,
        "clip_top2_prob": np.asarray([0.2] * n),
        "candidate_A": task_top1,
        "candidate_B": clip_top1,
        "task_evidence": np.asarray([3.5] * n),
        "clip_evidence": np.asarray([3.5] * n),
        "log_task_evidence": np.asarray([1.25] * n),
        "log_clip_evidence": np.asarray([1.25] * n),
        "signed_log_gap": np.asarray([0.0] * n),
        "absolute_log_gap": np.asarray([0.0] * n),
        "passed_gate": np.asarray([True] * 4 + [False] * 2),
        "passed_direction_filter": np.asarray([True] * 4 + [False] * 2),
        "choose_task": np.asarray([True, True, True, False, False, False]),
        "choose_clip": np.asarray([False, False, False, True, True, True]),
        "swap_selected": np.asarray([True] * 4 + [False] * 2),
        "abstain_reason": np.asarray(
            ["selected_task"] * 3 + ["selected_clip"] + ["not_conflict"] * 2,
            dtype=object,
        ),
    }


def _payload(
    *,
    cycle=2,
    base_mix=None,
    final=None,
    base_mask=None,
    real=None,
    selected=None,
):
    n = 6
    if real is None:
        real = np.asarray([3, 3, 3, 2, 0, 1], dtype=np.int64)
    if base_mix is None:
        base_mix = np.asarray([2, 3, 3, 2, 0, 1], dtype=np.int64)
    if final is None:
        final = np.asarray([3, 2, 3, 1, 0, 1], dtype=np.int64)
    if base_mask is None:
        base_mask = np.asarray([False, True, True, False, False, True])
    if selected is None:
        selected = np.asarray([True, True, True, True, False, False])
    task = np.full((n, 12), 1.0 / 12.0)
    clip = np.full((n, 12), 1.0 / 12.0)
    rng = np.random.default_rng(0)
    return build_swap_audit_payload(
        cycle=cycle,
        task_prob=task,
        clip_prob=clip,
        task_feat=rng.standard_normal((n, 512)).astype(np.float32),
        strong_feat=rng.standard_normal((n, 512)).astype(np.float32),
        base_mix_label=base_mix,
        final_mem_label=final,
        base_label_mask=base_mask,
        final_label_mask=base_mask | selected,
        prev_label_mask=None,
        current_agreement=np.zeros(n, bool),
        swap_selected=selected,
        swap_diagnostics=_minimal_diagnostics(),
        real_label=real,
        sample_index=np.arange(n),
        image_paths=[f"img{i}.jpg" for i in range(n)],
        class_names=CLASS_NAMES,
        gate_D=2.0,
        min_direction_accuracy=0.8,
    )


def test_correction_types_and_stats():
    payload = _payload()
    assert list(payload["correction_type"]) == [
        "W2R",
        "R2W",
        "UNCHANGED",
        "R2W",
        "UNCHANGED",
        "UNCHANGED",
    ]
    stats = correction_stats(payload, np.ones(6, bool))
    assert stats["W2R"] == 1
    assert stats["R2W"] == 2
    assert stats["R2R"] == 1
    assert stats["W2W"] == 0
    assert stats["net_correction"] == -1
    assert stats["sample_count"] == 4
    assert stats["base_correct_count"] == 3
    assert stats["final_correct_count"] == 2
    assert stats["accuracy_delta"] == -25.0


def test_newly_admitted_and_label_changed():
    payload = _payload()
    assert list(payload["already_admitted"]) == [False, True, True, False, False, False]
    assert list(payload["newly_admitted"]) == [True, False, False, True, False, False]
    assert list(payload["label_changed"]) == [True, True, False, True, False, False]
    assert np.all(payload["newly_admitted"] <= payload["swap_selected"])
    assert np.all(payload["label_changed"] <= payload["swap_selected"])
    assert np.array_equal(
        payload["final_label_mask"],
        payload["base_label_mask"] | payload["swap_selected"],
    )


def test_confusion_matrix_orientation():
    payload = _payload()
    matrices = build_cycle_confusion_matrices(payload)
    # CM04: selected rows GT -> cols final_mem.
    cm04 = matrices["cm04_swap_selected_final"]
    assert cm04[3, 3] == 2  # samples 0 and 2
    assert cm04[3, 2] == 1  # sample 1
    assert cm04[2, 1] == 1  # sample 3
    assert cm04.shape == (12, 12)
    # CM06: label-changed rows base -> cols final.
    cm06 = matrices["cm06_label_changed_direction"]
    assert cm06[2, 3] == 1
    assert cm06[3, 2] == 1
    assert cm06[2, 1] == 1
    # CM07: W2R rows base -> cols real.
    cm07 = matrices["cm07_w2r_correction"]
    assert cm07[2, 3] == 1
    # CM08: R2W rows real -> cols final.
    cm08 = matrices["cm08_r2w_damage"]
    assert cm08[3, 2] == 1
    assert cm08[2, 1] == 1


def test_auditor_skips_cycle1_and_writes_cycle2_3(tmp_path):
    auditor = SwapInterventionAuditor(tmp_path, CLASS_NAMES)
    payload = _payload()
    auditor.record_cycle(0, payload)  # cycle 1 must not be recorded
    assert not (tmp_path / "swap_intervention_audit" / "cycle01").exists()
    auditor.record_cycle(1, payload)
    auditor.record_cycle(2, _payload(cycle=3))
    root = tmp_path / "swap_intervention_audit"
    assert (root / "cycle02" / "cycle02_all_samples.csv").is_file()
    assert (root / "cycle03" / "cycle03_all_samples.csv").is_file()
    assert (root / "cycle02" / "summary.json").is_file()
    assert (root / "cycle02" / "cm04_swap_selected_final_raw.png").is_file()
    assert (root / "cycle02" / "cm04_swap_selected_final_row_normalized.csv").is_file()
    assert (root / "cycle02" / "cycle02_all_samples.npz").is_file()
    assert (root / "cycle03" / "cycle03_all_samples.npz").is_file()
    assert (root / "cycle02_cycle03_transition.csv").is_file()
    assert (root / "cycle_transition_summary.json").is_file()
    assert (root / "cycle02" / "pair_summary_sorted_best.csv").is_file()
    assert (root / "cycle02" / "pair_summary_sorted_worst.csv").is_file()
    assert (root / "cycle02" / "cycle02_class_summary.csv").is_file()


def test_npz_contains_full_probabilities_and_features(tmp_path):
    auditor = SwapInterventionAuditor(tmp_path, CLASS_NAMES)
    payload = _payload(cycle=2)
    auditor.record_cycle(1, payload)
    npz_path = (
        tmp_path
        / "swap_intervention_audit"
        / "cycle02"
        / "cycle02_all_samples.npz"
    )
    with np.load(npz_path) as data:
        assert data["task_prob"].shape == (6, 12)
        assert data["clip_prob"].shape == (6, 12)
        assert data["task_feat"].shape == (6, 512)
        assert data["strong_feat"].shape == (6, 512)
        assert set(data.files) >= {
            "task_prob",
            "clip_prob",
            "task_feat",
            "strong_feat",
            "signed_log_gap",
            "correction_type",
        }


def test_cross_cycle_transition_alignment():
    p2 = _payload()
    p3 = _payload()
    transition = build_cross_cycle_transition(p2, p3)
    assert len(transition["rows"]) == 6
    assert transition["rows"][0][0] == 0  # sample_index aligned
    for key in (
        "cycle2_swap_to_cycle3_agreement",
        "cycle2_selected_to_cycle3_selected_same_side",
        "cycle2_w2r_to_cycle3_remains_correct",
    ):
        assert key in transition["summary"]


def test_return_diagnostics_does_not_change_selection():
    task = np.asarray(
        [
            [0.70, 0.20, 0.05, 0.05],
            [0.05, 0.10, 0.70, 0.15],
            [0.60, 0.30, 0.05, 0.05],
        ]
    )
    clip = np.asarray(
        [
            [0.20, 0.70, 0.05, 0.05],
            [0.05, 0.10, 0.20, 0.65],
            [0.10, 0.55, 0.30, 0.05],
        ]
    )
    labels_plain, selected_plain = select_swap_labels(
        task, clip, cycle=2, gate_D=0.0
    )
    labels_diag, selected_diag, diagnostics = select_swap_labels(
        task,
        clip,
        cycle=2,
        gate_D=0.0,
        return_diagnostics=True,
    )
    assert np.array_equal(labels_plain, labels_diag)
    assert np.array_equal(selected_plain, selected_diag)
    for key in (
        "is_conflict",
        "is_swap_candidate",
        "abstain_reason",
        "signed_log_gap",
        "passed_gate",
    ):
        assert key in diagnostics
    assert set(diagnostics["abstain_reason"]) <= {
        "not_conflict",
        "conflict_non_swap",
        "inactive_cycle",
        "direction_filter_failed",
        "gate_failed",
        "selected_task",
        "selected_clip",
    }


def test_audit_supports_torch_tensors():
    torch = pytest.importorskip("torch")
    payload = _payload(
        base_mix=np.asarray([2, 3, 3, 2, 0, 1]),
        final=np.asarray([3, 2, 3, 1, 0, 1]),
    )
    # Rebuild from torch tensors to verify .detach().cpu() path.
    rebuilt = build_swap_audit_payload(
        cycle=2,
        task_prob=torch.zeros(6, 12).fill_(1.0 / 12.0),
        clip_prob=torch.zeros(6, 12).fill_(1.0 / 12.0),
        base_mix_label=torch.from_numpy(np.asarray([2, 3, 3, 2, 0, 1])),
        final_mem_label=torch.from_numpy(np.asarray([3, 2, 3, 1, 0, 1])),
        base_label_mask=torch.from_numpy(payload["base_label_mask"]),
        final_label_mask=torch.from_numpy(payload["final_label_mask"]),
        prev_label_mask=None,
        current_agreement=torch.zeros(6, dtype=torch.bool),
        swap_selected=torch.from_numpy(payload["swap_selected"]),
        swap_diagnostics=_minimal_diagnostics(),
        real_label=torch.from_numpy(np.asarray([3, 3, 3, 2, 0, 1])),
        sample_index=torch.arange(6),
        image_paths=[f"img{i}.jpg" for i in range(6)],
        class_names=CLASS_NAMES,
        gate_D=2.0,
        min_direction_accuracy=0.8,
    )
    assert list(rebuilt["correction_type"]) == list(payload["correction_type"])
