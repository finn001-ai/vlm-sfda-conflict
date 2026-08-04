import csv
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.utils.swap_conflict_selection import (
    CYCLE0_DIRECTION_ACCURACY,
    DEFAULT_GATE_D,
    EPS,
    decide_swap_evidence,
    select_swap_labels,
    swap_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/Users/stranger/Downloads/task_TV_seed_2020")
ARCHIVED_CURVE = (
    REPO_ROOT.parent
    / "archive"
    / "sfda_conflict_visda_topk_swap_analysis_2026-08-04"
    / "data"
    / "selection_curve.csv"
)


# ---------------------------------------------------------------- unit tests


def _swap_task_clip():
    """Two pure-swap rows plus one non-swap conflict and one agreement."""
    task = np.asarray(
        [
            [0.70, 0.20, 0.05, 0.05],  # swap: A=0, B=1
            [0.05, 0.10, 0.70, 0.15],  # swap: A=2, B=3
            [0.60, 0.30, 0.05, 0.05],  # non-swap conflict (clip top2 != task top1)
            [0.40, 0.35, 0.15, 0.10],  # agreement (task top1 == clip top1 == 0)
        ],
        dtype=np.float64,
    )
    clip = np.asarray(
        [
            [0.20, 0.70, 0.05, 0.05],  # swap: clip top1=1, top2=0
            [0.05, 0.10, 0.20, 0.65],  # swap: clip top1=3, top2=2
            [0.10, 0.55, 0.30, 0.05],  # non-swap conflict
            [0.50, 0.30, 0.10, 0.10],  # agreement
        ],
        dtype=np.float64,
    )
    return task, clip


def test_swap_detection_marks_only_bidirectional_conflicts():
    task, clip = _swap_task_clip()
    evidence = swap_evidence(task, clip)
    np.testing.assert_array_equal(evidence["swap_mask"], [True, True, False, False])
    np.testing.assert_array_equal(evidence["A"], [0, 2, 0, 0])
    np.testing.assert_array_equal(evidence["B"], [1, 3, 1, 0])
    np.testing.assert_allclose(evidence["pA"][:2], [0.70, 0.70])
    np.testing.assert_allclose(evidence["pB"][:2], [0.20, 0.15])
    np.testing.assert_allclose(evidence["qA"][:2], [0.20, 0.20])
    np.testing.assert_allclose(evidence["qB"][:2], [0.70, 0.65])


def test_cycle0_always_picks_clip_top1_without_gate():
    task, clip = _swap_task_clip()
    for gate in (0.0, 4.0, 100.0):
        labels, selected = select_swap_labels(
            task, clip, cycle=0, gate_D=gate
        )
        assert list(selected) == [True, True, False, False]
        assert list(labels) == [1, 3, -1, -1]


def test_gate_chooses_a_when_eA_dominates():
    prefer_a, decided = decide_swap_evidence(
        pA=[0.8], pB=[0.1], qA=[0.6], qB=[0.25], cycle=1, gate_D=2.0
    )
    assert decided.tolist() == [True]
    assert prefer_a.tolist() == [True]


def test_gate_chooses_b_when_eB_dominates():
    prefer_a, decided = decide_swap_evidence(
        pA=[0.1], pB=[0.8], qA=[0.25], qB=[0.6], cycle=1, gate_D=2.0
    )
    assert decided.tolist() == [True]
    assert prefer_a.tolist() == [False]


def test_gate_abstains_when_evidence_is_close():
    prefer_a, decided = decide_swap_evidence(
        pA=[0.5], pB=[0.4], qA=[0.5], qB=[0.4], cycle=1, gate_D=2.0
    )
    assert decided.tolist() == [False]
    # The same sample is decided (to A) when the gate is off.
    prefer_a0, decided0 = decide_swap_evidence(
        pA=[0.5], pB=[0.4], qA=[0.5], qB=[0.4], cycle=1, gate_D=0.0
    )
    assert decided0.tolist() == [True]
    assert prefer_a0.tolist() == [True]


def test_tie_falls_back_to_b_matching_archive():
    prefer_a, decided = decide_swap_evidence(
        pA=[0.5], pB=[0.5], qA=[0.5], qB=[0.5], cycle=1, gate_D=0.0
    )
    assert decided.tolist() == [True]
    assert prefer_a.tolist() == [False]


def test_zero_probability_boundary_uses_eps():
    # pB == 0 and qA == 0 must not produce inf/nan; eps clamps the log.
    prefer_a, decided = decide_swap_evidence(
        pA=[1.0], pB=[0.0], qA=[0.0], qB=[0.1], cycle=1, gate_D=2.0
    )
    assert np.isfinite(prefer_a).all() and np.isfinite(decided).all()
    assert decided.tolist() == [True]
    assert prefer_a.tolist() == [True]
    # log(10) ~ 2.30 < 4.0, so the default gate abstains instead of crashing.
    _, decided_gate = decide_swap_evidence(
        pA=[1.0], pB=[0.0], qA=[0.0], qB=[0.1], cycle=1, gate_D=4.0
    )
    assert decided_gate.tolist() == [False]
    # Both evidence products collapse to eps*eps -> tie -> B, no crash.
    prefer_a2, decided2 = decide_swap_evidence(
        pA=[0.0], pB=[0.0], qA=[0.0], qB=[0.0], cycle=1, gate_D=0.0
    )
    assert decided2.tolist() == [True]
    assert prefer_a2.tolist() == [False]


def test_eps_is_positive_and_matches_archive_value():
    assert EPS == 1e-9
    assert DEFAULT_GATE_D == 4.0


def test_select_swap_labels_accepts_torch_tensors():
    task, clip = _swap_task_clip()
    labels, selected = select_swap_labels(
        torch.from_numpy(task),
        torch.from_numpy(clip),
        cycle=1,
        gate_D=0.0,
    )
    # Row 0: eA=0.70*0.20=0.14 > eB=0.20*0.70=0.14 -> tie -> B(1).
    # Row 1: eA=0.70*0.20=0.14 > eB=0.15*0.65=0.0975 -> A(2).
    assert list(selected) == [True, True, False, False]
    assert list(labels) == [1, 2, -1, -1]


def test_invalid_inputs_are_rejected():
    task, clip = _swap_task_clip()
    with pytest.raises(ValueError):
        select_swap_labels(task, clip, cycle=-1)
    with pytest.raises(ValueError):
        select_swap_labels(task, clip, cycle=1, gate_D=-1.0)
    with pytest.raises(ValueError):
        swap_evidence(task, np.ones((4, 3)))
    with pytest.raises(ValueError):
        decide_swap_evidence([0.5], [0.5], [0.5], [0.5, 0.5], cycle=1)
    with pytest.raises(ValueError):
        select_swap_labels(task, clip, cycle=1, min_direction_accuracy=1.5)


def _swap_orientation(task_a, task_b, clip_top1):
    """Build a 2-class swap where task top1=task_a and clip top1=clip_top1."""
    task = np.zeros((1, 12), dtype=np.float64)
    clip = np.zeros((1, 12), dtype=np.float64)
    task[0, task_a] = 0.7
    task[0, task_b] = 0.2
    task[0, 0] += 1.0 - task[0].sum()
    clip[0, clip_top1] = 0.7
    clip[0, task_a] = 0.2
    clip[0, 0] += 1.0 - clip[0].sum()
    return task, clip


def test_direction_accuracy_table_matches_archived_cycle0():
    data_dir = _regression_data_dir()
    if not data_dir.is_dir():
        pytest.skip("swap regression data not found")
    import collections

    rows = [
        r
        for r in csv.DictReader(
            (data_dir / "cycle_000" / "conflict_samples.csv").open()
        )
        if r["bidirectional_cross_support"] == "True"
    ]
    agg = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        a, b, gt = (
            int(row["task_top1_id"]),
            int(row["clip_top1_id"]),
            int(row["gt_label_probe"]),
        )
        agg[(a, b)][0] += 1
        agg[(a, b)][1] += b == gt
    assert len(agg) == len(CYCLE0_DIRECTION_ACCURACY) == 65
    for (a, b), (count, correct) in agg.items():
        expected = correct / count
        assert abs(CYCLE0_DIRECTION_ACCURACY[(a, b)] - expected) <= 1e-3


def test_direction_gate_abstains_unreliable_orientations():
    # car(3)->truck(11): locked CLIP accuracy 69.2% < 0.8 -> abstain.
    task, clip = _swap_orientation(3, 11, 11)
    labels, selected = select_swap_labels(
        task, clip, cycle=0, min_direction_accuracy=0.8
    )
    assert selected.tolist() == [False]
    assert labels.tolist() == [-1]
    # motorcycle(6)->bicycle(1): locked 90.2% >= 0.8 -> kept.
    task, clip = _swap_orientation(6, 1, 1)
    labels, selected = select_swap_labels(
        task, clip, cycle=0, min_direction_accuracy=0.8
    )
    assert selected.tolist() == [True]
    assert labels.tolist() == [1]


def test_direction_gate_applies_to_later_cycles_too():
    task, clip = _swap_orientation(3, 11, 11)
    labels, selected = select_swap_labels(
        task, clip, cycle=3, gate_D=0.0, min_direction_accuracy=0.8
    )
    assert selected.tolist() == [False]
    # Without the direction gate the same sample is decided.
    _, selected = select_swap_labels(
        task, clip, cycle=3, gate_D=0.0, min_direction_accuracy=0.0
    )
    assert selected.tolist() == [True]


def test_last_active_cycle_stops_new_labels():
    task, clip = _swap_task_clip()
    labels, selected = select_swap_labels(
        task, clip, cycle=6, gate_D=0.0, last_active_cycle=6
    )
    assert selected.tolist() == [False, False, False, False]
    assert labels.tolist() == [-1, -1, -1, -1]
    # cycle 5 (1-based cycle 6) is still active.
    _, selected = select_swap_labels(
        task, clip, cycle=5, gate_D=0.0, last_active_cycle=6
    )
    assert selected.tolist() == [True, True, False, False]
    with pytest.raises(ValueError):
        select_swap_labels(task, clip, cycle=1, last_active_cycle=0)


# ------------------------------------------------------------ regression test


def _load_swap_rows(data_dir: Path):
    per_cycle = []
    for cycle in range(8):
        path = data_dir / f"cycle_{cycle:03d}" / "conflict_samples.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = [
            row
            for row in csv.DictReader(path.open())
            if row["bidirectional_cross_support"] == "True"
        ]
        per_cycle.append(rows)
    return per_cycle


def _regression_data_dir():
    override = os.environ.get("SWAP_CONFLICT_DATA_DIR")
    data_dir = Path(override) if override else DEFAULT_DATA_DIR
    return data_dir


def test_regression_matches_archived_selection_curve():
    data_dir = _regression_data_dir()
    if not data_dir.is_dir():
        pytest.skip(
            "swap regression data not found; set SWAP_CONFLICT_DATA_DIR to "
            "the task_TV_seed_2020 directory"
        )
    thresholds = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    curve = {t: [0, 0] for t in thresholds}
    cycle0 = None
    for cycle, rows in enumerate(_load_swap_rows(data_dir)):
        pA = np.array([float(r["task_top1_prob"]) for r in rows])
        pB = np.array([float(r["task_top2_prob"]) for r in rows])
        qB = np.array([float(r["clip_top1_score"]) for r in rows])
        qA = np.array([float(r["clip_top2_score"]) for r in rows])
        gt = np.array([int(r["gt_label_probe"]) for r in rows])
        a_ids = np.array([int(r["task_top1_id"]) for r in rows])
        b_ids = np.array([int(r["clip_top1_id"]) for r in rows])
        prefer_a, decided = decide_swap_evidence(
            pA, pB, qA, qB, cycle=cycle, gate_D=0.0
        )
        real = np.where(prefer_a, a_ids, b_ids)
        if cycle == 0:
            cycle0 = int((decided & (real == gt)).sum()), int(decided.sum())
        for threshold in thresholds:
            _, sel = decide_swap_evidence(
                pA, pB, qA, qB, cycle=cycle, gate_D=threshold
            )
            curve[threshold][0] += int(sel.sum())
            curve[threshold][1] += int((sel & (real == gt)).sum())

    total_swap = sum(int(len(rows)) for rows in _load_swap_rows(data_dir))
    assert total_swap == 19398

    # Cycle 0 special case: always CLIP top1 with 77.0% precision.
    assert cycle0[1] == 1949
    assert abs(100.0 * cycle0[0] / cycle0[1] - 77.0) <= 0.2

    archived = list(csv.DictReader(ARCHIVED_CURVE.open()))
    assert len(archived) == len(thresholds)
    for threshold, archived_row in zip(thresholds, archived):
        decisions, correct = curve[threshold]
        assert decisions == int(archived_row["decisions"])
        assert correct == int(archived_row["correct"])
        assert abs(100.0 * correct / decisions - float(archived_row["precision"])) <= 0.2
        assert abs(100.0 * decisions / total_swap - float(archived_row["coverage_pct"])) <= 0.2

    # Key operating points called out by the archived analysis.
    assert curve[2.0] == [9196, 6013]
    assert abs(100.0 * curve[2.0][1] / curve[2.0][0] - 65.4) <= 0.2
    assert abs(100.0 * curve[2.0][0] / total_swap - 47.4) <= 0.2
    assert curve[0.0][1] == 11279
    assert abs(100.0 * curve[0.0][1] / curve[0.0][0] - 58.1) <= 0.2


def test_regression_per_cycle_swap_counts_match_archive():
    data_dir = _regression_data_dir()
    if not data_dir.is_dir():
        pytest.skip("swap regression data not found")
    expected = [1949, 3256, 2849, 2537, 2423, 2266, 2145, 1973]
    actual = [len(rows) for rows in _load_swap_rows(data_dir)]
    assert actual == expected


def test_regression_direction_gate_variant_matches_archive_evidence():
    """D=2.0 + direction gate 0.8: more net-correct labels than the D=4.0
    full-coverage run while cutting wrong labels by ~2/3."""
    data_dir = _regression_data_dir()
    if not data_dir.is_dir():
        pytest.skip("swap regression data not found")
    decisions = correct = 0
    for cycle, rows in enumerate(_load_swap_rows(data_dir)):
        pA = np.array([float(r["task_top1_prob"]) for r in rows])
        pB = np.array([float(r["task_top2_prob"]) for r in rows])
        qB = np.array([float(r["clip_top1_score"]) for r in rows])
        qA = np.array([float(r["clip_top2_score"]) for r in rows])
        gt = np.array([int(r["gt_label_probe"]) for r in rows])
        a_ids = np.array([int(r["task_top1_id"]) for r in rows])
        b_ids = np.array([int(r["clip_top1_id"]) for r in rows])
        prefer_a, decided = decide_swap_evidence(
            pA, pB, qA, qB, cycle=cycle, gate_D=2.0
        )
        direction_ok = np.asarray(
            [
                CYCLE0_DIRECTION_ACCURACY.get((int(a), int(b)), 0.0) >= 0.8
                for a, b in zip(a_ids, b_ids)
            ]
        )
        decided &= direction_ok
        real = np.where(prefer_a, a_ids, b_ids)
        decisions += int(decided.sum())
        correct += int((decided & (real == gt)).sum())
    assert decisions == 4220
    assert correct == 3168
    assert abs(100.0 * correct / decisions - 75.1) <= 0.2
    assert correct - (decisions - correct) == 2116
    # Compared with the archived D>=2.0 curve: 9,196 decisions / 6,013 correct.
    # The direction gate keeps 4,220 high-quality labels and far fewer wrongs.


def test_regression_early_stop_variant_matches_archive_evidence():
    """D=2.0 + direction 0.8 + last_active_cycle=6: stopping before the
    unreliable late cycles trades only ~+85 net labels (decision level) for
    cutting late wrong labels from ~307 to zero."""
    data_dir = _regression_data_dir()
    if not data_dir.is_dir():
        pytest.skip("swap regression data not found")
    decisions = correct = 0
    for cycle, rows in enumerate(_load_swap_rows(data_dir)):
        if cycle + 1 > 6:
            continue
        pA = np.array([float(r["task_top1_prob"]) for r in rows])
        pB = np.array([float(r["task_top2_prob"]) for r in rows])
        qB = np.array([float(r["clip_top1_score"]) for r in rows])
        qA = np.array([float(r["clip_top2_score"]) for r in rows])
        gt = np.array([int(r["gt_label_probe"]) for r in rows])
        a_ids = np.array([int(r["task_top1_id"]) for r in rows])
        b_ids = np.array([int(r["clip_top1_id"]) for r in rows])
        prefer_a, decided = decide_swap_evidence(
            pA, pB, qA, qB, cycle=cycle, gate_D=2.0
        )
        direction_ok = np.asarray(
            [
                CYCLE0_DIRECTION_ACCURACY.get((int(a), int(b)), 0.0) >= 0.8
                for a, b in zip(a_ids, b_ids)
            ]
        )
        decided &= direction_ok
        real = np.where(prefer_a, a_ids, b_ids)
        decisions += int(decided.sum())
        correct += int((decided & (real == gt)).sum())
    assert decisions == 3337
    assert correct == 2592
    assert abs(100.0 * correct / decisions - 77.7) <= 0.2
    assert correct - (decisions - correct) == 1847


# --------------------------------------------------- training-path wiring tests


def test_training_path_keeps_swap_selection_opt_in():
    plmatch = (REPO_ROOT / "src/methods/oh/plmatch.py").read_text()
    wrapper = (
        REPO_ROOT
        / "src/methods/oh/duet_first_cycle_prior_swap_selection.py"
    ).read_text()
    entrypoint = (REPO_ROOT / "image_target_of_oh_vs.py").read_text()
    yaml = (REPO_ROOT / "cfgs/visda/duet_first_cycle_prior_swap_selection.yaml").read_text()
    conf = (REPO_ROOT / "conf.py").read_text()

    assert "swap_conflict_selection=False" in plmatch
    assert "DUET_SWAP.ENABLED" in plmatch
    assert "all_mix_output_pred[selected] = swap_selection_payload" in plmatch
    assert "label_mask = label_mask | swap_selection_payload" in plmatch
    assert "swap_conflict_selection=True" in wrapper
    assert "first_cycle_prior=True" in wrapper
    assert "duet_first_cycle_prior_swap_selection" in entrypoint
    assert "DUET_SWAP:\n  ENABLED: True" in yaml
    assert "GATE_D: 2.0" in yaml
    assert "MIN_DIRECTION_ACCURACY: 0.8" in yaml
    assert "LAST_ACTIVE_CYCLE: 6" in yaml
    assert "_C.DUET_SWAP.ENABLED = False" in conf
    assert "_C.DUET_SWAP.GATE_D = 4.0" in conf
    assert "_C.DUET_SWAP.MIN_DIRECTION_ACCURACY = 0.0" in conf
    assert "_C.DUET_SWAP.LAST_ACTIVE_CYCLE = 8" in conf
    assert "LAST_ACTIVE_CYCLE" in plmatch
    # The exclusivity guard must not treat first_cycle_prior as a competing
    # candidate: swap selection is built on top of DUET-FCP.
    assert "first_cycle_prior is the base of this method" in plmatch


def _minimal_train_cfg():
    return SimpleNamespace(
        PCGRAD_PARAMETER_AUDIT=SimpleNamespace(ENABLED=False),
        SETTING=SimpleNamespace(DATASET="VISDA-C"),
        ACTIVE=SimpleNamespace(CYCLE=8, ARCH="ViT-B/32"),
        TEST=SimpleNamespace(BATCH_SIZE=64),
        FAILURE_AUDIT=SimpleNamespace(ENABLED=False, STOP_AFTER_PRE_CYCLE=0),
        DUET_SWAP=SimpleNamespace(
            ENABLED=True,
            GATE_D=4.0,
            MIN_DIRECTION_ACCURACY=0.0,
            LAST_ACTIVE_CYCLE=8,
        ),
        DUET_FCP=SimpleNamespace(POWER=0.5),
        DUET_BOUNDARY=SimpleNamespace(TOP_FRACTION=0.2),
        DUET_CLIP_DELAY=SimpleNamespace(FRACTION=0.1),
    )


def _import_plmatch_or_skip():
    try:
        from src.methods.oh import plmatch
    except ModuleNotFoundError as error:
        pytest.skip(f"training-path dependencies unavailable: {error}")
    return plmatch


def test_duet_fcp_plus_swap_passes_train_preflight(monkeypatch):
    plmatch = _import_plmatch_or_skip()

    class PreflightPassed(Exception):
        pass

    monkeypatch.setattr(
        plmatch.clip,
        "load",
        lambda arch: (_ for _ in ()).throw(PreflightPassed()),
    )
    # The method wrapper combination (first_cycle_prior + swap_conflict_selection)
    # must get past validation and reach model loading.
    with pytest.raises(PreflightPassed):
        plmatch.train_target(
            _minimal_train_cfg(),
            first_cycle_prior=True,
            swap_conflict_selection=True,
        )


def test_swap_rejects_other_candidates():
    plmatch = _import_plmatch_or_skip()

    with pytest.raises(ValueError, match="first_cycle_prior"):
        plmatch.train_target(
            _minimal_train_cfg(),
            swap_conflict_selection=True,
            support_conditioned_clip=True,
        )
    with pytest.raises(ValueError, match="must be run separately"):
        plmatch.train_target(
            _minimal_train_cfg(),
            first_cycle_prior=True,
            swap_conflict_selection=True,
            boundary_router=True,
        )
