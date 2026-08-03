#!/usr/bin/env python3
"""CPU-only temporal-persistence audit for locked patch-to-CLS evidence.

The candidate is an inference-only diagnostic: on the previously locked
proxy patch cohort, compare a frozen cycle-1 task candidate with the task,
CLIP, and fixed combined predictions observed before cycle 2.  It adds no loss
and performs no parameter update.  Target labels and prior oracle summaries
are parsed only after the new label-free signal has been SHA256-locked.

The available cycle-2 snapshot came from the first-cycle
support-conditioned-CLIP run.  Therefore even PASS authorizes only a matched
pure-DUET snapshot confirmation, never proxy/full training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.conflict_boundary import paired_accuracy_bootstrap_ci  # noqa: E402
from src.utils.patch_cls_temporal_persistence_audit import (  # noqa: E402
    apply_frozen_patch_memory,
    evaluate_patch_temporal_persistence_gate,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_CONFLICTS = 7_070
EXPECTED_SELECTED = 208
CLASS_NAMES = (
    "aeroplane", "bicycle", "bus", "car", "horse", "knife",
    "motorcycle", "person", "plant", "skateboard", "train", "truck",
)
BASE = Path(
    "output/uda/VISDA-C/TV/"
    "duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
)
PATCH_BASE = BASE / "patch_cls_contribution_audit/risk_control_audit"
HOLDOUT_BASE = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_feature_gravity_audit_seed2020/feature_gravity_audit/"
    "patch_cls_holdout_audit"
)
STEM = "visda_patch_cls_temporal_persistence"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    snapshots = BASE / "cycle2_conflict_memory_snapshots"
    parser.add_argument("--pre-cycle1", type=Path, default=snapshots / "pre_cycle01.npz")
    parser.add_argument("--pre-cycle2", type=Path, default=snapshots / "pre_cycle02.npz")
    parser.add_argument(
        "--snapshot-lock",
        type=Path,
        default=BASE / "cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json",
    )
    parser.add_argument(
        "--risk-signal", type=Path,
        default=PATCH_BASE / "visda_conflict_patch_cls_risk_control_label_free.npz",
    )
    parser.add_argument(
        "--risk-lock", type=Path,
        default=PATCH_BASE / "visda_conflict_patch_cls_risk_control_signal_lock.json",
    )
    parser.add_argument(
        "--risk-summary", type=Path,
        default=PATCH_BASE / "visda_conflict_patch_cls_risk_control_summary.json",
    )
    parser.add_argument(
        "--holdout-summary", type=Path,
        default=HOLDOUT_BASE / "visda_conflict_patch_cls_risk_control_holdout_summary.json",
    )
    parser.add_argument(
        "--target-list", type=Path,
        default=Path("data/VISDA-C/validation_proxy25_seed2020_list.txt"),
    )
    parser.add_argument(
        "--class-names", type=Path, default=Path("data/VISDA-C/classname.txt")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PATCH_BASE / "temporal_persistence_audit",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2_020)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_snapshot_label_free(path: Path) -> dict[str, Any]:
    required = {
        "cycle", "phase", "mix_label", "label_mask", "source_label",
        "clip_label", "task_prob", "clip_prob", "strong_task_prob",
        "sample_index", "target_label",
    }
    with np.load(path, allow_pickle=False) as source:
        missing = required.difference(source.files)
        if missing:
            raise RuntimeError(f"Snapshot is missing keys: {sorted(missing)}")
        # target_label is deliberately not indexed in this phase.
        return {
            "cycle": int(np.asarray(source["cycle"]).item()),
            "phase": str(np.asarray(source["phase"]).item()),
            "mix_label": np.asarray(source["mix_label"], dtype=np.int64).copy(),
            "label_mask": np.asarray(source["label_mask"], dtype=bool).copy(),
            "source_label": np.asarray(source["source_label"], dtype=np.int64).copy(),
            "clip_label": np.asarray(source["clip_label"], dtype=np.int64).copy(),
            "task_prob": np.asarray(source["task_prob"], dtype=np.float64).copy(),
            "clip_prob": np.asarray(source["clip_prob"], dtype=np.float64).copy(),
            "strong_task_prob": np.asarray(
                source["strong_task_prob"], dtype=np.float64
            ).copy(),
            "sample_index": np.asarray(source["sample_index"], dtype=np.int64).copy(),
        }


def _parse_labels_after_lock(path: Path) -> np.ndarray:
    labels: list[int] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            _image_path, label_text = stripped.rsplit(maxsplit=1)
            labels.append(int(label_text))
        except ValueError as error:
            raise ValueError(f"Malformed target row {line_number}: {stripped}") from error
    result = np.asarray(labels, dtype=np.int64)
    if result.shape != (EXPECTED_SAMPLES,):
        raise ValueError(f"Expected {EXPECTED_SAMPLES} labels, found {result.size}")
    if np.any(result < 0) or np.any(result >= EXPECTED_CLASSES):
        raise ValueError("Target label outside class range")
    return result


def _comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    candidate_correct = np.asarray(candidate) == labels
    baseline_correct = np.asarray(baseline) == labels
    if labels.size == 0:
        return {
            "samples": 0,
            "candidate_accuracy_pct": 0.0,
            "baseline_accuracy_pct": 0.0,
            "gain_pp": 0.0,
            "net_corrections": 0,
            "paired_bootstrap_95_ci_pp": [0.0, 0.0],
        }
    interval = paired_accuracy_bootstrap_ci(
        candidate_correct, baseline_correct, repeats=repeats, seed=seed
    )
    candidate_accuracy = float(candidate_correct.mean() * 100.0)
    baseline_accuracy = float(baseline_correct.mean() * 100.0)
    return {
        "samples": int(labels.size),
        "candidate_accuracy_pct": candidate_accuracy,
        "baseline_accuracy_pct": baseline_accuracy,
        "gain_pp": candidate_accuracy - baseline_accuracy,
        "net_corrections": int(candidate_correct.sum() - baseline_correct.sum()),
        "paired_bootstrap_95_ci_pp": list(interval),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    label_free = summary["label_free_metrics"]
    oracle = summary["oracle_diagnostic"]
    best = summary["gate"]["best_selected_baseline"]
    best_comparison = oracle["selected_comparisons"][best]
    lines = [
        "# VisDA Patch-to-CLS Temporal-Persistence Audit",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Locked patch cohort | "
            f"`{label_free['selected_samples']}` / `{EXPECTED_CONFLICTS}` conflicts "
            "| Prior label-free risk-control lock |"
        ),
        (
            "| Effective cycle-2 task corrections | "
            f"`{label_free['effective_corrections']}` "
            "| New label-free lock |"
        ),
        (
            f"| Frozen memory gain vs best selected control `{best}` | "
            f"`{best_comparison['gain_pp']:.6f}` pp; CI "
            f"`{best_comparison['paired_bootstrap_95_ci_pp']}` "
            "| Oracle diagnostic after lock |"
        ),
        (
            "| Full-proxy task macro gain | "
            f"`{oracle['full_proxy_task_macro_gain_pp']:.6f}` pp "
            "| Oracle diagnostic after lock |"
        ),
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "## Candidate",
        "",
        "Keep the cycle-1 task top-1 as frozen memory on the already locked",
        "patch-selected cohort. At the pre-cycle-2 checkpoint, replace only a",
        "different current task top-1; there is no new threshold, class route,",
        "loss, optimizer step, or target-label rule.",
        "",
        "## Provenance limitation",
        "",
        "The available snapshots came from the first-cycle support-conditioned",
        "CLIP run, whose cycle-1 accuracy replay differed from its matched control",
        "by at most 0.06 pp. PASS can authorize only a pure-DUET cycle-2 snapshot",
        "confirmation. It cannot authorize proxy or full training.",
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    required_paths = (
        args.pre_cycle1, args.pre_cycle2, args.snapshot_lock, args.risk_signal,
        args.risk_lock, args.risk_summary, args.holdout_summary,
        args.target_list, args.class_names,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    class_names = tuple(
        line.strip().replace("_", " ")
        for line in args.class_names.read_text().splitlines()
        if line.strip()
    )
    if class_names != CLASS_NAMES:
        raise RuntimeError("VisDA class-name contract changed")

    snapshot_lock = json.loads(args.snapshot_lock.read_text())
    risk_lock = json.loads(args.risk_lock.read_text())
    pre1 = _load_snapshot_label_free(args.pre_cycle1)
    pre2 = _load_snapshot_label_free(args.pre_cycle2)
    with np.load(args.risk_signal, allow_pickle=False) as source:
        risk = {name: np.asarray(source[name]).copy() for name in source.files}

    index1 = np.asarray(pre1["sample_index"], dtype=np.int64)
    index2 = np.asarray(pre2["sample_index"], dtype=np.int64)
    position_by_index = {int(index): row for row, index in enumerate(index2)}
    query_index = np.asarray(risk["query_index"], dtype=np.int64)
    query_position = np.asarray(
        [position_by_index.get(int(index), -1) for index in query_index],
        dtype=np.int64,
    )
    selected = np.asarray(risk["selected"], dtype=bool)
    initial_task = np.asarray(risk["task_candidate"], dtype=np.int64)
    initial_clip = np.asarray(risk["clip_candidate"], dtype=np.int64)
    task2_probability = np.asarray(pre2["task_prob"], dtype=np.float64)
    clip2_probability = np.asarray(pre2["clip_prob"], dtype=np.float64)
    task2 = np.asarray(pre2["source_label"], dtype=np.int64)
    clip2 = np.asarray(pre2["clip_label"], dtype=np.int64)
    row = np.arange(EXPECTED_SAMPLES)
    arithmetic = np.argmax(0.5 * (task2_probability + clip2_probability), axis=1)
    rms = np.argmax(
        np.sqrt(0.5 * (task2_probability ** 2 + clip2_probability ** 2)), axis=1
    )
    task_confidence = task2_probability[row, task2]
    clip_confidence = clip2_probability[row, clip2]
    confidence = np.where(task_confidence >= clip_confidence, task2, clip2)
    memory = apply_frozen_patch_memory(task2, query_position, selected, initial_task)

    pre1_task = np.asarray(pre1["source_label"], dtype=np.int64)
    pre1_clip = np.asarray(pre1["clip_label"], dtype=np.int64)
    input_checks = {
        "snapshot_lock_is_label_free": (
            snapshot_lock.get("phase") == "LABEL_FREE_CYCLE2_CONFLICT_MEMORY_LOCK"
            and snapshot_lock.get("contains_target_labels") is False
        ),
        "snapshot_hashes_match_lock": (
            snapshot_lock.get("inputs", {}).get("pre_cycle1_sha256")
            == _sha256(args.pre_cycle1)
            and snapshot_lock.get("inputs", {}).get("pre_cycle2_sha256")
            == _sha256(args.pre_cycle2)
        ),
        "risk_lock_is_label_free": (
            risk_lock.get("phase") == "LABEL_FREE_VISDA_PATCH_CLS_RISK_CONTROL_LOCK"
            and risk_lock.get("contains_target_labels") is False
        ),
        "risk_signal_hash_matches_lock": (
            risk_lock.get("signal_npz", {}).get("sha256") == _sha256(args.risk_signal)
        ),
        "expected_cycles_and_phase": (
            pre1["cycle"] == 1 and pre2["cycle"] == 2
            and pre1["phase"] == pre2["phase"] == "pre_cycle"
        ),
        "expected_sample_shapes": (
            index1.shape == index2.shape == task2.shape == (EXPECTED_SAMPLES,)
            and task2_probability.shape == clip2_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "snapshot_indices_match_and_are_unique": (
            np.array_equal(index1, index2)
            and np.unique(index2).size == EXPECTED_SAMPLES
        ),
        "query_contract": (
            query_index.shape == selected.shape == initial_task.shape
            == initial_clip.shape == (EXPECTED_CONFLICTS,)
            and np.unique(query_index).size == EXPECTED_CONFLICTS
            and np.all(query_position >= 0)
        ),
        "expected_selected_count": int(selected.sum()) == EXPECTED_SELECTED,
        "cycle1_candidates_reproduced": (
            np.array_equal(pre1_task[query_position], initial_task)
            and np.array_equal(pre1_clip[query_position], initial_clip)
            and np.all(initial_task != initial_clip)
        ),
        "cycle2_saved_predictions_reproduced": (
            np.array_equal(task2_probability.argmax(1), task2)
            and np.array_equal(clip2_probability.argmax(1), clip2)
        ),
        "cycle2_mix_reproduces_arithmetic": np.array_equal(
            np.asarray(pre2["mix_label"], dtype=np.int64), arithmetic
        ),
        "probabilities_finite": bool(
            np.isfinite(task2_probability).all()
            and np.isfinite(clip2_probability).all()
        ),
        "no_label_dependent_temporal_filter": bool(
            np.array_equal(memory["selected"], selected)
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Patch temporal-persistence input contract failed: {failed}")

    selected_position = query_position[selected]
    effective = np.asarray(memory["effective_correction"], dtype=bool)
    effective_position = query_position[effective]
    task2_query = task2[query_position]
    clip2_query = clip2[query_position]
    strong2_query = np.asarray(pre2["strong_task_prob"]).argmax(1)[query_position]
    selected_initial_task = initial_task[selected]
    selected_initial_clip = initial_clip[selected]
    selected_task2 = task2_query[selected]
    selected_clip2 = clip2_query[selected]
    selected_agreement = selected_task2 == selected_clip2
    transition_counts = {
        "cycle2_task_keeps_initial_task": int((selected_task2 == selected_initial_task).sum()),
        "cycle2_task_moves_to_initial_clip": int((selected_task2 == selected_initial_clip).sum()),
        "cycle2_task_moves_to_other": int(
            ((selected_task2 != selected_initial_task)
             & (selected_task2 != selected_initial_clip)).sum()
        ),
        "cycle2_task_strong_keeps_initial_task": int(
            (strong2_query[selected] == selected_initial_task).sum()
        ),
        "cycle2_task_clip_agree": int(selected_agreement.sum()),
        "cycle2_agree_on_initial_task": int(
            (selected_agreement & (selected_task2 == selected_initial_task)).sum()
        ),
        "cycle2_agree_on_initial_clip": int(
            (selected_agreement & (selected_task2 == selected_initial_clip)).sum()
        ),
        "cycle2_still_exact_initial_pair": int(
            (
                ~selected_agreement
                & (((selected_task2 == selected_initial_task)
                    & (selected_clip2 == selected_initial_clip))
                   | ((selected_task2 == selected_initial_clip)
                      & (selected_clip2 == selected_initial_task)))
            ).sum()
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    signal_path = args.output_dir / f"{STEM}_label_free.npz"
    lock_path = args.output_dir / f"{STEM}_signal_lock.json"
    oracle_path = args.output_dir / f"{STEM}_oracle_diagnostic.csv"
    classwise_path = args.output_dir / f"{STEM}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{STEM}_summary.json"
    markdown_path = args.output_dir / f"{STEM}_summary.md"
    np.savez_compressed(
        signal_path,
        query_index=query_index,
        query_position=query_position,
        selected=selected,
        effective_correction=effective,
        initial_task_candidate=initial_task,
        initial_clip_candidate=initial_clip,
        cycle2_task_prediction=task2_query,
        cycle2_clip_prediction=clip2_query,
        cycle2_strong_task_prediction=strong2_query,
        cycle2_confidence_prediction=confidence[query_position],
        cycle2_arithmetic_prediction=arithmetic[query_position],
        cycle2_rms_prediction=rms[query_position],
        cycle2_mix_prediction=np.asarray(pre2["mix_label"])[query_position],
        full_candidate_prediction=memory["prediction"],
    )
    class_shift = (
        np.bincount(memory["prediction"], minlength=EXPECTED_CLASSES)
        - np.bincount(task2, minlength=EXPECTED_CLASSES)
    )
    class_mass_shift_pp = class_shift / float(EXPECTED_SAMPLES) * 100.0
    label_free_metrics = {
        "initial_conflict_queries": EXPECTED_CONFLICTS,
        "selected_samples": int(selected.sum()),
        "selected_coverage_pct": float(selected.mean() * 100.0),
        "effective_corrections": int(effective.sum()),
        "effective_correction_full_proxy_coverage_pct": float(
            effective.sum() / EXPECTED_SAMPLES * 100.0
        ),
        "class_count_shift_vs_cycle2_task": class_shift.tolist(),
        "max_class_mass_shift_pp": float(np.abs(class_mass_shift_pp).max()),
        "transition_counts": transition_counts,
    }
    lock = {
        "phase": "LABEL_FREE_VISDA_PATCH_CLS_TEMPORAL_PERSISTENCE_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "snapshot_target_label_arrays_not_accessed_before_lock": True,
        "target_list_labels_parsed_after_this_manifest": True,
        "prior_oracle_summaries_parsed_after_this_manifest": True,
        "confirmatory_status": "exploratory_temporal_snapshot_with_known_run_provenance_limit",
        "candidate_contract": {
            "cohort": "locked_cycle1_patch_risk_control_selection",
            "memory": "locked_cycle1_task_top1_candidate",
            "application": "replace_cycle2_task_top1_on_the_same_locked_cohort",
            "new_score_threshold": False,
            "class_specific_route": False,
            "target_label_rule": False,
            "loss_change": False,
            "parameter_update": False,
        },
        "snapshot_provenance_limit": {
            "run": "first_cycle_support_conditioned_clip_cycle2_memory_audit",
            "cycle1_candidate_changes_clip_kl_soft_target": True,
            "hard_top1_changed": False,
            "reported_max_cycle1_accuracy_replay_drift_pp": 0.06,
            "pure_duet_snapshot": False,
        },
        "predeclared_gate": {
            "selected_coverage_pct": [2.0, 10.0],
            "min_effective_corrections": 20,
            "min_gain_vs_best_selected_baseline_pp": 1.0,
            "best_selected_baseline_ci_lower": "> 0",
            "must_beat_selected_baselines": [
                "cycle2_task", "cycle2_clip", "cycle2_confidence",
                "cycle2_arithmetic", "cycle2_rms", "cycle2_mix",
            ],
            "effective_gain_vs_cycle2_task_ci_lower": "> 0",
            "min_full_proxy_task_macro_gain_pp": 0.20,
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
            "max_class_mass_shift_pp": 1.0,
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "inputs": {
            "pre_cycle1": {"path": str(args.pre_cycle1), "sha256": _sha256(args.pre_cycle1)},
            "pre_cycle2": {"path": str(args.pre_cycle2), "sha256": _sha256(args.pre_cycle2)},
            "snapshot_lock": {"path": str(args.snapshot_lock), "sha256": _sha256(args.snapshot_lock)},
            "risk_signal": {"path": str(args.risk_signal), "sha256": _sha256(args.risk_signal)},
            "risk_lock": {"path": str(args.risk_lock), "sha256": _sha256(args.risk_lock)},
            "opaque_risk_summary_sha256": _sha256(args.risk_summary),
            "opaque_holdout_summary_sha256": _sha256(args.holdout_summary),
            "opaque_target_list_sha256": _sha256(args.target_list),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/patch_cls_temporal_persistence_audit.py": _sha256(
                REPO_ROOT / "src/utils/patch_cls_temporal_persistence_audit.py"
            ),
            "tools/audit_visda_patch_cls_temporal_persistence.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: reveal oracle labels and prior oracle decisions only after lock.
    for path, expected in (
        (args.risk_summary, lock["inputs"]["opaque_risk_summary_sha256"]),
        (args.holdout_summary, lock["inputs"]["opaque_holdout_summary_sha256"]),
        (args.target_list, lock["inputs"]["opaque_target_list_sha256"]),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"Input changed after temporal signal lock: {path}")
    risk_summary = json.loads(args.risk_summary.read_text())
    holdout_summary = json.loads(args.holdout_summary.read_text())
    exploratory_pass = (
        risk_summary.get("decision") == "PASS_EXPLORATORY_PATCH_CLS_RISK_CONTROL"
    )
    heldout_pass = holdout_summary.get("decision") == "PASS_HELDOUT_PATCH_CLS_RISK_CONTROL"
    labels_by_dataset_index = _parse_labels_after_lock(args.target_list)
    snapshot_labels = labels_by_dataset_index[index2]
    selected_labels = labels_by_dataset_index[query_index[selected]]
    selected_candidate = selected_initial_task
    selected_baselines = {
        "cycle2_task": task2[selected_position],
        "cycle2_clip": clip2[selected_position],
        "cycle2_confidence": confidence[selected_position],
        "cycle2_arithmetic": arithmetic[selected_position],
        "cycle2_rms": rms[selected_position],
        "cycle2_mix": np.asarray(pre2["mix_label"])[selected_position],
    }
    selected_comparisons = {
        name: _comparison(
            selected_candidate,
            baseline,
            selected_labels,
            repeats=args.bootstrap_repeats,
            seed=args.seed + offset,
        )
        for offset, (name, baseline) in enumerate(selected_baselines.items())
    }
    effective_comparison = _comparison(
        initial_task[effective],
        task2[effective_position],
        labels_by_dataset_index[query_index[effective]],
        repeats=args.bootstrap_repeats,
        seed=args.seed + 100,
    )

    candidate_correct = memory["prediction"] == snapshot_labels
    task2_correct = task2 == snapshot_labels
    class_delta = np.zeros(EXPECTED_CLASSES, dtype=np.float64)
    class_rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_mask = snapshot_labels == class_index
        selected_class = class_mask[selected_position]
        net = int(
            candidate_correct[class_mask].sum() - task2_correct[class_mask].sum()
        )
        class_delta[class_index] = net / float(class_mask.sum()) * 100.0
        class_rows.append({
            "class_index": class_index,
            "class": class_name,
            "proxy_samples": int(class_mask.sum()),
            "selected_samples": int(selected_class.sum()),
            "effective_corrections": int(class_mask[effective_position].sum()),
            "net_corrections_vs_cycle2_task": net,
            "full_proxy_accuracy_delta_pp": float(class_delta[class_index]),
            "oracle_usage": "diagnostic_only_after_temporal_signal_lock",
        })
    _write_csv(classwise_path, class_rows)
    oracle_rows: list[dict[str, Any]] = []
    for offset, risk_row in enumerate(np.flatnonzero(selected)):
        position = int(query_position[risk_row])
        oracle_rows.append({
            "sample_index": int(query_index[risk_row]),
            "oracle_target_label": int(labels_by_dataset_index[query_index[risk_row]]),
            "initial_task_candidate": int(initial_task[risk_row]),
            "initial_clip_candidate": int(initial_clip[risk_row]),
            "cycle2_task_prediction": int(task2[position]),
            "cycle2_clip_prediction": int(clip2[position]),
            "effective_correction": bool(effective[risk_row]),
            "frozen_memory_correct": bool(
                initial_task[risk_row]
                == labels_by_dataset_index[query_index[risk_row]]
            ),
            "cycle2_task_correct": bool(
                task2[position] == labels_by_dataset_index[query_index[risk_row]]
            ),
            "memory_minus_cycle2_task_correct": (
                int(
                    initial_task[risk_row]
                    == labels_by_dataset_index[query_index[risk_row]]
                )
                - int(
                    task2[position]
                    == labels_by_dataset_index[query_index[risk_row]]
                )
            ),
            "oracle_usage": "diagnostic_only_after_temporal_signal_lock",
        })
    _write_csv(oracle_path, oracle_rows)

    car_delta = float(class_delta[3])
    truck_delta = float(class_delta[11])
    car_truck_mean = float((car_delta + truck_delta) / 2.0)
    other_indices = [index for index in range(EXPECTED_CLASSES) if index not in (3, 11)]
    other_ten_mean = float(class_delta[other_indices].mean())
    macro_gain = float(class_delta.mean())
    gate = evaluate_patch_temporal_persistence_gate(
        input_contract_valid=all(input_checks.values()),
        exploratory_selector_pass_preserved=exploratory_pass,
        heldout_selector_pass_preserved=heldout_pass,
        selected_coverage_pct=label_free_metrics["selected_coverage_pct"],
        effective_corrections=label_free_metrics["effective_corrections"],
        selected_comparisons=selected_comparisons,
        effective_task_comparison=effective_comparison,
        full_proxy_task_macro_gain_pp=macro_gain,
        car_delta_pp=car_delta,
        truck_delta_pp=truck_delta,
        car_truck_mean_delta_pp=car_truck_mean,
        other_ten_mean_delta_pp=other_ten_mean,
        max_class_mass_shift_pp=label_free_metrics["max_class_mass_shift_pp"],
    )
    summary = {
        "decision": gate["decision"],
        "method": "frozen_cycle1_patch_task_candidate_as_cycle2_inference_memory",
        "confirmatory_status": "exploratory_temporal_snapshot_with_known_run_provenance_limit",
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "labels_read_only_after_signal_lock": True,
            "exploratory_selector_pass_preserved": exploratory_pass,
            "heldout_selector_pass_preserved": heldout_pass,
            "selected_comparisons": selected_comparisons,
            "effective_comparison_vs_cycle2_task": effective_comparison,
            "full_proxy_task_macro_gain_pp": macro_gain,
            "classwise": class_rows,
            "car_delta_pp": car_delta,
            "truck_delta_pp": truck_delta,
            "car_truck_mean_delta_pp": car_truck_mean,
            "other_ten_mean_delta_pp": other_ten_mean,
        },
        "gate": gate,
        "outputs": {
            "label_free_signal": str(signal_path),
            "signal_lock": str(lock_path),
            "oracle_diagnostic": str(oracle_path),
            "classwise_oracle_diagnostic": str(classwise_path),
            "markdown": str(markdown_path),
        },
        "safety": {
            "target_images_loaded": False,
            "model_or_checkpoint_loaded": False,
            "forward_calls": 0,
            "backward_calls": 0,
            "optimizer_constructed": False,
            "parameter_updates": 0,
            "proxy_training_authorized": False,
            "full_training_authorized": False,
        },
        "scope_limit": (
            "PASS authorizes only one pure-DUET run stopped after writing the "
            "pre-cycle-2 snapshot. It never authorizes proxy/full training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote label-free temporal signal: {signal_path}")
    print(f"Locked temporal signal before oracle files: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {classwise_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
