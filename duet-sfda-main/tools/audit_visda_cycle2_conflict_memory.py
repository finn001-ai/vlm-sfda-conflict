#!/usr/bin/env python
"""Audit cycle-2 support conditioning on the locked initial-conflict cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.candidate_set_gradient_audit import (  # noqa: E402
    kl_logit_descent,
    oracle_ce_logit_descent,
    paired_mean_bootstrap_ci,
    rowwise_oracle_alignment,
)
from src.utils.cycle2_conflict_memory_audit import (  # noqa: E402
    build_cycle2_conflict_memory_target,
    evaluate_cycle2_conflict_memory_gate,
)
from src.utils.support_conditioned_clip_audit import (  # noqa: E402
    full_target_class_mass_shift_pp,
    negative_first_order_burden,
    normalize_probability_matrix,
)


CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
)
REQUIRED_LABEL_FREE_KEYS = (
    "cycle",
    "phase",
    "sample_index",
    "label_mask",
    "source_label",
    "clip_label",
    "task_prob",
    "clip_prob",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_label_free(path: Path, expected_cycle: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as snapshot:
        missing = [key for key in REQUIRED_LABEL_FREE_KEYS if key not in snapshot.files]
        if missing:
            raise RuntimeError(f"{path} is missing label-free keys: {missing}")
        if "target_label" not in snapshot.files:
            raise RuntimeError(f"{path} is missing its later oracle diagnostic")
        values = {key: np.asarray(snapshot[key]).copy() for key in REQUIRED_LABEL_FREE_KEYS}
    if int(values["cycle"]) != expected_cycle or str(values["phase"]) != "pre_cycle":
        raise RuntimeError(f"{path} is not pre_cycle{expected_cycle:02d}")
    return values


def _comparison(candidate: dict[str, np.ndarray], baseline: dict[str, np.ndarray]):
    report = {}
    for metric in ("cosine", "oracle_unit_projection", "first_order"):
        difference = candidate[metric] - baseline[metric]
        report[metric] = {
            "mean_difference": float(difference.mean()),
            "paired_bootstrap_95_ci": list(paired_mean_bootstrap_ci(difference)),
        }
    return report


def _first_order_comparison(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    mask: np.ndarray,
):
    difference = candidate["first_order"][mask] - baseline["first_order"][mask]
    if difference.size == 0:
        raise RuntimeError("cycle-2 diagnostic group is empty")
    return {
        "samples": int(difference.size),
        "mean_difference": float(difference.mean()),
        "paired_bootstrap_95_ci": list(paired_mean_bootstrap_ci(difference)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-cycle1", required=True)
    parser.add_argument("--pre-cycle2", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full-target-samples", type=int, default=55388)
    return parser.parse_args()


def main():
    started = time.monotonic()
    args = _parse_args()
    pre1_path = Path(args.pre_cycle1)
    pre2_path = Path(args.pre_cycle2)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: read only label-free arrays and lock the proposed target.
    pre1 = _load_label_free(pre1_path, 1)
    pre2 = _load_label_free(pre2_path, 2)
    index1 = np.asarray(pre1["sample_index"], dtype=np.int64)
    index2 = np.asarray(pre2["sample_index"], dtype=np.int64)
    task1 = normalize_probability_matrix(pre1["task_prob"], name="cycle1_task")
    clip1 = normalize_probability_matrix(pre1["clip_prob"], name="cycle1_clip")
    task2 = normalize_probability_matrix(pre2["task_prob"], name="cycle2_task")
    clip2 = normalize_probability_matrix(pre2["clip_prob"], name="cycle2_clip")
    source1 = np.asarray(pre1["source_label"], dtype=np.int64)
    clip_label1 = np.asarray(pre1["clip_label"], dtype=np.int64)
    source2 = np.asarray(pre2["source_label"], dtype=np.int64)
    clip_label2 = np.asarray(pre2["clip_label"], dtype=np.int64)
    initial_conflict = source1 != clip_label1
    initial_position = np.flatnonzero(initial_conflict)
    cycle2_still_conflict = source2[initial_conflict] != clip_label2[initial_conflict]
    target = build_cycle2_conflict_memory_target(
        task2[initial_conflict], clip2[initial_conflict]
    )
    input_checks = {
        "sample_count_matches": index1.shape == index2.shape == source1.shape,
        "sample_index_matches": np.array_equal(index1, index2),
        "sample_index_unique": np.unique(index1).size == index1.size,
        "cycle1_saved_predictions_match": (
            np.array_equal(task1.argmax(1), source1)
            and np.array_equal(clip1.argmax(1), clip_label1)
        ),
        "cycle2_saved_predictions_match": (
            np.array_equal(task2.argmax(1), source2)
            and np.array_equal(clip2.argmax(1), clip_label2)
        ),
        "cycle1_mask_is_agreement": np.array_equal(
            np.asarray(pre1["label_mask"], dtype=bool), ~initial_conflict
        ),
        "expected_proxy_sample_count": index1.size == 13847,
        "initial_conflict_cohort_nonempty": initial_position.size > 0,
        "resolved_and_still_conflict_nonempty": (
            cycle2_still_conflict.any() and (~cycle2_still_conflict).any()
        ),
        "candidate_top1_matches_clip": np.array_equal(
            target["probability"].argmax(1), clip2[initial_conflict].argmax(1)
        ),
    }
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Cycle-2 label-free input contract failed: {failed}")

    signal_path = output_dir / "visda_cycle2_conflict_memory_label_free.npz"
    np.savez_compressed(
        signal_path,
        index=index2[initial_conflict],
        task_probability=task2[initial_conflict],
        clip_probability=clip2[initial_conflict],
        candidate_probability=target["probability"],
        support_mask=target["support"],
        cycle2_still_conflict=cycle2_still_conflict,
        cycle2_label_mask=np.asarray(pre2["label_mask"], dtype=bool)[initial_conflict],
    )
    mass_shift = full_target_class_mass_shift_pp(
        target["probability"],
        clip2[initial_conflict],
        full_target_samples=args.full_target_samples,
    )
    lock_path = output_dir / "visda_cycle2_conflict_memory_signal_lock.json"
    lock = {
        "phase": "LABEL_FREE_CYCLE2_CONFLICT_MEMORY_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "labels_read_after_this_manifest": True,
        "candidate": "cycle2_clip_conditioned_on_current_top2_union_for_cycle1_conflicts",
        "candidate_contract": {
            "memory": "cycle-1 task/CLIP top-1 conflict identity",
            "cycle2_support": "current task top-2 union current CLIP top-2",
            "target": "current CLIP probability conditioned on support",
            "hard_pseudo_label_changed": False,
            "loss_term_added": False,
            "fitted_thresholds": False,
            "target_labels": False,
        },
        "inputs": {
            "pre_cycle1_sha256": _sha256(pre1_path),
            "pre_cycle2_sha256": _sha256(pre2_path),
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": {
            "initial_conflict_samples": int(initial_position.size),
            "cycle2_still_conflict_samples": int(cycle2_still_conflict.sum()),
            "cycle2_resolved_samples": int((~cycle2_still_conflict).sum()),
            "mean_support_size": float(target["support_size"].mean()),
            "mean_retained_clip_mass": float(target["retained_clip_mass"].mean()),
            "max_abs_full_target_class_mass_shift_pp": float(
                np.abs(mass_shift).max()
            ),
            "class_mass_shift_pp": mass_shift.tolist(),
        },
        "signal_npz_sha256": _sha256(signal_path),
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: oracle diagnostic only, after the label-free target is locked.
    with np.load(pre1_path, allow_pickle=False) as snapshot1:
        labels1 = np.asarray(snapshot1["target_label"], dtype=np.int64).copy()
    with np.load(pre2_path, allow_pickle=False) as snapshot2:
        labels2 = np.asarray(snapshot2["target_label"], dtype=np.int64).copy()
    if not np.array_equal(labels1, labels2):
        raise RuntimeError("cycle snapshots disagree on oracle labels")
    labels = labels2[initial_conflict]
    task = task2[initial_conflict]
    clip = clip2[initial_conflict]
    oracle = oracle_ce_logit_descent(task, labels)
    baseline_alignment = rowwise_oracle_alignment(
        kl_logit_descent(task, clip), oracle
    )
    candidate_alignment = rowwise_oracle_alignment(
        kl_logit_descent(task, target["probability"]), oracle
    )
    comparison = _comparison(candidate_alignment, baseline_alignment)
    resolved_comparison = _first_order_comparison(
        candidate_alignment, baseline_alignment, ~cycle2_still_conflict
    )
    still_comparison = _first_order_comparison(
        candidate_alignment, baseline_alignment, cycle2_still_conflict
    )
    coverage = target["support"][np.arange(labels.size), labels]

    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_index
        difference = (
            candidate_alignment["first_order"][mask]
            - baseline_alignment["first_order"][mask]
        )
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "samples": int(mask.sum()),
                "top2_union_oracle_coverage_pct": float(coverage[mask].mean() * 100.0),
                "candidate_minus_clip_mean_first_order": float(difference.mean()),
                "clip_negative_first_order_burden": negative_first_order_burden(
                    baseline_alignment["first_order"][mask]
                ),
                "candidate_negative_first_order_burden": negative_first_order_burden(
                    candidate_alignment["first_order"][mask]
                ),
            }
        )
    classwise_path = output_dir / "visda_cycle2_conflict_memory_classwise_oracle_diagnostic.csv"
    _write_csv(classwise_path, class_rows)

    sample_rows = []
    delta_first_order = (
        candidate_alignment["first_order"] - baseline_alignment["first_order"]
    )
    for row in range(labels.size):
        sample_rows.append(
            {
                "index": int(index2[initial_conflict][row]),
                "label": int(labels[row]),
                "label_name": CLASS_NAMES[int(labels[row])],
                "cycle2_still_conflict": bool(cycle2_still_conflict[row]),
                "top2_union_covers_label": bool(coverage[row]),
                "clip_first_order": float(baseline_alignment["first_order"][row]),
                "candidate_first_order": float(candidate_alignment["first_order"][row]),
                "candidate_minus_clip_first_order": float(delta_first_order[row]),
            }
        )
    oracle_path = output_dir / "visda_cycle2_conflict_memory_oracle_diagnostic.csv"
    _write_csv(oracle_path, sample_rows)

    candidate_burden = negative_first_order_burden(
        candidate_alignment["first_order"]
    )
    baseline_burden = negative_first_order_burden(
        baseline_alignment["first_order"]
    )
    minimum_class_delta = min(
        row["candidate_minus_clip_mean_first_order"] for row in class_rows
    )
    oracle_coverage_pct = float(coverage.mean() * 100.0)
    gate = evaluate_cycle2_conflict_memory_gate(
        input_contract_valid=all(input_checks.values()),
        candidate_top1_matches_clip=input_checks["candidate_top1_matches_clip"],
        overall_comparison=comparison,
        resolved_first_order=resolved_comparison,
        still_conflict_first_order=still_comparison,
        minimum_class_first_order_delta=minimum_class_delta,
        candidate_negative_burden=candidate_burden,
        clip_negative_burden=baseline_burden,
        top2_union_oracle_coverage_pct=oracle_coverage_pct,
        max_full_mass_shift_pp=float(np.abs(mass_shift).max()),
    )
    summary = {
        "dataset": "VISDA-C",
        "seed": 2020,
        "decision": gate["decision"],
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "candidate": lock["candidate"],
        "input_contract": {"passed": all(input_checks.values()), "checks": input_checks},
        "label_free_metrics": lock["label_free_metrics"],
        "oracle_metrics": {
            "overall_comparison_vs_clip": comparison,
            "resolved_first_order_comparison_vs_clip": resolved_comparison,
            "still_conflict_first_order_comparison_vs_clip": still_comparison,
            "top2_union_oracle_coverage_pct": oracle_coverage_pct,
            "minimum_class_first_order_delta_vs_clip": minimum_class_delta,
            "clip_negative_first_order_burden": baseline_burden,
            "candidate_negative_first_order_burden": candidate_burden,
            "classwise": class_rows,
        },
        "gate": gate,
        "scope_limit": (
            "Passing authorizes one persistent-memory proxy design only; it never "
            "authorizes or starts full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path = output_dir / "visda_cycle2_conflict_memory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote label-free signal: {signal_path}")
    print(f"Locked signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {classwise_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
