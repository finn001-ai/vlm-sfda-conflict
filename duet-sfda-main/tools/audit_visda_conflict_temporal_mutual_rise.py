#!/usr/bin/env python3
"""CPU-only VisDA cycle-2 task/CLIP mutual-rise audit.

Phase 1 reads the two previously locked DUET snapshots, identifies cycle-1
conflicts still unresolved at cycle 2, and locks a rule based only on signed
cross-cycle task/CLIP centered-log probability changes.  Phase 2 reads target
labels only for explicit oracle diagnostics.  No image, model, checkpoint,
forward, backward, optimizer, parameter update, proxy run, or training is used.
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
from src.utils.spatial_causal_audit import topk_union_candidates  # noqa: E402
from src.utils.temporal_mutual_rise_audit import (  # noqa: E402
    evaluate_temporal_mutual_rise_gate,
    route_mutual_rise,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_CYCLE1_AGREEMENTS = 6_777
EXPECTED_CYCLE1_CONFLICTS = 7_070
EXPECTED_CYCLE2_UNRESOLVED = 1_978
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
DEFAULT_BASE = Path(
    "output/uda/VISDA-C/TV/"
    "duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
)
STEM = "visda_conflict_temporal_mutual_rise"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    snapshot_dir = DEFAULT_BASE / "cycle2_conflict_memory_snapshots"
    parser.add_argument(
        "--pre-cycle1", type=Path, default=snapshot_dir / "pre_cycle01.npz"
    )
    parser.add_argument(
        "--pre-cycle2", type=Path, default=snapshot_dir / "pre_cycle02.npz"
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=(
            DEFAULT_BASE
            / "cycle2_conflict_memory_audit"
            / "visda_cycle2_conflict_memory_signal_lock.json"
        ),
    )
    parser.add_argument(
        "--target-list",
        type=Path,
        default=Path("data/VISDA-C/validation_proxy25_seed2020_list.txt"),
    )
    parser.add_argument(
        "--class-names", type=Path, default=Path("data/VISDA-C/classname.txt")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE / "temporal_mutual_rise_audit",
    )
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


def _parse_labels_after_lock(path: Path) -> np.ndarray:
    labels = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            _image_path, label_text = stripped.rsplit(maxsplit=1)
            labels.append(int(label_text))
        except ValueError as error:
            raise ValueError(
                f"Malformed target row {line_number}: {stripped}"
            ) from error
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
        candidate_correct, baseline_correct, repeats=2_000, seed=seed
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


def _load_label_free_snapshot(path: Path) -> dict[str, Any]:
    required = {
        "cycle",
        "phase",
        "label_mask",
        "source_label",
        "clip_label",
        "task_prob",
        "clip_prob",
        "strong_task_prob",
        "sample_index",
        "target_label",
    }
    # target_label is required for the later contract check but not accessed here.
    with np.load(path, allow_pickle=False) as snapshot:
        missing = required.difference(snapshot.files)
        if missing:
            raise RuntimeError(f"Snapshot is missing keys: {sorted(missing)}")
        return {
            "cycle": int(np.asarray(snapshot["cycle"]).item()),
            "phase": str(np.asarray(snapshot["phase"]).item()),
            "label_mask": np.asarray(snapshot["label_mask"], dtype=bool).copy(),
            "source_label": np.asarray(
                snapshot["source_label"], dtype=np.int64
            ).copy(),
            "clip_label": np.asarray(snapshot["clip_label"], dtype=np.int64).copy(),
            "task_prob": np.asarray(snapshot["task_prob"], dtype=np.float64).copy(),
            "clip_prob": np.asarray(snapshot["clip_prob"], dtype=np.float64).copy(),
            "strong_task_prob": np.asarray(
                snapshot["strong_task_prob"], dtype=np.float64
            ).copy(),
            "sample_index": np.asarray(
                snapshot["sample_index"], dtype=np.int64
            ).copy(),
        }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    label_free = summary["label_free_metrics"]
    oracle = summary["oracle_diagnostic"]
    best = oracle["best_baseline_name"]
    comparison = oracle["comparisons"][best]
    lines = [
        "# VisDA Temporal Mutual-Rise Audit",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Cycle-2 unresolved conflict queries | "
            f"`{label_free['unresolved_queries']}` | Label-free lock |"
        ),
        (
            "| Mutual-rise routed coverage | "
            f"`{label_free['route_coverage_pct']:.6f}%` | Label-free lock |"
        ),
        (
            "| Weak/strong routed-union decision stability | "
            f"`{label_free['routed_union_decision_stability_pct']:.6f}%` "
            "| Label-free lock |"
        ),
        (
            f"| Gain vs best matched baseline `{best}` | "
            f"`{comparison['gain_pp']:.6f}` pp; CI "
            f"`{comparison['paired_bootstrap_95_ci_pp']}` "
            "| Oracle diagnostic after lock |"
        ),
        (
            "| Top-2 union oracle coverage | "
            f"`{oracle['candidate_set_coverage_pct']:.6f}%` "
            "| Oracle diagnostic after lock |"
        ),
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "## Label-free rule",
        "",
        "For cycle-1 conflicts still unresolved at cycle 2, compute each model's",
        "centered-log probability change. Inside the current task-top2 union",
        "CLIP-top2 set, route away from current CLIP only when the selected class",
        "has positive relative support change in both task and CLIP. The proposed",
        "soft target only swaps the selected and CLIP-top1 probabilities.",
        "",
        "This is not the archived graph-temporal fused teacher or model-snapshot",
        "ensemble. It uses no graph, agreement reference pool, fitted threshold,",
        "new loss coefficient, target label, or class-specific route.",
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend(
        [
            "",
            "PASS authorizes review of one matched proxy design only. This audit",
            "does not authorize or start proxy/full VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for input_path in (
        args.pre_cycle1,
        args.pre_cycle2,
        args.source_lock,
        args.target_list,
        args.class_names,
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing temporal mutual-rise input: {input_path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    class_names = tuple(
        line.strip().replace("_", " ")
        for line in args.class_names.read_text().splitlines()
        if line.strip()
    )
    if class_names != CLASS_NAMES:
        raise RuntimeError("VisDA class-name contract changed")
    source_lock = json.loads(args.source_lock.read_text())
    pre1_sha256 = _sha256(args.pre_cycle1)
    pre2_sha256 = _sha256(args.pre_cycle2)
    pre1 = _load_label_free_snapshot(args.pre_cycle1)
    pre2 = _load_label_free_snapshot(args.pre_cycle2)

    index1 = pre1["sample_index"]
    index2 = pre2["sample_index"]
    task1 = pre1["task_prob"]
    task2 = pre2["task_prob"]
    clip1 = pre1["clip_prob"]
    clip2 = pre2["clip_prob"]
    strong_task1 = pre1["strong_task_prob"]
    strong_task2 = pre2["strong_task_prob"]
    task1_label = pre1["source_label"]
    task2_label = pre2["source_label"]
    clip1_label = pre1["clip_label"]
    clip2_label = pre2["clip_label"]
    mask1 = pre1["label_mask"]
    mask2 = pre2["label_mask"]
    cycle1_conflict = task1_label != clip1_label
    cycle2_agreement = task2_label == clip2_label
    expected_mask2 = mask1 | (~mask1 & cycle2_agreement)
    unresolved = cycle1_conflict & ~mask2
    query = np.flatnonzero(unresolved)
    candidates = topk_union_candidates(task2[query], clip2[query], top_k=2)
    routed = route_mutual_rise(
        task1[query], task2[query], clip1[query], clip2[query], candidates
    )
    strong_routed = route_mutual_rise(
        strong_task1[query],
        strong_task2[query],
        clip1[query],
        clip2[query],
        candidates,
    )
    routed_union = routed["routed"] | strong_routed["routed"]
    routed_union_stability = (
        float(
            (
                routed["prediction"][routed_union]
                == strong_routed["prediction"][routed_union]
            ).mean()
            * 100.0
        )
        if routed_union.any()
        else 0.0
    )
    route_coverage = float(routed["routed"].mean() * 100.0)

    candidate_full_target = clip2.copy()
    candidate_full_target[query] = routed["target_probability"]
    full_target_mass_shift_pp = (
        candidate_full_target - clip2
    ).mean(axis=0) * 100.0
    max_full_target_mass_shift = float(np.abs(full_target_mass_shift_pp).max())

    probability_arrays = (task1, task2, clip1, clip2, strong_task1, strong_task2)
    input_checks = {
        "snapshots_match_cycle_memory_lock": (
            pre1_sha256 == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
            and pre2_sha256
            == source_lock.get("inputs", {}).get("pre_cycle2_sha256")
        ),
        "snapshots_are_pre_cycles_one_and_two": (
            pre1["cycle"] == 1
            and pre2["cycle"] == 2
            and pre1["phase"] == pre2["phase"] == "pre_cycle"
        ),
        "sample_indices_align_and_match_proxy_order": (
            np.array_equal(index1, index2)
            and np.array_equal(index2, np.arange(EXPECTED_SAMPLES))
        ),
        "probability_shapes": all(
            value.shape == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
            for value in probability_arrays
        ),
        "probabilities_finite_normalized": all(
            np.isfinite(value).all()
            and np.all(value >= 0.0)
            and np.allclose(value.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5)
            for value in probability_arrays
        ),
        "saved_predictions_match_weak_probabilities": (
            np.array_equal(task1_label, task1.argmax(axis=1))
            and np.array_equal(task2_label, task2.argmax(axis=1))
            and np.array_equal(clip1_label, clip1.argmax(axis=1))
            and np.array_equal(clip2_label, clip2.argmax(axis=1))
        ),
        "cycle1_mask_matches_agreement": np.array_equal(
            mask1, ~cycle1_conflict
        ),
        "cycle2_mask_matches_monotonic_duet": np.array_equal(
            mask2, expected_mask2
        ),
        "expected_cycle1_counts": (
            int(mask1.sum()) == EXPECTED_CYCLE1_AGREEMENTS
            and int(cycle1_conflict.sum()) == EXPECTED_CYCLE1_CONFLICTS
        ),
        "expected_cycle2_unresolved_count": query.size == EXPECTED_CYCLE2_UNRESOLVED,
        "unresolved_queries_are_current_conflicts": bool(
            np.all(task2_label[query] != clip2_label[query])
        ),
        "candidate_never_leaves_top2_union": bool(
            np.all((candidates == routed["prediction"][:, None]).any(axis=1))
        ),
        "candidate_target_preserves_row_mass": bool(
            np.allclose(
                routed["target_probability"].sum(axis=1),
                1.0,
                atol=1e-12,
                rtol=1e-12,
            )
        ),
        "route_uses_only_joint_positive_rise": bool(
            np.all(routed["selected_task_velocity"][routed["routed"]] > 0.0)
            and np.all(
                routed["selected_clip_velocity"][routed["routed"]] > 0.0
            )
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Temporal mutual-rise input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    signal_path = args.output_dir / f"{STEM}_label_free.npz"
    lock_path = args.output_dir / f"{STEM}_signal_lock.json"
    oracle_path = args.output_dir / f"{STEM}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{STEM}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{STEM}_summary.json"
    markdown_path = args.output_dir / f"{STEM}_summary.md"
    np.savez_compressed(
        signal_path,
        query_index=query,
        candidate_set=candidates,
        candidate_prediction=routed["prediction"],
        candidate_target_probability=routed["target_probability"].astype(np.float32),
        routed=routed["routed"],
        selected_class=routed["selected_class"],
        selected_task_velocity=routed["selected_task_velocity"].astype(np.float32),
        selected_clip_velocity=routed["selected_clip_velocity"].astype(np.float32),
        selected_mutual_score=routed["selected_mutual_score"].astype(np.float32),
        fallback_mutual_score=routed["fallback_mutual_score"].astype(np.float32),
        strong_view_prediction=strong_routed["prediction"],
        strong_view_routed=strong_routed["routed"],
        fixed_task_prediction=task2_label[query],
        fixed_clip_prediction=clip2_label[query],
    )
    label_free_metrics = {
        "samples": EXPECTED_SAMPLES,
        "cycle1_conflicts": EXPECTED_CYCLE1_CONFLICTS,
        "unresolved_queries": int(query.size),
        "routed_queries": int(routed["routed"].sum()),
        "route_coverage_pct": route_coverage,
        "strong_view_routed_queries": int(strong_routed["routed"].sum()),
        "weak_strong_routed_union_queries": int(routed_union.sum()),
        "routed_union_decision_stability_pct": routed_union_stability,
        "mean_routed_task_velocity": (
            float(routed["selected_task_velocity"][routed["routed"]].mean())
            if routed["routed"].any()
            else 0.0
        ),
        "mean_routed_clip_velocity": (
            float(routed["selected_clip_velocity"][routed["routed"]].mean())
            if routed["routed"].any()
            else 0.0
        ),
        "full_target_mass_shift_pp": {
            name: float(full_target_mass_shift_pp[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "max_abs_full_target_mass_shift_pp": max_full_target_mass_shift,
    }
    lock = {
        "phase": "LABEL_FREE_VISDA_TEMPORAL_MUTUAL_RISE_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshots_contain_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "memory": "same_proxy_sample_pre_cycle1_and_pre_cycle2",
            "query": "cycle1_task_clip_conflicts_still_unresolved_at_cycle2",
            "candidate_set": "current_task_top2_union_current_clip_top2",
            "task_signal": "cycle2_minus_cycle1_centered_log_probability",
            "clip_signal": "cycle2_minus_cycle1_centered_log_probability",
            "score": "minimum_of_task_and_clip_signed_velocity",
            "route_condition": "selected_class_differs_from_clip_and_both_velocities_positive",
            "fallback": "current_clip_top1_and_current_clip_soft_probability",
            "soft_target_change": "swap_selected_and_clip_top1_probability_mass_only",
            "strong_view_used_for_rule": False,
            "strong_view_used_only_for_stability_diagnostic": True,
            "graph": False,
            "agreement_reference_pool": False,
            "fitted_thresholds": False,
            "class_specific_route": False,
            "target_label_thresholds": False,
            "new_loss_or_loss_weight": False,
            "training_change_in_this_audit": False,
        },
        "predeclared_gate": {
            "route_coverage_pct": [5.0, 50.0],
            "min_routed_union_decision_stability_pct": 90.0,
            "min_candidate_set_coverage_pct": 90.0,
            "min_per_class_candidate_set_coverage_pct": 85.0,
            "min_accuracy_gain_vs_best_baseline_pp": 1.0,
            "best_baseline_paired_ci_lower": "> 0",
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
            "max_full_target_mass_shift_pp": 1.0,
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "literature_provenance": {
            "paper": "Source-Free Domain Adaptation with Vision-Language Prior",
            "method": "DIFO++",
            "paper_url": "https://arxiv.org/abs/2604.17748",
            "official_code": "https://github.com/tntek/DIFO-Plus",
            "borrowed_information": "historical_prediction_memory_only",
            "not_claimed": "the_published_DIFO++_memory_or_gap_reduction_method",
        },
        "historical_nonduplication": {
            "stage10_graph_temporal_residual": (
                "used stable graph/fused-teacher top1 and an added residual loss"
            ),
            "stage14_visda_final": 91.04,
            "stage17_trajectory_ensemble": "averaged model snapshot logits",
            "this_audit": (
                "uses joint signed task/CLIP class velocity and an exact CLIP "
                "probability swap; no graph, fused teacher, or model ensemble"
            ),
        },
        "inputs": {
            "pre_cycle1_snapshot": {
                "path": str(args.pre_cycle1),
                "sha256": pre1_sha256,
            },
            "pre_cycle2_snapshot": {
                "path": str(args.pre_cycle2),
                "sha256": pre2_sha256,
            },
            "cycle2_memory_signal_lock": {
                "path": str(args.source_lock),
                "sha256": _sha256(args.source_lock),
            },
            "target_list_opaque_sha256": _sha256(args.target_list),
            "class_names_sha256": _sha256(args.class_names),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/temporal_mutual_rise_audit.py": _sha256(
                REPO_ROOT / "src/utils/temporal_mutual_rise_audit.py"
            ),
            "tools/audit_visda_conflict_temporal_mutual_rise.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: explicit oracle diagnostic, strictly after the signal lock.
    target_hash_matches = (
        _sha256(args.target_list) == lock["inputs"]["target_list_opaque_sha256"]
    )
    labels = _parse_labels_after_lock(args.target_list)
    with np.load(args.pre_cycle1, allow_pickle=False) as snapshot1:
        embedded_labels1 = np.asarray(
            snapshot1["target_label"], dtype=np.int64
        ).copy()
    with np.load(args.pre_cycle2, allow_pickle=False) as snapshot2:
        embedded_labels2 = np.asarray(
            snapshot2["target_label"], dtype=np.int64
        ).copy()
    labels_match_snapshots = bool(
        np.array_equal(embedded_labels1, embedded_labels2)
        and np.array_equal(labels[index1], embedded_labels1)
    )
    labels = labels[index2]
    query_labels = labels[query]

    task_confidence = task2.max(axis=1)
    clip_confidence = clip2.max(axis=1)
    confidence_prediction = np.where(
        task_confidence >= clip_confidence, task2_label, clip2_label
    )
    arithmetic_prediction = (0.5 * (task2 + clip2)).argmax(axis=1)
    rms_prediction = np.sqrt(0.5 * (task2**2 + clip2**2)).argmax(axis=1)
    baselines = {
        "fixed_task": task2_label[query],
        "fixed_clip": clip2_label[query],
        "confidence_choice": confidence_prediction[query],
        "arithmetic": arithmetic_prediction[query],
        "rms": rms_prediction[query],
    }
    comparisons = {
        name: _comparison(
            routed["prediction"], baseline, query_labels, seed=2_020 + offset
        )
        for offset, (name, baseline) in enumerate(baselines.items())
    }
    best_baseline_name = max(
        comparisons, key=lambda name: comparisons[name]["baseline_accuracy_pct"]
    )
    best_baseline = baselines[best_baseline_name]
    candidate_coverage = (candidates == query_labels[:, None]).any(axis=1)

    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_mask = query_labels == class_index
        if not class_mask.any():
            raise RuntimeError(f"No unresolved oracle rows for class {class_name}")
        candidate_accuracy = float(
            (routed["prediction"][class_mask] == query_labels[class_mask]).mean()
            * 100.0
        )
        baseline_accuracy = float(
            (best_baseline[class_mask] == query_labels[class_mask]).mean() * 100.0
        )
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "unresolved_samples": int(class_mask.sum()),
                "routed_samples": int((routed["routed"] & class_mask).sum()),
                "route_coverage_pct": float(
                    routed["routed"][class_mask].mean() * 100.0
                ),
                "candidate_set_coverage_pct": float(
                    candidate_coverage[class_mask].mean() * 100.0
                ),
                "candidate_accuracy_pct": candidate_accuracy,
                "best_baseline_name": best_baseline_name,
                "best_baseline_accuracy_pct": baseline_accuracy,
                "candidate_minus_best_baseline_pp": (
                    candidate_accuracy - baseline_accuracy
                ),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(class_path, class_rows)

    oracle_rows = []
    for row, index in enumerate(query):
        oracle_rows.append(
            {
                "proxy_index": int(index2[index]),
                "oracle_target_label": int(query_labels[row]),
                "cycle1_task_top1": int(task1_label[index]),
                "cycle1_clip_top1": int(clip1_label[index]),
                "cycle2_task_top1": int(task2_label[index]),
                "cycle2_clip_top1": int(clip2_label[index]),
                "candidate_prediction": int(routed["prediction"][row]),
                "candidate_correct": bool(
                    routed["prediction"][row] == query_labels[row]
                ),
                "routed": bool(routed["routed"][row]),
                "selected_class": int(routed["selected_class"][row]),
                "selected_task_velocity": float(
                    routed["selected_task_velocity"][row]
                ),
                "selected_clip_velocity": float(
                    routed["selected_clip_velocity"][row]
                ),
                "candidate_set_covers_label": bool(candidate_coverage[row]),
                "strong_view_prediction": int(strong_routed["prediction"][row]),
                "strong_view_routed": bool(strong_routed["routed"][row]),
                "best_baseline_name": best_baseline_name,
                "best_baseline_prediction": int(best_baseline[row]),
                "candidate_net_correction": int(
                    routed["prediction"][row] == query_labels[row]
                )
                - int(best_baseline[row] == query_labels[row]),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(oracle_path, oracle_rows)

    class_delta = {
        row["class"]: row["candidate_minus_best_baseline_pp"]
        for row in class_rows
    }
    car_delta = float(class_delta["car"])
    truck_delta = float(class_delta["truck"])
    car_truck_mean_delta = 0.5 * (car_delta + truck_delta)
    other_ten_mean_delta = float(
        np.mean(
            [
                value
                for name, value in class_delta.items()
                if name not in {"car", "truck"}
            ]
        )
    )
    input_contract_valid = bool(
        all(input_checks.values()) and target_hash_matches and labels_match_snapshots
    )
    gate = evaluate_temporal_mutual_rise_gate(
        input_contract_valid=input_contract_valid,
        route_coverage_pct=route_coverage,
        routed_union_decision_stability_pct=routed_union_stability,
        candidate_set_coverage_pct=float(candidate_coverage.mean() * 100.0),
        minimum_class_candidate_coverage_pct=min(
            row["candidate_set_coverage_pct"] for row in class_rows
        ),
        comparisons=comparisons,
        best_baseline_name=best_baseline_name,
        car_delta_pp=car_delta,
        truck_delta_pp=truck_delta,
        car_truck_mean_delta_pp=car_truck_mean_delta,
        other_ten_mean_delta_pp=other_ten_mean_delta,
        max_full_target_mass_shift_pp=max_full_target_mass_shift,
    )
    summary = {
        "dataset": "VisDA-C",
        "seed": 2020,
        "decision": gate["decision"],
        "labels_used_only_after_signal_lock": True,
        "signal_lock_sha256": _sha256(lock_path),
        "candidate_contract": lock["candidate_contract"],
        "literature_provenance": lock["literature_provenance"],
        "historical_nonduplication": lock["historical_nonduplication"],
        "input_contract": {
            "passed": input_contract_valid,
            "checks": {
                **input_checks,
                "target_list_hash_matches_after_lock": target_hash_matches,
                "target_labels_match_embedded_snapshots_after_lock": (
                    labels_match_snapshots
                ),
            },
        },
        "label_free_metrics": label_free_metrics,
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "candidate_set_coverage_pct": float(
                candidate_coverage.mean() * 100.0
            ),
            "minimum_class_candidate_coverage_pct": min(
                row["candidate_set_coverage_pct"] for row in class_rows
            ),
            "comparisons": comparisons,
            "best_baseline_name": best_baseline_name,
            "routed_subset_comparison_vs_fixed_clip": _comparison(
                routed["prediction"][routed["routed"]],
                baselines["fixed_clip"][routed["routed"]],
                query_labels[routed["routed"]],
                seed=2_040,
            ),
            "classwise": class_rows,
            "car_delta_pp": car_delta,
            "truck_delta_pp": truck_delta,
            "car_truck_mean_delta_pp": car_truck_mean_delta,
            "other_ten_mean_delta_pp": other_ten_mean_delta,
        },
        "gate": gate,
        "outputs": {
            "label_free_signal": str(signal_path),
            "signal_lock": str(lock_path),
            "oracle_diagnostic": str(oracle_path),
            "classwise_oracle_diagnostic": str(class_path),
            "markdown": str(markdown_path),
        },
        "scope_limit": (
            "PASS authorizes review of one matched proxy design only. This audit "
            "never authorizes or starts proxy/full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote label-free mutual-rise signal: {signal_path}")
    print(f"Locked mutual-rise signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
