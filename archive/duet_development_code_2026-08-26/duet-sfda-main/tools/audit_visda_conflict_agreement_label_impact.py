#!/usr/bin/env python3
"""CPU-only agreement-anchored label-impact audit for VisDA conflicts.

Phase 1 uses cycle-1 DUET agreements, task probabilities, and frozen task
features to form a diagonal empirical-Fisher loss landscape.  Every conflict
is adjudicated only within the task-top2 union CLIP-top2 candidate set.  The
target list and the snapshot's embedded labels are accessed strictly after the
label-free signal and lock are written.  No image, model, checkpoint, forward,
backward, optimizer, parameter update, proxy run, or training is used.
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

from src.utils.agreement_label_impact_audit import (  # noqa: E402
    evaluate_agreement_label_impact_gate,
    fit_agreement_label_impact,
    label_impact_score,
    select_candidate_by_label_impact,
    stratified_alternating_reference_masks,
)
from src.utils.conflict_boundary import paired_accuracy_bootstrap_ci  # noqa: E402
from src.utils.spatial_causal_audit import topk_union_candidates  # noqa: E402


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_AGREEMENTS = 6_777
EXPECTED_CONFLICTS = 7_070
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
STEM = "visda_conflict_agreement_label_impact"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_BASE / "cycle2_conflict_memory_snapshots/pre_cycle01.npz",
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
        default=DEFAULT_BASE / "agreement_label_impact_audit",
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
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    candidate_correct = candidate == labels
    baseline_correct = baseline == labels
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
    oracle = summary["oracle_diagnostic"]
    best_name = oracle["best_baseline_name"]
    best = oracle["comparisons"][best_name]
    lines = [
        "# VisDA Agreement-Anchored Label-Impact Audit",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Agreement reference accuracy | "
            f"`{oracle['agreement_reference_accuracy_pct']:.6f}%` "
            "| Oracle diagnostic after lock |"
        ),
        (
            "| Minimum alternating-split decision stability | "
            f"`{summary['label_free_metrics']['minimum_split_decision_stability_pct']:.6f}%` "
            "| Label-free lock |"
        ),
        (
            f"| Conflict gain vs best baseline `{best_name}` | "
            f"`{best['gain_pp']:.6f}` pp; CI "
            f"`{best['paired_bootstrap_95_ci_pp']}` "
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
        "Cycle-1 task/CLIP agreements define a class-conditional classifier-head",
        "loss landscape. A diagonal empirical Fisher rescales each head-gradient",
        "coordinate. For every conflict and candidate label, the audit computes",
        "the first-order improvement that its hypothetical CE step would produce",
        "on agreement references of the same pseudo class, then chooses the",
        "maximum-impact member of task-top2 union CLIP-top2. No target label,",
        "class-specific route, fitted threshold, or learned coefficient enters",
        "the rule.",
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
            "PASS authorizes review of one exact trainable-parameter audit only.",
            "It does not authorize or start a proxy or full VisDA training run.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    if args.bootstrap_repeats < 100:
        raise ValueError("bootstrap repeats must be at least 100")
    for input_path in (
        args.snapshot,
        args.source_lock,
        args.target_list,
        args.class_names,
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing label-impact input: {input_path}")
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
    snapshot_sha256 = _sha256(args.snapshot)

    required = {
        "cycle",
        "label_mask",
        "source_label",
        "clip_label",
        "task_prob",
        "clip_prob",
        "task_feature",
        "sample_index",
        "target_label",
    }
    # Phase 1 deliberately does not access snapshot["target_label"].
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        missing = required.difference(snapshot.files)
        if missing:
            raise RuntimeError(f"Snapshot is missing keys: {sorted(missing)}")
        cycle = int(np.asarray(snapshot["cycle"]).item())
        admitted = np.asarray(snapshot["label_mask"], dtype=bool).copy()
        task_label = np.asarray(snapshot["source_label"], dtype=np.int64).copy()
        clip_label = np.asarray(snapshot["clip_label"], dtype=np.int64).copy()
        task_probability = np.asarray(snapshot["task_prob"], dtype=np.float64).copy()
        clip_probability = np.asarray(snapshot["clip_prob"], dtype=np.float64).copy()
        task_feature_storage_dtype = str(snapshot["task_feature"].dtype)
        task_feature = np.asarray(snapshot["task_feature"], dtype=np.float64).copy()
        sample_index = np.asarray(snapshot["sample_index"], dtype=np.int64).copy()

    agreement = task_label == clip_label
    conflict = ~agreement
    query = np.flatnonzero(conflict)
    candidates = topk_union_candidates(
        task_probability[query], clip_probability[query], top_k=2
    )

    full_model = fit_agreement_label_impact(
        task_probability,
        task_feature,
        task_label,
        agreement,
        class_count=EXPECTED_CLASSES,
    )
    full_score = label_impact_score(
        task_probability[query], task_feature[query], full_model
    )
    candidate = select_candidate_by_label_impact(full_score, candidates)

    first_reference, second_reference = stratified_alternating_reference_masks(
        task_label,
        agreement,
        sample_index,
        class_count=EXPECTED_CLASSES,
    )
    first_model = fit_agreement_label_impact(
        task_probability,
        task_feature,
        task_label,
        first_reference,
        class_count=EXPECTED_CLASSES,
    )
    second_model = fit_agreement_label_impact(
        task_probability,
        task_feature,
        task_label,
        second_reference,
        class_count=EXPECTED_CLASSES,
    )
    first_candidate = select_candidate_by_label_impact(
        label_impact_score(task_probability[query], task_feature[query], first_model),
        candidates,
    )
    second_candidate = select_candidate_by_label_impact(
        label_impact_score(task_probability[query], task_feature[query], second_model),
        candidates,
    )
    split_stability = {
        "first_alternating_half_fit_pct": float(
            (first_candidate["prediction"] == candidate["prediction"]).mean() * 100.0
        ),
        "second_alternating_half_fit_pct": float(
            (second_candidate["prediction"] == candidate["prediction"]).mean() * 100.0
        ),
        "half_to_half_pct": float(
            (first_candidate["prediction"] == second_candidate["prediction"]).mean()
            * 100.0
        ),
    }
    minimum_split_stability = float(min(split_stability.values()))

    confidence_prediction = np.where(
        task_probability.max(axis=1) >= clip_probability.max(axis=1),
        task_label,
        clip_label,
    )
    arithmetic_prediction = (0.5 * (task_probability + clip_probability)).argmax(axis=1)
    rms_prediction = np.sqrt(
        0.5 * (task_probability**2 + clip_probability**2)
    ).argmax(axis=1)
    fixed_clip_full = clip_label.copy()
    candidate_full = fixed_clip_full.copy()
    candidate_full[query] = candidate["prediction"]
    class_mass_shift_pp = (
        (
            np.bincount(candidate_full, minlength=EXPECTED_CLASSES)
            - np.bincount(fixed_clip_full, minlength=EXPECTED_CLASSES)
        )
        / EXPECTED_SAMPLES
        * 100.0
    )

    input_checks = {
        "snapshot_matches_cycle_memory_lock": (
            snapshot_sha256 == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
        ),
        "snapshot_is_pre_cycle1": cycle == 1,
        "probability_shapes": (
            task_probability.shape
            == clip_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "task_feature_shape": (
            task_feature.ndim == 2
            and task_feature.shape[0] == EXPECTED_SAMPLES
            and task_feature.shape[1] > 0
        ),
        "label_free_arrays_finite": all(
            np.isfinite(value).all()
            for value in (task_probability, clip_probability, task_feature)
        ),
        "probabilities_normalized": all(
            np.allclose(value.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5)
            for value in (task_probability, clip_probability)
        ),
        "saved_predictions_match_probabilities": (
            np.array_equal(task_label, task_probability.argmax(axis=1))
            and np.array_equal(clip_label, clip_probability.argmax(axis=1))
        ),
        "admitted_mask_matches_cycle1_agreement": np.array_equal(admitted, agreement),
        "expected_agreement_and_conflict_counts": (
            int(agreement.sum()) == EXPECTED_AGREEMENTS
            and int(conflict.sum()) == EXPECTED_CONFLICTS
        ),
        "sample_indices_are_proxy_order": np.array_equal(
            sample_index, np.arange(EXPECTED_SAMPLES)
        ),
        "every_class_has_split_references": bool(
            np.all(first_model["reference_count"] >= 2)
            and np.all(second_model["reference_count"] >= 2)
        ),
        "candidate_never_leaves_top2_union": bool(
            np.all((candidates == candidate["prediction"][:, None]).any(axis=1))
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Label-impact input contract failed: {failed}")

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
        candidate_prediction=candidate["prediction"],
        candidate_impact_margin=candidate["margin"].astype(np.float32),
        full_reference_count=full_model["reference_count"],
        fisher_diagonal=full_model["fisher_diagonal"].astype(np.float32),
        first_alternating_reference_count=first_model["reference_count"],
        second_alternating_reference_count=second_model["reference_count"],
        first_alternating_candidate_prediction=first_candidate["prediction"],
        second_alternating_candidate_prediction=second_candidate["prediction"],
        fixed_task_prediction=task_label[query],
        fixed_clip_prediction=clip_label[query],
        confidence_prediction=confidence_prediction[query],
        arithmetic_prediction=arithmetic_prediction[query],
        rms_prediction=rms_prediction[query],
    )
    label_free_metrics = {
        "samples": EXPECTED_SAMPLES,
        "agreement_references": EXPECTED_AGREEMENTS,
        "conflict_queries": EXPECTED_CONFLICTS,
        "task_feature_dimension": int(task_feature.shape[1]),
        "task_feature_storage_dtype": task_feature_storage_dtype,
        "reference_count_by_pseudo_class": full_model["reference_count"].tolist(),
        "numerical_fisher_floor": float(full_model["numerical_floor"]),
        "stratified_alternating_split_decision_stability_pct": split_stability,
        "minimum_split_decision_stability_pct": minimum_split_stability,
        "changed_from_fixed_clip": int(
            np.sum(candidate["prediction"] != clip_label[query])
        ),
        "class_mass_shift_pp": {
            name: float(class_mass_shift_pp[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "max_class_mass_shift_pp": float(np.abs(class_mass_shift_pp).max()),
    }
    lock = {
        "phase": "LABEL_FREE_VISDA_AGREEMENT_LABEL_IMPACT_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "reference_pool": "cycle1_task_clip_top1_agreements",
            "reference_pseudo_label": "shared_task_clip_top1",
            "query": "cycle1_task_clip_top1_conflicts",
            "candidate_set": "task_top2_union_clip_top2",
            "parameter_space": "frozen_linear_classifier_head_weight_and_bias_surrogate",
            "landscape": "global_diagonal_empirical_Fisher_on_agreements",
            "reference_direction": "per_pseudo_class_mean_CE_head_gradient",
            "selection": "maximum_first_order_reference_loss_improvement_within_candidate_set",
            "class_specific_route": False,
            "fitted_damping": False,
            "fitted_thresholds": False,
            "target_label_thresholds": False,
            "training_change_in_this_audit": False,
        },
        "predeclared_gate": {
            "min_agreement_reference_accuracy_pct": 90.0,
            "min_split_decision_stability_pct": 90.0,
            "min_candidate_set_coverage_pct": 90.0,
            "min_per_class_candidate_set_coverage_pct": 85.0,
            "min_accuracy_gain_vs_best_baseline_pp": 1.0,
            "best_baseline_paired_ci_lower": "> 0",
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
            "max_class_mass_shift_pp": 1.0,
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "literature_provenance": {
            "paper": "Partial Label Learning via Label Influence Function",
            "venue": "ICML 2022",
            "paper_url": "https://proceedings.mlr.press/v162/gong22c.html",
            "borrowed_information": "candidate_labels_should_be_compared_by_model_impact_not_only_loss_or_confidence",
            "not_claimed": "the_published_PLL_IF_optimizer_or_exact_inverse_Hessian",
        },
        "inputs": {
            "pre_cycle1_snapshot": {
                "path": str(args.snapshot),
                "sha256": snapshot_sha256,
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
            "src/utils/agreement_label_impact_audit.py": _sha256(
                REPO_ROOT / "src/utils/agreement_label_impact_audit.py"
            ),
            "tools/audit_visda_conflict_agreement_label_impact.py": _sha256(
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
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        embedded_labels = np.asarray(snapshot["target_label"], dtype=np.int64).copy()
    labels_match_snapshot = np.array_equal(labels, embedded_labels)
    if not target_hash_matches or not labels_match_snapshot:
        raise RuntimeError("Oracle label provenance check failed after lock")

    query_labels = labels[query]
    predictions = {
        "candidate": candidate["prediction"],
        "fixed_task": task_label[query],
        "fixed_clip": clip_label[query],
        "confidence_choice": confidence_prediction[query],
        "arithmetic": arithmetic_prediction[query],
        "rms": rms_prediction[query],
    }
    comparisons = {
        name: _comparison(
            predictions["candidate"],
            predictions[name],
            query_labels,
            repeats=args.bootstrap_repeats,
            seed=args.seed + offset,
        )
        for offset, name in enumerate(
            ("fixed_task", "fixed_clip", "confidence_choice", "arithmetic", "rms")
        )
    }
    baseline_accuracy = {
        name: result["baseline_accuracy_pct"] for name, result in comparisons.items()
    }
    best_baseline_name = max(
        baseline_accuracy, key=lambda name: (baseline_accuracy[name], name)
    )
    best_prediction = predictions[best_baseline_name]

    candidate_coverage = (candidates == query_labels[:, None]).any(axis=1)
    agreement_reference_accuracy = float(
        (task_label[agreement] == labels[agreement]).mean() * 100.0
    )
    oracle_rows = []
    for row, global_index in enumerate(query):
        oracle_rows.append(
            {
                "sample_index": int(global_index),
                "oracle_target_label": int(query_labels[row]),
                "oracle_candidate_set_contains_target": bool(candidate_coverage[row]),
                "candidate_prediction": int(predictions["candidate"][row]),
                "candidate_correct": bool(
                    predictions["candidate"][row] == query_labels[row]
                ),
                "fixed_task_prediction": int(predictions["fixed_task"][row]),
                "fixed_clip_prediction": int(predictions["fixed_clip"][row]),
                "confidence_prediction": int(predictions["confidence_choice"][row]),
                "arithmetic_prediction": int(predictions["arithmetic"][row]),
                "rms_prediction": int(predictions["rms"][row]),
                "candidate_minus_best_correct": int(
                    predictions["candidate"][row] == query_labels[row]
                )
                - int(best_prediction[row] == query_labels[row]),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(oracle_path, oracle_rows)

    class_rows = []
    class_delta = np.empty(EXPECTED_CLASSES, dtype=np.float64)
    for class_index, class_name in enumerate(CLASS_NAMES):
        selected = query_labels == class_index
        candidate_accuracy = float(
            (predictions["candidate"][selected] == query_labels[selected]).mean()
            * 100.0
        )
        best_accuracy = float(
            (best_prediction[selected] == query_labels[selected]).mean() * 100.0
        )
        class_delta[class_index] = candidate_accuracy - best_accuracy
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "conflict_samples": int(selected.sum()),
                "candidate_set_coverage_pct": float(
                    candidate_coverage[selected].mean() * 100.0
                ),
                "candidate_accuracy_pct": candidate_accuracy,
                "best_baseline_name": best_baseline_name,
                "best_baseline_accuracy_pct": best_accuracy,
                "candidate_minus_best_baseline_pp": float(class_delta[class_index]),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(class_path, class_rows)

    car_delta = float(class_delta[3])
    truck_delta = float(class_delta[11])
    car_truck_mean_delta = float((car_delta + truck_delta) / 2.0)
    other_indices = [index for index in range(EXPECTED_CLASSES) if index not in (3, 11)]
    other_ten_mean_delta = float(class_delta[other_indices].mean())
    candidate_set_coverage_pct = float(candidate_coverage.mean() * 100.0)
    minimum_class_coverage_pct = float(
        min(row["candidate_set_coverage_pct"] for row in class_rows)
    )
    gate = evaluate_agreement_label_impact_gate(
        input_contract_valid=all(input_checks.values()),
        agreement_reference_accuracy_pct=agreement_reference_accuracy,
        minimum_split_decision_stability_pct=minimum_split_stability,
        candidate_set_coverage_pct=candidate_set_coverage_pct,
        minimum_class_candidate_coverage_pct=minimum_class_coverage_pct,
        comparisons=comparisons,
        best_baseline_name=best_baseline_name,
        car_delta_pp=car_delta,
        truck_delta_pp=truck_delta,
        car_truck_mean_delta_pp=car_truck_mean_delta,
        other_ten_mean_delta_pp=other_ten_mean_delta,
        max_class_mass_shift_pp=label_free_metrics["max_class_mass_shift_pp"],
    )
    summary = {
        "decision": gate["decision"],
        "method": "agreement_anchored_diagonal_fisher_label_impact",
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "agreement_reference_accuracy_pct": agreement_reference_accuracy,
            "candidate_set_coverage_pct": candidate_set_coverage_pct,
            "minimum_class_candidate_coverage_pct": minimum_class_coverage_pct,
            "comparisons": comparisons,
            "best_baseline_name": best_baseline_name,
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
            "PASS authorizes review of one exact trainable-parameter audit only. "
            "This audit never authorizes or starts proxy/full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(
        json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2)
    )
    print(f"Wrote label-free label-impact signal: {signal_path}")
    print(f"Locked label-impact signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
