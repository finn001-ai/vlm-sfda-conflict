#!/usr/bin/env python3
"""CPU-only agreement-conditioned joint-evidence GMM audit for VisDA.

The label-free phase fits one uniform-prior diagonal Gaussian per pseudo class
on the joint centered-log task/CLIP predictions of cycle-1 agreements.  It then
chooses only within each conflict's task/CLIP top-2 union.  Target labels are
parsed strictly after the signal lock for oracle diagnostics.  No image, model,
checkpoint, forward, backward, optimizer, parameter update, or training is used.
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

from src.utils.agreement_gmm_audit import (  # noqa: E402
    diagonal_gaussian_log_likelihood,
    evaluate_agreement_gmm_gate,
    fit_diagonal_class_gaussians,
    joint_centered_log_probability,
    select_candidate_by_log_likelihood,
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
STEM = "visda_conflict_agreement_gmm"


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
        default=DEFAULT_BASE / "agreement_gmm_audit",
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
    candidate_correct = candidate == labels
    baseline_correct = baseline == labels
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


def _fit_and_score(
    evidence: np.ndarray,
    pseudo_label: np.ndarray,
    reference_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    model = fit_diagonal_class_gaussians(
        evidence,
        pseudo_label,
        reference_mask,
        class_count=EXPECTED_CLASSES,
    )
    score = diagonal_gaussian_log_likelihood(
        evidence, model["mean"], model["variance"]
    )
    return model, score


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    oracle = summary["oracle_diagnostic"]
    best = oracle["best_baseline_name"]
    comparison = oracle["comparisons"][best]
    lines = [
        "# VisDA Agreement-Conditioned Joint-Evidence Audit",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Agreement reference cross-fit pseudo accuracy | "
            f"`{summary['label_free_metrics']['reference_crossfit_accuracy_pct']:.6f}%` "
            "| Label-free lock |"
        ),
        (
            "| Minimum within-class alternating-split decision stability | "
            f"`{summary['label_free_metrics']['minimum_split_decision_stability_pct']:.6f}%` "
            "| Label-free lock |"
        ),
        (
            f"| Conflict gain vs best matched baseline `{best}` | "
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
        "Fit one uniform-prior diagonal Gaussian per pseudo class on the joint",
        "centered-log task/CLIP predictions of cycle-1 DUET agreements. For each",
        "task/CLIP conflict, select the maximum-likelihood class only within the",
        "task-top2 union CLIP-top2 candidate set. No class-specific route, fitted",
        "temperature, learned threshold, or target label enters this rule.",
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
        args.snapshot,
        args.source_lock,
        args.target_list,
        args.class_names,
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing agreement-GMM input: {input_path}")
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
        task_probability = np.asarray(
            snapshot["task_prob"], dtype=np.float64
        ).copy()
        clip_probability = np.asarray(
            snapshot["clip_prob"], dtype=np.float64
        ).copy()
        sample_index = np.asarray(snapshot["sample_index"], dtype=np.int64).copy()

    agreement = task_label == clip_label
    conflict = ~agreement
    evidence = joint_centered_log_probability(task_probability, clip_probability)
    full_model, full_score = _fit_and_score(evidence, task_label, agreement)
    query = np.flatnonzero(conflict)
    candidates = topk_union_candidates(
        task_probability[query], clip_probability[query], top_k=2
    )
    candidate = select_candidate_by_log_likelihood(full_score[query], candidates)

    first_reference, second_reference = stratified_alternating_reference_masks(
        task_label,
        agreement,
        sample_index,
        class_count=EXPECTED_CLASSES,
    )
    first_model, first_score = _fit_and_score(
        evidence, task_label, first_reference
    )
    second_model, second_score = _fit_and_score(
        evidence, task_label, second_reference
    )
    first_candidate = select_candidate_by_log_likelihood(
        first_score[query], candidates
    )
    second_candidate = select_candidate_by_log_likelihood(
        second_score[query], candidates
    )
    split_stability = {
        "first_alternating_half_fit_pct": float(
            (first_candidate["prediction"] == candidate["prediction"]).mean()
            * 100.0
        ),
        "second_alternating_half_fit_pct": float(
            (second_candidate["prediction"] == candidate["prediction"]).mean()
            * 100.0
        ),
    }
    minimum_split_stability = float(min(split_stability.values()))
    crossfit_correct = np.zeros(EXPECTED_SAMPLES, dtype=bool)
    crossfit_correct[second_reference] = (
        first_score[second_reference].argmax(axis=1) == task_label[second_reference]
    )
    crossfit_correct[first_reference] = (
        second_score[first_reference].argmax(axis=1) == task_label[first_reference]
    )
    reference_crossfit_accuracy = float(
        crossfit_correct[agreement].mean() * 100.0
    )

    confidence_prediction = np.where(
        task_probability.max(axis=1) >= clip_probability.max(axis=1),
        task_label,
        clip_label,
    )
    arithmetic_prediction = (0.5 * (task_probability + clip_probability)).argmax(
        axis=1
    )
    rms_prediction = np.sqrt(
        0.5 * (task_probability**2 + clip_probability**2)
    ).argmax(axis=1)
    fixed_clip_full = clip_label.copy()
    candidate_full = fixed_clip_full.copy()
    candidate_full[query] = candidate["prediction"]
    class_mass_shift_pp = (
        np.bincount(candidate_full, minlength=EXPECTED_CLASSES)
        - np.bincount(fixed_clip_full, minlength=EXPECTED_CLASSES)
    ) / EXPECTED_SAMPLES * 100.0

    input_checks = {
        "snapshot_matches_cycle_memory_lock": (
            snapshot_sha256
            == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
        ),
        "snapshot_is_pre_cycle1": cycle == 1,
        "probability_shapes": (
            task_probability.shape
            == clip_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "probabilities_finite_normalized": all(
            np.isfinite(value).all()
            and np.allclose(value.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5)
            for value in (task_probability, clip_probability)
        ),
        "saved_predictions_match_probabilities": (
            np.array_equal(task_label, task_probability.argmax(axis=1))
            and np.array_equal(clip_label, clip_probability.argmax(axis=1))
        ),
        "agreement_mask_matches_predictions": np.array_equal(
            agreement, task_label == clip_label
        ),
        "admitted_mask_matches_cycle1_agreement": np.array_equal(admitted, agreement),
        "expected_agreement_and_conflict_counts": (
            int(agreement.sum()) == EXPECTED_AGREEMENTS
            and int(conflict.sum()) == EXPECTED_CONFLICTS
        ),
        "sample_indices_are_proxy_order": np.array_equal(
            sample_index, np.arange(EXPECTED_SAMPLES)
        ),
        "every_class_has_alternating_split_references": bool(
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
        raise RuntimeError(f"Agreement-GMM input contract failed: {failed}")

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
        candidate_margin=candidate["margin"].astype(np.float32),
        full_reference_count=full_model["reference_count"],
        full_gmm_mean=full_model["mean"].astype(np.float32),
        full_gmm_variance=full_model["variance"].astype(np.float32),
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
        "evidence_dimension": int(evidence.shape[1]),
        "reference_count_by_pseudo_class": full_model["reference_count"].tolist(),
        "reference_crossfit_accuracy_pct": reference_crossfit_accuracy,
        "stratified_alternating_split_decision_stability_pct": split_stability,
        "minimum_split_decision_stability_pct": minimum_split_stability,
        "class_mass_shift_pp": {
            name: float(class_mass_shift_pp[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "max_class_mass_shift_pp": float(np.abs(class_mass_shift_pp).max()),
    }
    lock = {
        "phase": "LABEL_FREE_VISDA_AGREEMENT_JOINT_EVIDENCE_GMM_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "reference_pool": "cycle1_task_clip_top1_agreements",
            "reference_pseudo_label": "shared_task_clip_top1",
            "evidence": "concatenated_task_and_clip_centered_log_probabilities",
            "density": "uniform_prior_per_class_diagonal_Gaussian",
            "variance_floor": "pooled_variance_times_sqrt_float64_epsilon",
            "query": "cycle1_task_clip_top1_conflicts",
            "candidate_set": "task_top2_union_clip_top2",
            "selection": "maximum_Gaussian_log_likelihood_within_candidate_set",
            "class_specific_route": False,
            "fitted_temperature": False,
            "fitted_thresholds": False,
            "target_label_thresholds": False,
            "training_change_in_this_audit": False,
        },
        "predeclared_gate": {
            "min_reference_crossfit_accuracy_pct": 90.0,
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
            "paper": (
                "Memory-Efficient Pseudo-Labeling for Online Source-Free "
                "Universal Domain Adaptation using a Gaussian Mixture Model"
            ),
            "venue": "WACV 2025",
            "paper_url": (
                "https://openaccess.thecvf.com/content/WACV2025/html/"
                "Schlachter_Memory-Efficient_Pseudo-Labeling_for_Online_"
                "Source-Free_Universal_Domain_Adaptation_using_a_WACV_2025_paper.html"
            ),
            "official_code": "https://github.com/pascalschlachter/GMM",
            "borrowed_information": "target_class_conditional_Gaussian_likelihood",
            "not_claimed": "the published online universal-SFDA method",
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
            "src/utils/agreement_gmm_audit.py": _sha256(
                REPO_ROOT / "src/utils/agreement_gmm_audit.py"
            ),
            "tools/audit_visda_conflict_agreement_gmm.py": _sha256(
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
    labels_match_snapshot = np.array_equal(labels[sample_index], embedded_labels)
    labels = labels[sample_index]
    query_labels = labels[query]

    baselines = {
        "fixed_task": task_label[query],
        "fixed_clip": clip_label[query],
        "confidence_choice": confidence_prediction[query],
        "arithmetic": arithmetic_prediction[query],
        "rms": rms_prediction[query],
    }
    comparisons = {
        name: _comparison(
            candidate["prediction"], baseline, query_labels, seed=2_020 + offset
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
        mask = query_labels == class_index
        candidate_accuracy = float(
            (candidate["prediction"][mask] == query_labels[mask]).mean() * 100.0
        )
        baseline_accuracy = float(
            (best_baseline[mask] == query_labels[mask]).mean() * 100.0
        )
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "conflict_samples": int(mask.sum()),
                "agreement_reference_samples": int(
                    (agreement & (task_label == class_index)).sum()
                ),
                "candidate_set_coverage_pct": float(
                    candidate_coverage[mask].mean() * 100.0
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
                "proxy_index": int(sample_index[index]),
                "oracle_target_label": int(query_labels[row]),
                "task_top1": int(task_label[index]),
                "clip_top1": int(clip_label[index]),
                "candidate_prediction": int(candidate["prediction"][row]),
                "candidate_correct": bool(
                    candidate["prediction"][row] == query_labels[row]
                ),
                "candidate_set_covers_label": bool(candidate_coverage[row]),
                "candidate_margin": float(candidate["margin"][row]),
                "first_alternating_split_stable": bool(
                    first_candidate["prediction"][row]
                    == candidate["prediction"][row]
                ),
                "second_alternating_split_stable": bool(
                    second_candidate["prediction"][row]
                    == candidate["prediction"][row]
                ),
                "best_baseline_name": best_baseline_name,
                "best_baseline_prediction": int(best_baseline[row]),
                "candidate_net_correction": int(
                    candidate["prediction"][row] == query_labels[row]
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
        all(input_checks.values()) and target_hash_matches and labels_match_snapshot
    )
    gate = evaluate_agreement_gmm_gate(
        input_contract_valid=input_contract_valid,
        reference_crossfit_accuracy_pct=reference_crossfit_accuracy,
        minimum_split_decision_stability_pct=minimum_split_stability,
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
        max_class_mass_shift_pp=float(np.abs(class_mass_shift_pp).max()),
    )
    summary = {
        "dataset": "VisDA-C",
        "seed": 2020,
        "cycle": cycle,
        "decision": gate["decision"],
        "labels_used_only_after_signal_lock": True,
        "signal_lock_sha256": _sha256(lock_path),
        "candidate_contract": lock["candidate_contract"],
        "literature_provenance": lock["literature_provenance"],
        "input_contract": {
            "passed": input_contract_valid,
            "checks": {
                **input_checks,
                "target_list_hash_matches_after_lock": target_hash_matches,
                "target_labels_match_embedded_snapshot_after_lock": (
                    labels_match_snapshot
                ),
            },
        },
        "label_free_metrics": label_free_metrics,
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "agreement_reference_accuracy_pct": float(
                (task_label[agreement] == labels[agreement]).mean() * 100.0
            ),
            "candidate_set_coverage_pct": float(
                candidate_coverage.mean() * 100.0
            ),
            "minimum_class_candidate_coverage_pct": min(
                row["candidate_set_coverage_pct"] for row in class_rows
            ),
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
            "PASS authorizes one matched-proxy design review only. This audit "
            "never authorizes or starts proxy/full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote label-free GMM signal: {signal_path}")
    print(f"Locked GMM signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
