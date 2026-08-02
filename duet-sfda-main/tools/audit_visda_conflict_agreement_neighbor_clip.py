#!/usr/bin/env python3
"""CPU-only audit of agreement-neighbor CLIP evidence for VisDA conflicts.

The label-free phase retrieves the five nearest DUET agreements in task-feature
space for every cycle-1 conflict, averages their CLIP distributions, and picks
within the query's task/CLIP top-2 union. Signals are locked before target
labels are parsed for oracle diagnostics. No image, model, checkpoint,
optimizer, backward pass, parameter update, or training is used.
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
sys.path.insert(0, str(REPO_ROOT))

from src.utils.agreement_neighbor_clip_audit import (  # noqa: E402
    agreement_neighbor_clip_posterior,
    evaluate_agreement_neighbor_clip_gate,
    select_from_candidate_set,
)
from src.utils.conflict_boundary import paired_accuracy_bootstrap_ci  # noqa: E402
from src.utils.spatial_causal_audit import topk_union_candidates  # noqa: E402


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_AGREEMENTS = 6_777
NEIGHBORS = 5
CLASS_NAMES = [
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
]
DEFAULT_BASE = Path(
    "output/uda/VISDA-C/TV/"
    "duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
)


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
            DEFAULT_BASE / "cycle2_conflict_memory_audit/"
            "visda_cycle2_conflict_memory_signal_lock.json"
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
        default=DEFAULT_BASE / "agreement_neighbor_clip_audit",
    )
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
        raise ValueError("Target label is outside the class range")
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


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    oracle = summary["oracle_metrics"]
    best = oracle["best_baseline_name"]
    comparison = oracle["comparisons"][best]
    lines = [
        "# VisDA Agreement-Neighbor CLIP Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "Signals were SHA256-locked before oracle labels were parsed. The",
        "audit loads no image, model, or checkpoint and performs no training.",
        "",
        "## Label-free rule",
        "",
        "For every cycle-1 task/CLIP conflict, retrieve K=5 nearest DUET",
        "agreements in task-feature cosine space. Average those references'",
        "CLIP distributions and choose within the query's task/CLIP top-2 union.",
        "",
        "## Oracle diagnostic",
        "",
        f"- Best matched baseline: `{best}`.",
        f"- Candidate gain: `{comparison['gain_pp']:.6f}` pp.",
        f"- 95% CI: `{comparison['paired_bootstrap_95_ci_pp']}`.",
        f"- Candidate-set coverage: `{oracle['candidate_set_coverage_pct']:.6f}%`.",
        f"- Neighbor label match: `{oracle['neighbor_label_match_pct']:.6f}%`.",
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
            "PASS authorizes design review for one matched proxy only. It does",
            "not authorize or start proxy/full VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (args.snapshot, args.source_lock, args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing agreement-neighbor input: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    class_names = [
        line.strip().replace("_", " ")
        for line in args.class_names.read_text().splitlines()
        if line.strip()
    ]
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
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        missing = required.difference(snapshot.files)
        if missing:
            raise ValueError(f"Snapshot is missing keys: {sorted(missing)}")
        cycle = int(np.asarray(snapshot["cycle"]).item())
        label_mask = np.asarray(snapshot["label_mask"], dtype=bool).copy()
        task_label = np.asarray(snapshot["source_label"], dtype=np.int64).copy()
        clip_label = np.asarray(snapshot["clip_label"], dtype=np.int64).copy()
        task_probability = np.asarray(snapshot["task_prob"], dtype=np.float64).copy()
        clip_probability = np.asarray(snapshot["clip_prob"], dtype=np.float64).copy()
        task_feature = np.asarray(snapshot["task_feature"], dtype=np.float32).copy()
        sample_index = np.asarray(snapshot["sample_index"], dtype=np.int64).copy()

    agreement = task_label == clip_label
    conflict = ~agreement
    neighbor = agreement_neighbor_clip_posterior(
        task_feature,
        clip_probability,
        agreement,
        conflict,
        neighbors=NEIGHBORS,
    )
    query = neighbor["query_index"]
    candidates = topk_union_candidates(
        task_probability[query], clip_probability[query], top_k=2
    )
    candidate = select_from_candidate_set(neighbor["posterior"], candidates)
    leave_farthest = select_from_candidate_set(
        neighbor["posterior_leave_farthest"], candidates
    )
    arithmetic_probability = 0.5 * (task_probability + clip_probability)
    rms_probability = np.sqrt(0.5 * (task_probability**2 + clip_probability**2))
    rms_probability /= rms_probability.sum(axis=1, keepdims=True)
    arithmetic_prediction = arithmetic_probability.argmax(axis=1)
    rms_prediction = rms_probability.argmax(axis=1)
    task_confidence = task_probability[np.arange(EXPECTED_SAMPLES), task_label]
    clip_confidence = clip_probability[np.arange(EXPECTED_SAMPLES), clip_label]
    confidence_prediction = np.where(
        task_confidence >= clip_confidence, task_label, clip_label
    )
    decision_stability_pct = float(
        (candidate["prediction"] == leave_farthest["prediction"]).mean() * 100.0
    )
    full_candidate_prediction = arithmetic_prediction.copy()
    full_candidate_prediction[query] = candidate["prediction"]
    baseline_mass = (
        np.bincount(arithmetic_prediction, minlength=EXPECTED_CLASSES)
        / EXPECTED_SAMPLES
    )
    candidate_mass = (
        np.bincount(full_candidate_prediction, minlength=EXPECTED_CLASSES)
        / EXPECTED_SAMPLES
    )
    class_mass_shift_pp = (candidate_mass - baseline_mass) * 100.0

    input_checks = {
        "source_snapshot_matches_cycle2_signal_lock": (
            snapshot_sha256 == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
        ),
        "source_lock_declares_labels_after_manifest": bool(
            source_lock.get("labels_read_after_this_manifest")
        ),
        "cycle_is_one": cycle == 1,
        "expected_probability_shape": (
            task_probability.shape
            == clip_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "expected_feature_rows": task_feature.shape[0] == EXPECTED_SAMPLES,
        "probabilities_finite_and_normalized": bool(
            np.isfinite(task_probability).all()
            and np.isfinite(clip_probability).all()
            and np.allclose(task_probability.sum(1), 1.0, atol=1e-5)
            and np.allclose(clip_probability.sum(1), 1.0, atol=1e-5)
        ),
        "features_finite": bool(np.isfinite(task_feature).all()),
        "sample_indices_are_proxy_order": np.array_equal(
            sample_index, np.arange(EXPECTED_SAMPLES)
        ),
        "saved_predictions_match_probabilities": (
            np.array_equal(task_label, task_probability.argmax(1))
            and np.array_equal(clip_label, clip_probability.argmax(1))
        ),
        "duet_mask_equals_top1_agreement": np.array_equal(label_mask, agreement),
        "expected_agreement_count": int(agreement.sum()) == EXPECTED_AGREEMENTS,
        "all_references_are_agreements": bool(
            agreement[neighbor["neighbor_index"]].all()
        ),
        "all_queries_are_conflicts": bool(conflict[query].all()),
        "candidate_predictions_inside_top2_union": bool(
            (candidates == candidate["prediction"][:, None]).any(axis=1).all()
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Agreement-neighbor input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_agreement_neighbor_clip"
    signal_path = args.output_dir / f"{stem}_label_free.npz"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"
    np.savez_compressed(
        signal_path,
        query_index=query,
        reference_index=neighbor["reference_index"],
        neighbor_index=neighbor["neighbor_index"],
        neighbor_similarity=neighbor["neighbor_similarity"].astype(np.float32),
        neighbor_clip_posterior=neighbor["posterior"].astype(np.float32),
        candidate_set=candidates,
        candidate_prediction=candidate["prediction"],
        candidate_margin=candidate["margin"].astype(np.float32),
        leave_farthest_prediction=leave_farthest["prediction"],
        neighbor_clip_top1_consensus=(
            neighbor["neighbor_clip_top1_consensus"].astype(np.float32)
        ),
        fixed_task_prediction=task_label[query],
        fixed_clip_prediction=clip_label[query],
        confidence_prediction=confidence_prediction[query],
        arithmetic_prediction=arithmetic_prediction[query],
        rms_prediction=rms_prediction[query],
    )
    lock = {
        "phase": "LABEL_FREE_AGREEMENT_NEIGHBOR_CLIP_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "query": "cycle1 task/CLIP top1 conflicts",
            "reference_pool": "cycle1 task/CLIP top1 agreements",
            "neighbor_space": "L2-normalized task bottleneck feature",
            "neighbors": NEIGHBORS,
            "neighbor_setting_source": "ViLAaD VisDA-C public K=5 setting",
            "neighbor_teacher": "mean frozen CLIP probability of K references",
            "candidate_set": "query task top2 union query CLIP top2",
            "selection": "maximum neighbor-teacher probability within candidate set",
            "fitted_parameters": False,
            "target_label_thresholds": False,
            "training_change_in_this_audit": False,
        },
        "predeclared_gate": {
            "min_leave_farthest_decision_stability_pct": 90.0,
            "min_candidate_set_coverage_pct": 90.0,
            "min_per_class_candidate_set_coverage_pct": 85.0,
            "min_neighbor_oracle_label_match_pct": 60.0,
            "min_accuracy_gain_vs_best_baseline_pp": 1.0,
            "best_baseline_paired_ci_lower": "> 0",
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
            "max_class_mass_shift_pp": 1.0,
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": {
            "samples": EXPECTED_SAMPLES,
            "agreements": int(agreement.sum()),
            "conflicts": int(conflict.sum()),
            "neighbors": NEIGHBORS,
            "mean_neighbor_cosine": float(neighbor["neighbor_similarity"].mean()),
            "mean_neighbor_clip_top1_consensus": float(
                neighbor["neighbor_clip_top1_consensus"].mean()
            ),
            "leave_farthest_decision_stability_pct": decision_stability_pct,
            "class_mass_shift_pp": {
                name: float(class_mass_shift_pp[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "max_class_mass_shift_pp": float(np.abs(class_mass_shift_pp).max()),
        },
        "inputs": {
            "pre_cycle1_snapshot": {
                "path": str(args.snapshot),
                "sha256": snapshot_sha256,
            },
            "cycle2_signal_lock": {
                "path": str(args.source_lock),
                "sha256": _sha256(args.source_lock),
            },
            "target_list_opaque_sha256": _sha256(args.target_list),
            "class_names_sha256": _sha256(args.class_names),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/agreement_neighbor_clip_audit.py": _sha256(
                REPO_ROOT / "src/utils/agreement_neighbor_clip_audit.py"
            ),
            "tools/audit_visda_conflict_agreement_neighbor_clip.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    target_hash_matches = (
        _sha256(args.target_list) == lock["inputs"]["target_list_opaque_sha256"]
    )
    target_labels = _parse_labels_after_lock(args.target_list)
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        embedded_labels = np.asarray(snapshot["target_label"], dtype=np.int64).copy()
    labels_match_snapshot = np.array_equal(target_labels[sample_index], embedded_labels)
    labels = target_labels[sample_index[query]]

    baselines = {
        "fixed_task": task_label[query],
        "fixed_clip": clip_label[query],
        "confidence_choice": confidence_prediction[query],
        "arithmetic": arithmetic_prediction[query],
        "rms": rms_prediction[query],
    }
    comparisons = {
        name: _comparison(
            candidate["prediction"], baseline, labels, seed=2_020 + offset
        )
        for offset, (name, baseline) in enumerate(baselines.items())
    }
    best_baseline_name = max(
        comparisons,
        key=lambda name: comparisons[name]["baseline_accuracy_pct"],
    )
    best_baseline = baselines[best_baseline_name]
    candidate_coverage = (candidates == labels[:, None]).any(axis=1)
    neighbor_label_match = (
        target_labels[sample_index[neighbor["neighbor_index"]]] == labels[:, None]
    )

    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_index
        candidate_accuracy = float(
            (candidate["prediction"][mask] == labels[mask]).mean() * 100.0
        )
        baseline_accuracy = float((best_baseline[mask] == labels[mask]).mean() * 100.0)
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "conflict_samples": int(mask.sum()),
                "candidate_set_coverage_pct": float(
                    candidate_coverage[mask].mean() * 100.0
                ),
                "neighbor_label_match_pct": float(
                    neighbor_label_match[mask].mean() * 100.0
                ),
                "candidate_accuracy_pct": candidate_accuracy,
                "best_baseline_name": best_baseline_name,
                "best_baseline_accuracy_pct": baseline_accuracy,
                "candidate_minus_best_baseline_pp": (
                    candidate_accuracy - baseline_accuracy
                ),
            }
        )
    _write_csv(class_path, class_rows)

    oracle_rows = []
    for row, index in enumerate(query):
        oracle_rows.append(
            {
                "index": int(sample_index[index]),
                "label": int(labels[row]),
                "label_name": CLASS_NAMES[int(labels[row])],
                "task_top1": int(task_label[index]),
                "clip_top1": int(clip_label[index]),
                "candidate_prediction": int(candidate["prediction"][row]),
                "candidate_correct": bool(candidate["prediction"][row] == labels[row]),
                "candidate_set_covers_label": bool(candidate_coverage[row]),
                "neighbor_label_match_fraction": float(
                    neighbor_label_match[row].mean()
                ),
                "neighbor_clip_top1_consensus": float(
                    neighbor["neighbor_clip_top1_consensus"][row]
                ),
                "candidate_margin": float(candidate["margin"][row]),
                "leave_farthest_stable": bool(
                    candidate["prediction"][row] == leave_farthest["prediction"][row]
                ),
                "best_baseline_name": best_baseline_name,
                "best_baseline_prediction": int(best_baseline[row]),
                "candidate_net_correction": int(
                    candidate["prediction"][row] == labels[row]
                )
                - int(best_baseline[row] == labels[row]),
            }
        )
    _write_csv(oracle_path, oracle_rows)

    class_delta = {
        row["class"]: row["candidate_minus_best_baseline_pp"] for row in class_rows
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
    input_contract_valid = (
        all(input_checks.values()) and target_hash_matches and labels_match_snapshot
    )
    gate = evaluate_agreement_neighbor_clip_gate(
        input_contract_valid=input_contract_valid,
        neighbors=NEIGHBORS,
        decision_stability_pct=decision_stability_pct,
        candidate_set_coverage_pct=float(candidate_coverage.mean() * 100.0),
        minimum_class_candidate_coverage_pct=min(
            row["candidate_set_coverage_pct"] for row in class_rows
        ),
        neighbor_label_match_pct=float(neighbor_label_match.mean() * 100.0),
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
        "decision": gate["decision"],
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "candidate_contract": lock["candidate_contract"],
        "literature_provenance": {
            "paper": "ViLAaD: Enhancing Attracting and Dispersing SFDA with ViL",
            "url": "https://arxiv.org/pdf/2503.23529",
            "borrowed_information": (
                "task-feature nearest-neighbor CLIP predictions and VisDA K=5"
            ),
            "not_claimed": "the published ViLAaD or ViLAaD++ method",
        },
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
        "label_free_metrics": lock["label_free_metrics"],
        "oracle_metrics": {
            "candidate_set_coverage_pct": float(candidate_coverage.mean() * 100.0),
            "minimum_class_candidate_coverage_pct": min(
                row["candidate_set_coverage_pct"] for row in class_rows
            ),
            "neighbor_label_match_pct": float(neighbor_label_match.mean() * 100.0),
            "reference_agreement_accuracy_pct": float(
                (task_label[agreement] == target_labels[sample_index[agreement]]).mean()
                * 100.0
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
        "scope_limit": (
            "PASS authorizes design of one matched proxy only. This audit never "
            "authorizes or starts proxy/full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(
        json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2)
    )
    print(f"Wrote label-free neighbor signal: {signal_path}")
    print(f"Locked neighbor signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
