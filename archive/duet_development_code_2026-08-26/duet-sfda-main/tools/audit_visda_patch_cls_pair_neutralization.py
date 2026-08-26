#!/usr/bin/env python3
"""Audit pairwise CLIP-target neutralization on locked patch rescues.

The source patch audit established a disjoint held-out task-rescue cohort, but
removing the complete CLIP-KL component also removed useful non-top-1 class
structure.  This candidate changes only the CLIP target probabilities of the
task and CLIP top-1 candidates: their total mass is split equally.  Every other
class probability, consistency descent, mask, and loss coefficient is fixed.

Only prior locked arrays and the frozen classifier head are loaded.  Oracle
labels are read after the new label-free NPZ and manifest are written and
hashed.  No image/model forward, backward, optimizer, update, or training runs.
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
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.candidate_set_gradient_audit import (  # noqa: E402
    oracle_ce_logit_descent,
    paired_mean_bootstrap_ci,
    rowwise_oracle_alignment,
)
from src.utils.patch_cls_pair_neutralization_audit import (  # noqa: E402
    evaluate_patch_pair_neutralization_gate,
    neutralize_candidate_pair,
)
from src.utils.pcgrad_feature_jacobian_audit import (  # noqa: E402
    effective_weight_normalized_linear,
    map_joint_logit_descent_to_feature,
)


EXPECTED_CLASSES = 12
EXPECTED_FEATURES = 512
EXPECTED_FULL_SAMPLES = 55_388
KL_WEIGHT = 0.4
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
SUPPRESSION_DIR = Path(
    "output/uda/VISDA-C/TV/plmatch_visda_feature_gravity_audit_seed2020/"
    "feature_gravity_audit/patch_cls_holdout_audit/kl_suppression_impact_audit"
)
STEM = "visda_patch_cls_pair_neutralization"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-signal",
        type=Path,
        default=SUPPRESSION_DIR
        / "visda_patch_cls_kl_suppression_impact_label_free.npz",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=SUPPRESSION_DIR
        / "visda_patch_cls_kl_suppression_impact_signal_lock.json",
    )
    parser.add_argument(
        "--source-oracle",
        type=Path,
        default=SUPPRESSION_DIR
        / "visda_patch_cls_kl_suppression_impact_oracle_diagnostic.csv",
    )
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=SUPPRESSION_DIR / "visda_patch_cls_kl_suppression_impact_summary.json",
    )
    parser.add_argument(
        "--source-classifier",
        type=Path,
        default=Path("source/uda/VISDA-C/T/source_C.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUPPRESSION_DIR / "pair_neutralization_audit",
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


def _load_classifier(path: Path) -> tuple[np.ndarray, list[str]]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("source classifier is not a state dictionary")
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    keys = sorted(state)
    if "fc.weight_g" in state and "fc.weight_v" in state:
        weight = effective_weight_normalized_linear(
            np.asarray(state["fc.weight_v"].detach().cpu()),
            np.asarray(state["fc.weight_g"].detach().cpu()),
        )
    elif "fc.weight" in state:
        weight = np.asarray(state["fc.weight"].detach().cpu(), dtype=np.float64)
    else:
        raise RuntimeError(f"Unsupported source classifier keys: {keys}")
    if weight.shape != (EXPECTED_CLASSES, EXPECTED_FEATURES):
        raise RuntimeError(f"Unexpected source classifier shape: {weight.shape}")
    return weight, keys


def _read_labels_after_lock(path: Path, expected_index: np.ndarray) -> np.ndarray:
    labels_by_index: dict[int, int] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("oracle_usage") != (
                "diagnostic_only_after_kl_suppression_signal_lock"
            ):
                raise ValueError("Source oracle provenance is invalid")
            index = int(row["sample_index"])
            if index in labels_by_index:
                raise ValueError("Source oracle contains a duplicate index")
            labels_by_index[index] = int(row["oracle_target_label"])
    if set(labels_by_index) != set(map(int, expected_index)):
        raise ValueError("Source oracle indices do not match the locked signal")
    labels = np.asarray(
        [labels_by_index[int(index)] for index in expected_index], dtype=np.int64
    )
    if np.any(labels < 0) or np.any(labels >= EXPECTED_CLASSES):
        raise ValueError("Oracle label outside class range")
    return labels


def _paired_metric(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    metric: str,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    difference = np.asarray(candidate[metric] - baseline[metric], dtype=np.float64)
    interval = paired_mean_bootstrap_ci(difference, repeats=repeats, seed=seed)
    return {
        "samples": int(difference.size),
        "baseline_mean": float(baseline[metric].mean()),
        "candidate_mean": float(candidate[metric].mean()),
        "mean_difference": float(difference.mean()),
        "paired_bootstrap_95_ci": list(interval),
    }


def _comparison_set(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    return {
        metric: _paired_metric(
            candidate, baseline, metric, repeats=repeats, seed=seed + offset
        )
        for offset, metric in enumerate(
            ("cosine", "oracle_unit_projection", "first_order")
        )
    }


def _negative_burden(values: np.ndarray) -> float:
    return float(np.minimum(np.asarray(values, dtype=np.float64), 0.0).mean())


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    oracle = summary["oracle_diagnostic"]
    versus_duet = oracle["versus_original_duet"]["feature"]["first_order"]
    versus_suppression = oracle["versus_full_kl_suppression"]["feature"]["first_order"]
    lines = [
        "# VisDA Patch Pair-Neutralization Impact Audit",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Locked selected conflicts | "
            f"`{summary['label_free_metrics']['selected_samples']}` "
            "| Prior label-free suppression signal |"
        ),
        (
            "| Feature first-order delta vs original DUET | "
            f"`{versus_duet['mean_difference']:.9f}`; CI "
            f"`{versus_duet['paired_bootstrap_95_ci']}` "
            "| Oracle diagnostic after new lock |"
        ),
        (
            "| Feature first-order delta vs full KL suppression | "
            f"`{versus_suppression['mean_difference']:.9f}`; CI "
            f"`{versus_suppression['paired_bootstrap_95_ci']}` "
            "| Oracle diagnostic after new lock |"
        ),
        (
            "| Maximum full-target class-mass shift | "
            f"`{summary['label_free_metrics']['max_full_target_class_mass_shift_pp']:.6f}` pp "
            "| Label-free target probabilities |"
        ),
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The candidate equalizes only the task/CLIP candidate pair in the CLIP",
        "soft target. It preserves pair mass and every non-pair probability; no",
        "hard task label, mask change, fitted threshold, or extra loss is added.",
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
            "Even PASS authorizes one exact no-update parameter audit only, not a",
            "proxy run or training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for required in (
        args.source_signal,
        args.source_lock,
        args.source_oracle,
        args.source_summary,
        args.source_classifier,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_lock = json.loads(args.source_lock.read_text())
    with np.load(args.source_signal, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    query_index = np.asarray(arrays["query_index"], dtype=np.int64)
    task_candidate = np.asarray(arrays["task_candidate"], dtype=np.int64)
    clip_candidate = np.asarray(arrays["clip_candidate"], dtype=np.int64)
    weak_probability = np.asarray(arrays["weak_probability"], dtype=np.float64)
    strong_probability = np.asarray(arrays["strong_probability"], dtype=np.float64)
    recovered_clip_descent = np.asarray(
        arrays["recovered_clip_descent"], dtype=np.float64
    )
    baseline_logit = np.asarray(arrays["baseline_logit_descent"], dtype=np.float64)
    suppression_logit = np.asarray(arrays["candidate_logit_descent"], dtype=np.float64)
    row_batch_size = np.asarray(arrays["row_batch_size"], dtype=np.float64)
    classifier_weight, classifier_keys = _load_classifier(args.source_classifier)
    class_count = weak_probability.shape[1]

    clip_target = weak_probability + (
        recovered_clip_descent * row_batch_size[:, None] / KL_WEIGHT
    )
    target_row_sum = clip_target.sum(axis=1, keepdims=True)
    clip_target = clip_target / target_row_sum
    replayed_clip_descent = (
        KL_WEIGHT / row_batch_size[:, None] * (clip_target - weak_probability)
    )
    target_replay_error = float(
        np.max(np.abs(replayed_clip_descent - recovered_clip_descent))
    )
    neutral_target, transferred_mass = neutralize_candidate_pair(
        clip_target, task_candidate, clip_candidate
    )
    row = np.arange(query_index.size)
    pair_mass_error = float(
        np.max(
            np.abs(
                neutral_target[row, task_candidate]
                + neutral_target[row, clip_candidate]
                - clip_target[row, task_candidate]
                - clip_target[row, clip_candidate]
            )
        )
    )
    pair_mask = np.zeros_like(clip_target, dtype=bool)
    pair_mask[row, task_candidate] = True
    pair_mask[row, clip_candidate] = True
    nonpair_error = float(
        np.max(np.abs(neutral_target[~pair_mask] - clip_target[~pair_mask]))
    )
    candidate_clip_descent = (
        KL_WEIGHT / row_batch_size[:, None] * (neutral_target - weak_probability)
    )
    consistency_logit = suppression_logit
    candidate_logit = consistency_logit.copy()
    candidate_logit[:, :class_count] += candidate_clip_descent
    baseline_feature = map_joint_logit_descent_to_feature(
        baseline_logit, classifier_weight
    )
    suppression_feature = map_joint_logit_descent_to_feature(
        suppression_logit, classifier_weight
    )
    candidate_feature = map_joint_logit_descent_to_feature(
        candidate_logit, classifier_weight
    )
    class_mass_shift = (
        (neutral_target - clip_target).sum(axis=0) / EXPECTED_FULL_SAMPLES * 100.0
    )

    input_checks = {
        "source_lock_is_label_free": (
            source_lock.get("phase")
            == "LABEL_FREE_VISDA_PATCH_CLS_KL_SUPPRESSION_IMPACT_LOCK"
            and source_lock.get("contains_target_labels") is False
        ),
        "source_signal_hash_matches_lock": (
            _sha256(args.source_signal)
            == source_lock.get("signal_npz", {}).get("sha256")
        ),
        "expected_shapes": (
            weak_probability.shape
            == strong_probability.shape
            == recovered_clip_descent.shape
            == (query_index.size, EXPECTED_CLASSES)
            and baseline_logit.shape
            == suppression_logit.shape
            == (query_index.size, 2 * EXPECTED_CLASSES)
            and row_batch_size.shape == (query_index.size,)
        ),
        "query_indices_unique": np.unique(query_index).size == query_index.size,
        "candidate_rows_are_conflicts": bool(np.all(task_candidate != clip_candidate)),
        "recovered_clip_target_is_probability": (
            np.isfinite(clip_target).all()
            and np.all(clip_target >= -5e-6)
            and np.allclose(clip_target.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8)
        ),
        "neutral_target_is_probability": (
            np.isfinite(neutral_target).all()
            and np.all(neutral_target >= 0.0)
            and np.allclose(neutral_target.sum(axis=1), 1.0, atol=1e-10, rtol=1e-10)
        ),
        "mass_moves_from_clip_to_task": bool(np.all(transferred_mass >= -1e-12)),
        "all_label_free_values_finite": all(
            np.isfinite(value).all()
            for value in (
                baseline_logit,
                suppression_logit,
                candidate_logit,
                baseline_feature,
                suppression_feature,
                candidate_feature,
                class_mass_shift,
            )
        ),
        "source_classifier_shape": classifier_weight.shape
        == (EXPECTED_CLASSES, EXPECTED_FEATURES),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Patch pair-neutralization input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    signal_path = args.output_dir / f"{STEM}_label_free.npz"
    lock_path = args.output_dir / f"{STEM}_signal_lock.json"
    oracle_path = args.output_dir / f"{STEM}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{STEM}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{STEM}_summary.json"
    markdown_path = args.output_dir / f"{STEM}_summary.md"
    np.savez_compressed(
        signal_path,
        query_index=query_index,
        task_candidate=task_candidate,
        clip_candidate=clip_candidate,
        recovered_clip_target=clip_target.astype(np.float32),
        neutralized_clip_target=neutral_target.astype(np.float32),
        transferred_mass=transferred_mass.astype(np.float32),
        baseline_logit_descent=baseline_logit.astype(np.float32),
        suppression_logit_descent=suppression_logit.astype(np.float32),
        candidate_logit_descent=candidate_logit.astype(np.float32),
        baseline_feature_descent=baseline_feature.astype(np.float32),
        suppression_feature_descent=suppression_feature.astype(np.float32),
        candidate_feature_descent=candidate_feature.astype(np.float32),
        class_mass_shift_pp=class_mass_shift.astype(np.float32),
        row_batch_size=row_batch_size.astype(np.int64),
    )
    baseline_norm = np.linalg.norm(baseline_feature, axis=1)
    candidate_norm = np.linalg.norm(candidate_feature, axis=1)
    nonzero = baseline_norm > 1e-15
    mean_norm_ratio = float(np.mean(candidate_norm[nonzero] / baseline_norm[nonzero]))
    selected_coverage = float(
        source_lock.get("label_free_metrics", {}).get(
            "selected_conflict_coverage_pct", np.nan
        )
    )
    label_free_metrics = {
        "selected_samples": int(query_index.size),
        "selected_conflict_coverage_pct": selected_coverage,
        "mean_transferred_clip_to_task_mass": float(transferred_mass.mean()),
        "minimum_transferred_clip_to_task_mass": float(transferred_mass.min()),
        "recovered_clip_target_replay_max_abs_error": target_replay_error,
        "nonpair_target_max_abs_error": nonpair_error,
        "candidate_pair_mass_max_abs_error": pair_mass_error,
        "max_full_target_class_mass_shift_pp": float(np.max(np.abs(class_mass_shift))),
        "candidate_to_baseline_feature_mean_norm_ratio": mean_norm_ratio,
    }
    lock = {
        "phase": "LABEL_FREE_VISDA_PATCH_CLS_PAIR_NEUTRALIZATION_LOCK",
        "contains_target_labels": False,
        "oracle_artifacts_not_read_before_lock": True,
        "candidate": "patch_selected_task_clip_pair_mass_neutralization",
        "candidate_contract": {
            "new_information": "locked_patch_to_cls_task_rescue_selector",
            "changed_target_entries": "task_and_clip_top1_only",
            "candidate_pair_total_mass_preserved": True,
            "noncandidate_probabilities_preserved": True,
            "task_hard_pseudo_labels_added": False,
            "admission_mask_changed": False,
            "consistency_changed": False,
            "loss_coefficients_changed": False,
            "fitted_thresholds": False,
            "target_label_rule": False,
        },
        "predeclared_gate": {
            "selected_coverage_pct": [2.0, 10.0],
            "target_replay_max_abs_error": "<=5e-6",
            "nonpair_probability_and_pair_mass_error": "<=1e-12",
            "output_and_feature_first_order_ci_lower_vs_duet": ">0",
            "output_and_feature_first_order_ci_lower_vs_suppression": ">0",
            "negative_burden": "not_worse_than_duet",
            "feature_helpful_retention_pct": ">=99",
            "feature_mean_norm_inflation": "<=1.5x",
            "max_full_target_class_mass_shift_pp": "<=1",
            "class_macro_car_person_truck_other9_feature_delta": "nonnegative",
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "inputs": {
            "source_signal": {
                "path": str(args.source_signal),
                "sha256": _sha256(args.source_signal),
            },
            "source_lock": {
                "path": str(args.source_lock),
                "sha256": _sha256(args.source_lock),
            },
            "opaque_source_oracle_sha256": _sha256(args.source_oracle),
            "opaque_source_summary_sha256": _sha256(args.source_summary),
            "source_classifier": {
                "path": str(args.source_classifier),
                "sha256": _sha256(args.source_classifier),
                "keys": classifier_keys,
            },
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/patch_cls_pair_neutralization_audit.py": _sha256(
                REPO_ROOT / "src/utils/patch_cls_pair_neutralization_audit.py"
            ),
            "tools/audit_visda_patch_cls_pair_neutralization.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Oracle phase begins only after the candidate signal and manifest are locked.
    for path, expected_hash in (
        (args.source_oracle, lock["inputs"]["opaque_source_oracle_sha256"]),
        (args.source_summary, lock["inputs"]["opaque_source_summary_sha256"]),
    ):
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"Oracle artifact changed after lock: {path}")
    labels = _read_labels_after_lock(args.source_oracle, query_index)
    source_summary = json.loads(args.source_summary.read_text())
    source_reject = source_summary.get("decision") == "REJECT"
    heldout_passed = bool(
        source_summary.get("oracle_diagnostic", {}).get(
            "heldout_selector_passed", False
        )
    )
    oracle_logit = np.concatenate(
        (
            oracle_ce_logit_descent(weak_probability, labels),
            oracle_ce_logit_descent(strong_probability, labels),
        ),
        axis=1,
    )
    oracle_feature = map_joint_logit_descent_to_feature(oracle_logit, classifier_weight)
    output_alignment = {
        "original_duet": rowwise_oracle_alignment(baseline_logit, oracle_logit),
        "full_kl_suppression": rowwise_oracle_alignment(
            suppression_logit, oracle_logit
        ),
        "pair_neutralization": rowwise_oracle_alignment(candidate_logit, oracle_logit),
    }
    feature_alignment = {
        "original_duet": rowwise_oracle_alignment(baseline_feature, oracle_feature),
        "full_kl_suppression": rowwise_oracle_alignment(
            suppression_feature, oracle_feature
        ),
        "pair_neutralization": rowwise_oracle_alignment(
            candidate_feature, oracle_feature
        ),
    }
    versus_duet = {
        "output": _comparison_set(
            output_alignment["pair_neutralization"],
            output_alignment["original_duet"],
            repeats=args.bootstrap_repeats,
            seed=args.seed,
        ),
        "feature": _comparison_set(
            feature_alignment["pair_neutralization"],
            feature_alignment["original_duet"],
            repeats=args.bootstrap_repeats,
            seed=args.seed + 10,
        ),
    }
    versus_suppression = {
        "output": _comparison_set(
            output_alignment["pair_neutralization"],
            output_alignment["full_kl_suppression"],
            repeats=args.bootstrap_repeats,
            seed=args.seed + 20,
        ),
        "feature": _comparison_set(
            feature_alignment["pair_neutralization"],
            feature_alignment["full_kl_suppression"],
            repeats=args.bootstrap_repeats,
            seed=args.seed + 30,
        ),
    }
    baseline_first = feature_alignment["original_duet"]["first_order"]
    candidate_first = feature_alignment["pair_neutralization"]["first_order"]
    feature_delta = candidate_first - baseline_first
    baseline_helpful = np.maximum(baseline_first, 0.0).sum()
    candidate_helpful = np.maximum(candidate_first, 0.0).sum()
    helpful_retention = float(
        candidate_helpful / baseline_helpful * 100.0 if baseline_helpful > 0.0 else 0.0
    )

    class_rows: list[dict[str, Any]] = []
    class_delta = np.empty(EXPECTED_CLASSES, dtype=np.float64)
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_index
        class_delta[class_index] = float(feature_delta[mask].mean())
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "selected_samples": int(mask.sum()),
                "baseline_mean_feature_first_order": float(baseline_first[mask].mean()),
                "candidate_mean_feature_first_order": float(
                    candidate_first[mask].mean()
                ),
                "candidate_minus_baseline_mean_feature_first_order": float(
                    class_delta[class_index]
                ),
                "oracle_usage": (
                    "diagnostic_only_after_pair_neutralization_signal_lock"
                ),
            }
        )
    _write_csv(class_path, class_rows)
    output_baseline_first = output_alignment["original_duet"]["first_order"]
    output_candidate_first = output_alignment["pair_neutralization"]["first_order"]
    oracle_rows = [
        {
            "sample_index": int(index),
            "oracle_target_label": int(labels[position]),
            "task_candidate": int(task_candidate[position]),
            "clip_candidate": int(clip_candidate[position]),
            "transferred_clip_to_task_mass": float(transferred_mass[position]),
            "baseline_output_first_order": float(output_baseline_first[position]),
            "candidate_output_first_order": float(output_candidate_first[position]),
            "baseline_feature_first_order": float(baseline_first[position]),
            "candidate_feature_first_order": float(candidate_first[position]),
            "candidate_minus_baseline_feature_first_order": float(
                feature_delta[position]
            ),
            "oracle_usage": "diagnostic_only_after_pair_neutralization_signal_lock",
        }
        for position, index in enumerate(query_index)
    ]
    _write_csv(oracle_path, oracle_rows)

    output_baseline_values = output_alignment["original_duet"]["first_order"]
    output_candidate_values = output_alignment["pair_neutralization"]["first_order"]
    other_nine = np.delete(class_delta, [3, 7, 11])
    gate = evaluate_patch_pair_neutralization_gate(
        input_contract_valid=all(input_checks.values()),
        source_suppression_reject_preserved=source_reject,
        heldout_selector_passed=heldout_passed,
        selected_coverage_pct=selected_coverage,
        target_replay_max_abs_error=target_replay_error,
        nonpair_target_max_abs_error=nonpair_error,
        pair_mass_max_abs_error=pair_mass_error,
        baseline_output_first_order=versus_duet["output"]["first_order"],
        baseline_feature_first_order=versus_duet["feature"]["first_order"],
        suppression_output_first_order=versus_suppression["output"]["first_order"],
        suppression_feature_first_order=versus_suppression["feature"]["first_order"],
        output_negative_burden_baseline=_negative_burden(output_baseline_values),
        output_negative_burden_candidate=_negative_burden(output_candidate_values),
        feature_negative_burden_baseline=_negative_burden(baseline_first),
        feature_negative_burden_candidate=_negative_burden(candidate_first),
        feature_helpful_retention_pct=helpful_retention,
        feature_mean_norm_ratio=mean_norm_ratio,
        max_full_target_class_mass_shift_pp=label_free_metrics[
            "max_full_target_class_mass_shift_pp"
        ],
        class_macro_feature_first_order_delta=float(class_delta.mean()),
        car_feature_first_order_delta=float(class_delta[3]),
        person_feature_first_order_delta=float(class_delta[7]),
        truck_feature_first_order_delta=float(class_delta[11]),
        other_nine_feature_first_order_delta=float(other_nine.mean()),
    )
    summary = {
        "decision": gate["decision"],
        "candidate": lock["candidate"],
        "candidate_contract": lock["candidate_contract"],
        "input_contract": {
            "passed": all(input_checks.values()),
            "checks": input_checks,
        },
        "label_free_metrics": label_free_metrics,
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "labels_read_only_after_signal_lock": True,
            "source_kl_suppression_reject_preserved": source_reject,
            "heldout_selector_passed": heldout_passed,
            "versus_original_duet": versus_duet,
            "versus_full_kl_suppression": versus_suppression,
            "output_negative_burden_baseline": _negative_burden(output_baseline_values),
            "output_negative_burden_candidate": _negative_burden(
                output_candidate_values
            ),
            "feature_negative_burden_baseline": _negative_burden(baseline_first),
            "feature_negative_burden_candidate": _negative_burden(candidate_first),
            "feature_helpful_retention_pct": helpful_retention,
            "class_macro_feature_first_order_delta": float(class_delta.mean()),
            "car_feature_first_order_delta": float(class_delta[3]),
            "person_feature_first_order_delta": float(class_delta[7]),
            "truck_feature_first_order_delta": float(class_delta[11]),
            "other_nine_feature_first_order_delta": float(other_nine.mean()),
            "classwise": class_rows,
        },
        "gate": gate,
        "outputs": {
            "label_free_signal": str(signal_path),
            "signal_lock": str(lock_path),
            "oracle_diagnostic": str(oracle_path),
            "classwise_oracle_diagnostic": str(class_path),
            "markdown": str(markdown_path),
        },
        "safety": {
            "target_images_loaded": False,
            "resnet_bottleneck_or_clip_loaded": False,
            "classifier_checkpoint_loaded_on_cpu": True,
            "model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_constructed": False,
            "parameter_updates": 0,
            "proxy_authorized": False,
            "training_authorized": False,
        },
        "next": (
            "design one exact no-update resident-parameter audit"
            if gate["decision"] == "NEEDS_EXACT_PARAMETER_AUDIT"
            else "close the complete patch-to-CLS branch without GPU work"
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(
        json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2)
    )
    print(f"Wrote label-free pair target: {signal_path}")
    print(f"Locked pair target before oracle artifacts: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
