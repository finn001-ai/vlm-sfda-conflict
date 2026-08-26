#!/usr/bin/env python3
"""CPU-only output/feature audit for patch-selected CLIP-KL suppression.

The candidate changes one item only: on conflicts selected by the locked,
disjointly confirmed patch-to-CLS risk control, the CLIP-KL component is
removed while DUET consistency remains unchanged.  No task hard pseudo-label
is added.  The audit reuses the locked full-target probabilities and maps the
directions through the exact frozen source classifier on CPU.  Oracle labels
are read only after the new signal NPZ and manifest are SHA256-locked.

No image, ResNet, bottleneck, or CLIP model is loaded.  There is no forward,
backward, optimizer, parameter update, proxy run, or training.
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
from src.utils.patch_cls_kl_suppression_audit import (  # noqa: E402
    consistency_logit_descent,
    evaluate_patch_cls_kl_suppression_gate,
)
from src.utils.pcgrad_feature_jacobian_audit import (  # noqa: E402
    effective_weight_normalized_linear,
    map_joint_logit_descent_to_feature,
)


EXPECTED_FULL_SAMPLES = 55_388
EXPECTED_CLASSES = 12
EXPECTED_FEATURES = 512
BATCH_SIZE = 64
CON_WEIGHT = 0.2
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
FEATURE_BASE = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_feature_gravity_audit_seed2020/feature_gravity_audit"
)
HOLDOUT_BASE = FEATURE_BASE / "patch_cls_holdout_audit"
STEM = "visda_patch_cls_kl_suppression_impact"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-signal",
        type=Path,
        default=FEATURE_BASE / "visda_conflict_feature_gravity_signals.npz",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=FEATURE_BASE / "visda_conflict_feature_gravity_signal_lock.json",
    )
    parser.add_argument(
        "--holdout-signal",
        type=Path,
        default=HOLDOUT_BASE
        / "visda_conflict_patch_cls_risk_control_holdout_label_free.npz",
    )
    parser.add_argument(
        "--holdout-lock",
        type=Path,
        default=HOLDOUT_BASE
        / "visda_conflict_patch_cls_risk_control_holdout_signal_lock.json",
    )
    parser.add_argument(
        "--holdout-oracle",
        type=Path,
        default=HOLDOUT_BASE
        / "visda_conflict_patch_cls_risk_control_holdout_oracle_diagnostic.csv",
    )
    parser.add_argument(
        "--holdout-summary",
        type=Path,
        default=HOLDOUT_BASE
        / "visda_conflict_patch_cls_risk_control_holdout_summary.json",
    )
    parser.add_argument(
        "--source-classifier",
        type=Path,
        default=Path("source/uda/VISDA-C/T/source_C.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HOLDOUT_BASE / "kl_suppression_impact_audit",
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


def _read_selected_labels_after_lock(
    path: Path, selected_query_index: np.ndarray
) -> np.ndarray:
    labels_by_index: dict[int, int] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("oracle_usage") != "diagnostic_only_after_heldout_signal_lock":
                raise ValueError("Held-out oracle provenance is invalid")
            index = int(row["sample_index"])
            if index in labels_by_index:
                raise ValueError("Held-out oracle contains a duplicate index")
            labels_by_index[index] = int(row["oracle_target_label"])
    missing = [
        int(index)
        for index in selected_query_index
        if int(index) not in labels_by_index
    ]
    if missing:
        raise ValueError(f"Held-out oracle is missing selected index {missing[0]}")
    labels = np.asarray(
        [labels_by_index[int(index)] for index in selected_query_index], dtype=np.int64
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


def _negative_burden(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.minimum(array, 0.0).mean())


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    oracle = summary["oracle_diagnostic"]
    output = oracle["output_alignment"]["first_order"]
    feature = oracle["feature_alignment"]["first_order"]
    lines = [
        "# VisDA Patch-Selected CLIP-KL Suppression Impact Audit",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Selected conflicts | "
            f"`{summary['label_free_metrics']['selected_samples']}` "
            "| Held-out label-free lock |"
        ),
        (
            "| Output first-order delta | "
            f"`{output['mean_difference']:.9f}`; CI "
            f"`{output['paired_bootstrap_95_ci']}` "
            "| Oracle diagnostic after new lock |"
        ),
        (
            "| Frozen-head feature first-order delta | "
            f"`{feature['mean_difference']:.9f}`; CI "
            f"`{feature['paired_bootstrap_95_ci']}` "
            "| Oracle diagnostic after new lock |"
        ),
        (
            "| Class-macro feature first-order delta | "
            f"`{oracle['class_macro_feature_first_order_delta']:.9f}` "
            "| Oracle diagnostic after new lock |"
        ),
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The candidate only suppresses the CLIP-KL component on the already",
        "locked patch-selected conflicts. It does not add task hard labels or",
        "change consistency, admission masks, loss coefficients, or any other row.",
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
            "Even PASS authorizes one exact resident-parameter audit only. It",
            "does not authorize a proxy run or training.",
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
        args.holdout_signal,
        args.holdout_lock,
        args.holdout_oracle,
        args.holdout_summary,
        args.source_classifier,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_lock = json.loads(args.source_lock.read_text())
    holdout_lock = json.loads(args.holdout_lock.read_text())
    with np.load(args.source_signal, allow_pickle=False) as source:
        source_arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    with np.load(args.holdout_signal, allow_pickle=False) as holdout:
        holdout_arrays = {
            name: np.asarray(holdout[name]).copy() for name in holdout.files
        }
    selected_mask = np.asarray(holdout_arrays["selected"], dtype=bool)
    selected_position = np.flatnonzero(selected_mask)
    source_row = np.asarray(holdout_arrays["source_signal_row"], dtype=np.int64)[
        selected_mask
    ]
    query_index = np.asarray(holdout_arrays["query_index"], dtype=np.int64)[
        selected_mask
    ]
    task_candidate = np.asarray(holdout_arrays["task_candidate"], dtype=np.int64)[
        selected_mask
    ]
    clip_candidate = np.asarray(holdout_arrays["clip_candidate"], dtype=np.int64)[
        selected_mask
    ]
    source_index = np.asarray(source_arrays["index"], dtype=np.int64)[source_row]
    weak_probability = np.asarray(source_arrays["weak_prob"], dtype=np.float64)[
        source_row
    ]
    strong_probability = np.asarray(source_arrays["strong_prob"], dtype=np.float64)[
        source_row
    ]
    baseline_logit = np.asarray(source_arrays["current_descent"], dtype=np.float64)[
        source_row
    ]
    classifier_weight, classifier_keys = _load_classifier(args.source_classifier)

    row_batch_size = np.full(query_index.size, BATCH_SIZE, dtype=np.int64)
    last_batch_start = (EXPECTED_FULL_SAMPLES // BATCH_SIZE) * BATCH_SIZE
    last_batch_size = EXPECTED_FULL_SAMPLES - last_batch_start
    row_batch_size[query_index >= last_batch_start] = last_batch_size
    consistency_weak, consistency_strong = consistency_logit_descent(
        weak_probability,
        strong_probability,
        row_batch_size,
        weight=CON_WEIGHT,
    )
    class_count = weak_probability.shape[1]
    baseline_weak = baseline_logit[:, :class_count]
    baseline_strong = baseline_logit[:, class_count:]
    recovered_clip_descent = baseline_weak - consistency_weak
    candidate_logit = np.concatenate((consistency_weak, baseline_strong), axis=1)
    baseline_feature = map_joint_logit_descent_to_feature(
        baseline_logit, classifier_weight
    )
    candidate_feature = map_joint_logit_descent_to_feature(
        candidate_logit, classifier_weight
    )
    strong_replay_error = float(np.max(np.abs(consistency_strong - baseline_strong)))
    input_checks = {
        "source_lock_label_free": (
            source_lock.get("phase") == "LABEL_FREE_SIGNAL_LOCK"
            and source_lock.get("contains_target_labels") is False
        ),
        "source_signal_hash_matches": (
            _sha256(args.source_signal)
            == source_lock.get("signal_npz", {}).get("sha256")
        ),
        "holdout_lock_label_free": (
            holdout_lock.get("phase")
            == "LABEL_FREE_VISDA_PATCH_CLS_RISK_CONTROL_HOLDOUT_LOCK"
            and holdout_lock.get("contains_target_labels") is False
        ),
        "holdout_signal_hash_matches": (
            _sha256(args.holdout_signal)
            == holdout_lock.get("signal_npz", {}).get("sha256")
        ),
        "selected_count_matches_lock": (
            int(selected_mask.sum())
            == int(
                holdout_lock.get("label_free_metrics", {}).get("selected_samples", -1)
            )
        ),
        "source_row_mapping_exact": np.array_equal(source_index, query_index),
        "task_candidate_mapping_exact": np.array_equal(
            np.asarray(source_arrays["task_pred"], dtype=np.int64)[source_row],
            task_candidate,
        ),
        "clip_candidate_mapping_exact": np.array_equal(
            np.asarray(source_arrays["clip_pred"], dtype=np.int64)[source_row],
            clip_candidate,
        ),
        "selected_rows_are_conflicts": bool(np.all(task_candidate != clip_candidate)),
        "probability_and_descent_shapes": (
            weak_probability.shape
            == strong_probability.shape
            == (query_index.size, EXPECTED_CLASSES)
            and baseline_logit.shape == (query_index.size, 2 * EXPECTED_CLASSES)
        ),
        "label_free_values_finite": all(
            np.isfinite(value).all()
            for value in (
                weak_probability,
                strong_probability,
                baseline_logit,
                recovered_clip_descent,
                candidate_logit,
                baseline_feature,
                candidate_feature,
            )
        ),
        "recovered_clip_descent_sums_to_zero": np.allclose(
            recovered_clip_descent.sum(axis=1), 0.0, atol=1e-10, rtol=1e-10
        ),
        "candidate_changes_only_selected_locked_rows": selected_position.size
        == query_index.size,
        "source_classifier_shape": classifier_weight.shape
        == (EXPECTED_CLASSES, EXPECTED_FEATURES),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Patch KL-suppression input contract failed: {failed}")

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
        source_signal_row=source_row,
        task_candidate=task_candidate,
        clip_candidate=clip_candidate,
        weak_probability=weak_probability.astype(np.float32),
        strong_probability=strong_probability.astype(np.float32),
        recovered_clip_descent=recovered_clip_descent.astype(np.float32),
        baseline_logit_descent=baseline_logit.astype(np.float32),
        candidate_logit_descent=candidate_logit.astype(np.float32),
        baseline_feature_descent=baseline_feature.astype(np.float32),
        candidate_feature_descent=candidate_feature.astype(np.float32),
        row_batch_size=row_batch_size,
    )
    baseline_norm = np.linalg.norm(baseline_feature, axis=1)
    candidate_norm = np.linalg.norm(candidate_feature, axis=1)
    nonzero_baseline = baseline_norm > 1e-15
    mean_norm_ratio = float(
        np.mean(candidate_norm[nonzero_baseline] / baseline_norm[nonzero_baseline])
    )
    label_free_metrics = {
        "selected_samples": int(query_index.size),
        "selected_conflict_coverage_pct": float(selected_mask.mean() * 100.0),
        "consistency_strong_replay_max_abs_error": strong_replay_error,
        "recovered_clip_descent_mean_norm": float(
            np.linalg.norm(recovered_clip_descent, axis=1).mean()
        ),
        "baseline_feature_mean_norm": float(baseline_norm.mean()),
        "candidate_feature_mean_norm": float(candidate_norm.mean()),
        "candidate_to_baseline_feature_mean_norm_ratio": mean_norm_ratio,
    }
    lock = {
        "phase": "LABEL_FREE_VISDA_PATCH_CLS_KL_SUPPRESSION_IMPACT_LOCK",
        "contains_target_labels": False,
        "oracle_artifacts_not_read_before_lock": True,
        "oracle_labels_read_after_this_manifest": True,
        "candidate": "suppress_clip_kl_only_on_locked_patch_risk_selected_conflicts",
        "candidate_contract": {
            "selected_rows": "unchanged_disjoint_heldout_patch_risk_control_mask",
            "removed_component": "CLIP_KL_logit_descent",
            "retained_component": "DUET_weak_strong_consistency",
            "task_hard_pseudo_labels_added": False,
            "admission_mask_changed": False,
            "loss_coefficients_changed": False,
            "nonselected_rows_changed": False,
            "fitted_thresholds": False,
            "target_label_rule": False,
        },
        "predeclared_gate": {
            "selected_coverage_pct": [2.0, 10.0],
            "consistency_strong_replay_max_abs_error": "<= 5e-8",
            "output_and_feature_first_order_mean_and_ci_lower": "> 0",
            "output_and_feature_negative_burden": "not_worse",
            "feature_helpful_retention_pct": ">= 99",
            "feature_mean_norm_inflation": "<= 1.5x",
            "class_macro_feature_first_order_delta": "> 0",
            "heldout_accuracy_gain_vs_clip_pp": ">= 1",
            "heldout_accuracy_ci_lower": "> 0",
            "max_individual_car_truck_accuracy_regression_pp": 0.5,
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
            "holdout_signal": {
                "path": str(args.holdout_signal),
                "sha256": _sha256(args.holdout_signal),
            },
            "holdout_lock": {
                "path": str(args.holdout_lock),
                "sha256": _sha256(args.holdout_lock),
            },
            "opaque_holdout_oracle_sha256": _sha256(args.holdout_oracle),
            "opaque_holdout_summary_sha256": _sha256(args.holdout_summary),
            "source_classifier": {
                "path": str(args.source_classifier),
                "sha256": _sha256(args.source_classifier),
                "keys": classifier_keys,
            },
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/patch_cls_kl_suppression_audit.py": _sha256(
                REPO_ROOT / "src/utils/patch_cls_kl_suppression_audit.py"
            ),
            "src/utils/pcgrad_feature_jacobian_audit.py": _sha256(
                REPO_ROOT / "src/utils/pcgrad_feature_jacobian_audit.py"
            ),
            "tools/audit_visda_patch_cls_kl_suppression_impact.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Oracle phase: reveal held-out labels and the prior PASS only after lock.
    for path, expected_hash in (
        (args.holdout_oracle, lock["inputs"]["opaque_holdout_oracle_sha256"]),
        (args.holdout_summary, lock["inputs"]["opaque_holdout_summary_sha256"]),
    ):
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"Held-out oracle artifact changed after lock: {path}")
    labels = _read_selected_labels_after_lock(args.holdout_oracle, query_index)
    holdout_summary = json.loads(args.holdout_summary.read_text())
    heldout_passed = (
        holdout_summary.get("decision") == "PASS_HELDOUT_PATCH_CLS_RISK_CONTROL"
    )
    oracle_logit = np.concatenate(
        (
            oracle_ce_logit_descent(weak_probability, labels),
            oracle_ce_logit_descent(strong_probability, labels),
        ),
        axis=1,
    )
    oracle_feature = map_joint_logit_descent_to_feature(oracle_logit, classifier_weight)
    output_baseline_alignment = rowwise_oracle_alignment(baseline_logit, oracle_logit)
    output_candidate_alignment = rowwise_oracle_alignment(candidate_logit, oracle_logit)
    feature_baseline_alignment = rowwise_oracle_alignment(
        baseline_feature, oracle_feature
    )
    feature_candidate_alignment = rowwise_oracle_alignment(
        candidate_feature, oracle_feature
    )
    metrics = ("cosine", "oracle_unit_projection", "first_order")
    output_comparisons = {
        metric: _paired_metric(
            output_candidate_alignment,
            output_baseline_alignment,
            metric,
            repeats=args.bootstrap_repeats,
            seed=args.seed + offset,
        )
        for offset, metric in enumerate(metrics)
    }
    feature_comparisons = {
        metric: _paired_metric(
            feature_candidate_alignment,
            feature_baseline_alignment,
            metric,
            repeats=args.bootstrap_repeats,
            seed=args.seed + 10 + offset,
        )
        for offset, metric in enumerate(metrics)
    }
    output_baseline_first = output_baseline_alignment["first_order"]
    output_candidate_first = output_candidate_alignment["first_order"]
    feature_baseline_first = feature_baseline_alignment["first_order"]
    feature_candidate_first = feature_candidate_alignment["first_order"]
    feature_delta = feature_candidate_first - feature_baseline_first
    baseline_helpful = np.maximum(feature_baseline_first, 0.0).sum()
    candidate_helpful = np.maximum(feature_candidate_first, 0.0).sum()
    helpful_retention = float(
        candidate_helpful / baseline_helpful * 100.0 if baseline_helpful > 0.0 else 0.0
    )

    class_rows: list[dict[str, Any]] = []
    class_mean_delta = np.empty(EXPECTED_CLASSES, dtype=np.float64)
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_index
        class_mean_delta[class_index] = float(feature_delta[mask].mean())
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "selected_samples": int(mask.sum()),
                "baseline_mean_feature_first_order": float(
                    feature_baseline_first[mask].mean()
                ),
                "candidate_mean_feature_first_order": float(
                    feature_candidate_first[mask].mean()
                ),
                "candidate_minus_baseline_mean_feature_first_order": float(
                    class_mean_delta[class_index]
                ),
                "oracle_usage": "diagnostic_only_after_kl_suppression_signal_lock",
            }
        )
    _write_csv(class_path, class_rows)
    oracle_rows: list[dict[str, Any]] = []
    for position, index in enumerate(query_index):
        oracle_rows.append(
            {
                "sample_index": int(index),
                "oracle_target_label": int(labels[position]),
                "task_candidate": int(task_candidate[position]),
                "clip_candidate": int(clip_candidate[position]),
                "baseline_output_first_order": float(output_baseline_first[position]),
                "candidate_output_first_order": float(output_candidate_first[position]),
                "baseline_feature_first_order": float(feature_baseline_first[position]),
                "candidate_feature_first_order": float(
                    feature_candidate_first[position]
                ),
                "candidate_minus_baseline_feature_first_order": float(
                    feature_delta[position]
                ),
                "oracle_usage": "diagnostic_only_after_kl_suppression_signal_lock",
            }
        )
    _write_csv(oracle_path, oracle_rows)

    heldout_clip = holdout_summary["oracle_diagnostic"]["comparisons"]["fixed_clip"]
    car_delta = float(holdout_summary["oracle_diagnostic"]["car_delta_pp"])
    truck_delta = float(holdout_summary["oracle_diagnostic"]["truck_delta_pp"])
    gate = evaluate_patch_cls_kl_suppression_gate(
        input_contract_valid=all(input_checks.values()),
        heldout_selector_passed=heldout_passed,
        selected_coverage_pct=label_free_metrics["selected_conflict_coverage_pct"],
        strong_replay_max_abs_error=strong_replay_error,
        output_first_order=output_comparisons["first_order"],
        feature_first_order=feature_comparisons["first_order"],
        output_negative_burden_baseline=_negative_burden(output_baseline_first),
        output_negative_burden_candidate=_negative_burden(output_candidate_first),
        feature_negative_burden_baseline=_negative_burden(feature_baseline_first),
        feature_negative_burden_candidate=_negative_burden(feature_candidate_first),
        feature_helpful_retention_pct=helpful_retention,
        feature_mean_norm_ratio=mean_norm_ratio,
        class_macro_feature_first_order_delta=float(class_mean_delta.mean()),
        heldout_accuracy_gain_vs_clip_pp=float(heldout_clip["gain_pp"]),
        heldout_accuracy_ci_lower_pp=float(
            heldout_clip["paired_bootstrap_95_ci_pp"][0]
        ),
        car_accuracy_delta_pp=car_delta,
        truck_accuracy_delta_pp=truck_delta,
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
            "heldout_selector_passed": heldout_passed,
            "output_alignment": output_comparisons,
            "feature_alignment": feature_comparisons,
            "output_negative_burden_baseline": _negative_burden(output_baseline_first),
            "output_negative_burden_candidate": _negative_burden(
                output_candidate_first
            ),
            "feature_negative_burden_baseline": _negative_burden(
                feature_baseline_first
            ),
            "feature_negative_burden_candidate": _negative_burden(
                feature_candidate_first
            ),
            "feature_helpful_retention_pct": helpful_retention,
            "class_macro_feature_first_order_delta": float(class_mean_delta.mean()),
            "classwise": class_rows,
            "heldout_accuracy_gain_vs_clip_pp": float(heldout_clip["gain_pp"]),
            "heldout_accuracy_gain_ci_lower_pp": float(
                heldout_clip["paired_bootstrap_95_ci_pp"][0]
            ),
            "car_accuracy_delta_pp": car_delta,
            "truck_accuracy_delta_pp": truck_delta,
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
            else "close patch-selected KL suppression without GPU or training"
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(
        json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2)
    )
    print(f"Wrote label-free KL-suppression signal: {signal_path}")
    print(f"Locked signal before oracle artifacts: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
