#!/usr/bin/env python3
"""CPU-only frozen-classifier Jacobian audit for the VisDA PCGrad signal.

The label-free phase loads only the locked cycle-2 probabilities, task
features, PCGrad output descents, and the small frozen source classifier.  It
replays the classifier and maps weak/strong logit descents through its exact
linear Jacobian.  Target labels and the prior oracle summary are read only
after the feature-space signals are SHA256-locked.

No target image, ResNet, bottleneck model, CLIP model, optimizer, backward
pass, parameter update, or training is used.  Passing only justifies design of
one exact-control resident parameter-gradient audit.
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
from src.utils.pcgrad_feature_jacobian_audit import (  # noqa: E402
    classifier_probability,
    effective_weight_normalized_linear,
    evaluate_feature_jacobian_gate,
    map_joint_logit_descent_to_feature,
)
from src.utils.support_conditioned_clip_audit import (  # noqa: E402
    negative_first_order_burden,
    normalize_probability_matrix,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_FEATURES = 512
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_BASE / "cycle2_conflict_memory_snapshots/pre_cycle02.npz",
    )
    parser.add_argument(
        "--pcgrad-signal",
        type=Path,
        default=DEFAULT_BASE / "conflict_pcgrad_audit/visda_conflict_pcgrad_label_free.npz",
    )
    parser.add_argument(
        "--pcgrad-lock",
        type=Path,
        default=DEFAULT_BASE / "conflict_pcgrad_audit/visda_conflict_pcgrad_signal_lock.json",
    )
    parser.add_argument(
        "--pcgrad-summary",
        type=Path,
        default=DEFAULT_BASE / "conflict_pcgrad_audit/visda_conflict_pcgrad_summary.json",
    )
    parser.add_argument(
        "--source-classifier",
        type=Path,
        default=Path("source/uda/VISDA-C/T/source_C.pt"),
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
        default=DEFAULT_BASE / "conflict_pcgrad_feature_jacobian_audit",
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


def _load_classifier(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("source_C checkpoint is not a state dictionary")
    state = {
        str(key).removeprefix("module."): value for key, value in state.items()
    }
    keys = sorted(state)
    if "fc.weight_g" in state and "fc.weight_v" in state:
        weight = effective_weight_normalized_linear(
            np.asarray(state["fc.weight_v"].detach().cpu()),
            np.asarray(state["fc.weight_g"].detach().cpu()),
        )
    elif "fc.weight" in state:
        weight = np.asarray(state["fc.weight"].detach().cpu(), dtype=np.float64)
    else:
        raise RuntimeError(f"Unsupported source_C weight keys: {keys}")
    if "fc.bias" not in state:
        raise RuntimeError("source_C checkpoint is missing fc.bias")
    bias = np.asarray(state["fc.bias"].detach().cpu(), dtype=np.float64)
    return weight, bias, keys


def _comparison(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    difference = candidate["first_order"][selected] - baseline["first_order"][selected]
    if difference.size == 0:
        return {
            "samples": 0,
            "mean_difference": 0.0,
            "paired_bootstrap_95_ci": [0.0, 0.0],
        }
    interval = paired_mean_bootstrap_ci(
        difference, repeats=repeats, seed=seed
    )
    return {
        "samples": int(difference.size),
        "mean_difference": float(difference.mean()),
        "paired_bootstrap_95_ci": list(interval),
    }


def _component_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dot = np.einsum("ij,ij->i", first, second)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.zeros(first.shape[0], dtype=np.float64)
    nonzero = denominator > 1e-15
    cosine[nonzero] = dot[nonzero] / denominator[nonzero]
    return np.clip(cosine, -1.0, 1.0)


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


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["oracle_metrics"]
    lines = [
        "# VisDA PCGrad Frozen-Classifier Feature-Jacobian Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The frozen source classifier was replayed from `source_C.pt`; no",
        "target image, ResNet, bottleneck model, CLIP model, optimizer,",
        "backward pass, or parameter update was used.",
        "",
        "## Label-free checks",
        "",
        f"- Max classifier probability replay error: `{summary['label_free_metrics']['max_probability_replay_error']:.9f}`.",
        f"- Feature component-conflict coverage: `{summary['label_free_metrics']['feature_component_conflict_coverage_pct']:.6f}%`.",
        f"- Output/feature conflict-mask agreement: `{summary['label_free_metrics']['output_feature_conflict_mask_agreement_pct']:.6f}%`.",
        "",
        "## Oracle diagnostic",
        "",
        f"- Overall feature first-order delta: `{metrics['overall_first_order']['mean_difference']:.9f}`.",
        f"- Overall 95% CI: `{metrics['overall_first_order']['paired_bootstrap_95_ci']}`.",
        f"- Active feature first-order delta: `{metrics['active_first_order']['mean_difference']:.9f}`.",
        f"- Active 95% CI: `{metrics['active_first_order']['paired_bootstrap_95_ci']}`.",
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
            "A pass still authorizes no GPU or training. It only justifies one",
            "exact arithmetic-DUET replay that measures resident parameter",
            "gradients without taking an optimizer step in the audit itself.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (
        args.snapshot,
        args.pcgrad_signal,
        args.pcgrad_lock,
        args.pcgrad_summary,
        args.source_classifier,
        args.target_list,
        args.class_names,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature-Jacobian input: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    class_names = tuple(
        line.strip().replace("_", " ")
        for line in args.class_names.read_text().splitlines()
        if line.strip()
    )
    if class_names != CLASS_NAMES:
        raise RuntimeError("VisDA class-name contract changed")
    pcgrad_lock = json.loads(args.pcgrad_lock.read_text())
    if _sha256(args.pcgrad_signal) != pcgrad_lock.get("signal_sha256"):
        raise RuntimeError("PCGrad signal hash does not match its label-free lock")
    if _sha256(args.snapshot) != pcgrad_lock.get("inputs", {}).get("snapshot_sha256"):
        raise RuntimeError("Cycle-2 snapshot hash does not match the PCGrad lock")

    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        required = {"sample_index", "task_prob", "task_feature", "target_label"}
        missing = required.difference(snapshot.files)
        if missing:
            raise RuntimeError(f"Snapshot is missing keys: {sorted(missing)}")
        sample_index = np.asarray(snapshot["sample_index"], dtype=np.int64).copy()
        full_task_probability = normalize_probability_matrix(
            snapshot["task_prob"], name="full_task_probability"
        )
        task_feature_storage_dtype = str(snapshot["task_feature"].dtype)
        task_feature = np.asarray(snapshot["task_feature"], dtype=np.float64).copy()
    with np.load(args.pcgrad_signal, allow_pickle=False) as signal:
        required = {
            "query_index",
            "weak_probability",
            "strong_probability",
            "consistency_joint",
            "clip_joint",
            "baseline_joint",
            "candidate_joint",
            "gradient_conflict",
        }
        missing = required.difference(signal.files)
        if missing:
            raise RuntimeError(f"PCGrad signal is missing keys: {sorted(missing)}")
        query = np.asarray(signal["query_index"], dtype=np.int64).copy()
        weak_probability = normalize_probability_matrix(
            signal["weak_probability"], name="weak_probability"
        )
        strong_probability = normalize_probability_matrix(
            signal["strong_probability"], name="strong_probability"
        )
        consistency_joint = np.asarray(
            signal["consistency_joint"], dtype=np.float64
        ).copy()
        clip_joint = np.asarray(signal["clip_joint"], dtype=np.float64).copy()
        baseline_joint = np.asarray(signal["baseline_joint"], dtype=np.float64).copy()
        candidate_joint = np.asarray(
            signal["candidate_joint"], dtype=np.float64
        ).copy()
        output_active = np.asarray(signal["gradient_conflict"], dtype=bool).copy()

    classifier_weight, classifier_bias, classifier_keys = _load_classifier(
        args.source_classifier
    )
    replay_probability = classifier_probability(
        task_feature, classifier_weight, classifier_bias
    )
    max_replay_error = float(
        np.max(np.abs(replay_probability - full_task_probability))
    )
    classifier_top1_reproduced = np.array_equal(
        replay_probability.argmax(1), full_task_probability.argmax(1)
    )
    feature_consistency = map_joint_logit_descent_to_feature(
        consistency_joint, classifier_weight
    )
    feature_clip = map_joint_logit_descent_to_feature(clip_joint, classifier_weight)
    feature_baseline = map_joint_logit_descent_to_feature(
        baseline_joint, classifier_weight
    )
    feature_candidate = map_joint_logit_descent_to_feature(
        candidate_joint, classifier_weight
    )
    feature_component_cosine = _component_cosine(feature_consistency, feature_clip)
    feature_component_conflict = feature_component_cosine < 0.0
    baseline_norm = np.linalg.norm(feature_baseline, axis=1)
    candidate_norm = np.linalg.norm(feature_candidate, axis=1)
    mean_norm_ratio = float(
        candidate_norm.mean() / baseline_norm.mean()
        if baseline_norm.mean() > 0.0
        else np.inf
    )
    input_checks = {
        "sample_indices_are_proxy_order": np.array_equal(
            sample_index, np.arange(EXPECTED_SAMPLES)
        ),
        "expected_feature_shape": task_feature.shape == (
            EXPECTED_SAMPLES,
            EXPECTED_FEATURES,
        ),
        "expected_classifier_shape": classifier_weight.shape == (
            EXPECTED_CLASSES,
            EXPECTED_FEATURES,
        ),
        "expected_classifier_bias_shape": classifier_bias.shape == (
            EXPECTED_CLASSES,
        ),
        "query_indices_unique_and_in_range": (
            np.unique(query).size == query.size
            and np.all(query >= 0)
            and np.all(query < EXPECTED_SAMPLES)
        ),
        "signal_probability_shapes_match": (
            weak_probability.shape
            == strong_probability.shape
            == (query.size, EXPECTED_CLASSES)
        ),
        "joint_logit_shapes_match": (
            consistency_joint.shape
            == clip_joint.shape
            == baseline_joint.shape
            == candidate_joint.shape
            == (query.size, 2 * EXPECTED_CLASSES)
        ),
        "baseline_is_component_sum": np.allclose(
            baseline_joint, consistency_joint + clip_joint, atol=2e-7, rtol=2e-7
        ),
        "query_probabilities_match_snapshot": np.allclose(
            weak_probability, full_task_probability[query], atol=2e-7, rtol=2e-7
        ),
        "classifier_state_keys_supported": (
            "fc.bias" in classifier_keys
            and (
                ("fc.weight_g" in classifier_keys and "fc.weight_v" in classifier_keys)
                or "fc.weight" in classifier_keys
            )
        ),
        "feature_descents_finite": all(
            np.isfinite(value).all()
            for value in (
                feature_consistency,
                feature_clip,
                feature_baseline,
                feature_candidate,
            )
        ),
    }
    input_checks = {name: bool(passed) for name, passed in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Feature-Jacobian input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_pcgrad_feature_jacobian"
    signal_path = args.output_dir / f"{stem}_label_free.npz"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"
    np.savez_compressed(
        signal_path,
        query_index=query,
        feature_consistency=feature_consistency.astype(np.float32),
        feature_clip=feature_clip.astype(np.float32),
        feature_baseline=feature_baseline.astype(np.float32),
        feature_candidate=feature_candidate.astype(np.float32),
        feature_component_cosine=feature_component_cosine.astype(np.float32),
        feature_component_conflict=feature_component_conflict,
        output_gradient_conflict=output_active,
    )
    label_free_metrics = {
        "query_samples": int(query.size),
        "classifier_top1_reproduced": bool(classifier_top1_reproduced),
        "max_probability_replay_error": max_replay_error,
        "feature_component_conflict_samples": int(feature_component_conflict.sum()),
        "feature_component_conflict_coverage_pct": float(
            feature_component_conflict.mean() * 100.0
        ),
        "output_feature_conflict_mask_agreement_pct": float(
            np.mean(output_active == feature_component_conflict) * 100.0
        ),
        "output_active_feature_conflict_precision_pct": float(
            np.mean(feature_component_conflict[output_active]) * 100.0
        ),
        "feature_component_cosine_quantiles": {
            str(quantile): float(np.quantile(feature_component_cosine, quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "feature_baseline_mean_descent_norm": float(baseline_norm.mean()),
        "feature_candidate_mean_descent_norm": float(candidate_norm.mean()),
        "feature_candidate_to_baseline_mean_norm_ratio": mean_norm_ratio,
    }
    lock = {
        "phase": "LABEL_FREE_PCGRAD_FEATURE_JACOBIAN_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "labels_read_after_this_manifest": True,
        "candidate": "locked_output_pcgrad_mapped_through_frozen_source_classifier",
        "candidate_contract": {
            "output_pcgrad_rule_changed": False,
            "classifier": "frozen_source_C_weight_normalized_linear",
            "classifier_parameters_updated": False,
            "task_feature_dtype_in_snapshot": task_feature_storage_dtype,
            "weak_strong_feature_jacobian": "same_exact_frozen_linear_weight",
            "strong_features_required": False,
            "resnet_or_bottleneck_jacobian_included": False,
            "thresholds_fitted": False,
        },
        "provenance_limitation": (
            "The cycle-2 snapshot comes from the prior first-cycle "
            "support-conditioned run, not a pure arithmetic-DUET replay."
        ),
        "inputs": {
            "snapshot_path": str(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "pcgrad_signal_path": str(args.pcgrad_signal),
            "pcgrad_signal_sha256": _sha256(args.pcgrad_signal),
            "pcgrad_lock_path": str(args.pcgrad_lock),
            "pcgrad_lock_sha256": _sha256(args.pcgrad_lock),
            "source_classifier_path": str(args.source_classifier),
            "source_classifier_sha256": _sha256(args.source_classifier),
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "signal_path": str(signal_path),
        "signal_sha256": _sha256(signal_path),
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Oracle phase: only now read the earlier oracle summary and target labels.
    prior_summary = json.loads(args.pcgrad_summary.read_text())
    prior_output_gate_passed = (
        prior_summary.get("decision") == "NEEDS_PARAMETER_AUDIT"
        and prior_summary.get("signal_lock_sha256") == _sha256(args.pcgrad_lock)
    )
    labels_from_list = _parse_labels_after_lock(args.target_list)
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        labels_from_snapshot = np.asarray(
            snapshot["target_label"], dtype=np.int64
        ).copy()
    if not np.array_equal(labels_from_list, labels_from_snapshot):
        raise RuntimeError("Target-list and snapshot oracle labels disagree")
    labels = labels_from_list[query]
    oracle_logit = np.concatenate(
        (
            oracle_ce_logit_descent(weak_probability, labels),
            oracle_ce_logit_descent(strong_probability, labels),
        ),
        axis=1,
    )
    oracle_feature = map_joint_logit_descent_to_feature(
        oracle_logit, classifier_weight
    )
    baseline_alignment = rowwise_oracle_alignment(feature_baseline, oracle_feature)
    candidate_alignment = rowwise_oracle_alignment(feature_candidate, oracle_feature)
    all_rows = np.ones(query.size, dtype=bool)
    overall_comparison = _comparison(
        candidate_alignment,
        baseline_alignment,
        all_rows,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    active_comparison = _comparison(
        candidate_alignment,
        baseline_alignment,
        output_active,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 1,
    )
    delta = candidate_alignment["first_order"] - baseline_alignment["first_order"]
    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_index
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "samples": int(mask.sum()),
                "output_active_samples": int(output_active[mask].sum()),
                "feature_component_conflict_samples": int(
                    feature_component_conflict[mask].sum()
                ),
                "baseline_mean_first_order": float(
                    baseline_alignment["first_order"][mask].mean()
                ),
                "candidate_mean_first_order": float(
                    candidate_alignment["first_order"][mask].mean()
                ),
                "candidate_minus_baseline_mean_first_order": float(delta[mask].mean()),
                "baseline_negative_burden": negative_first_order_burden(
                    baseline_alignment["first_order"][mask]
                ),
                "candidate_negative_burden": negative_first_order_burden(
                    candidate_alignment["first_order"][mask]
                ),
            }
        )
    _write_csv(class_path, class_rows)
    oracle_rows = []
    for position, index in enumerate(query):
        label = int(labels[position])
        oracle_rows.append(
            {
                "index": int(index),
                "label": label,
                "label_name": CLASS_NAMES[label],
                "output_gradient_conflict": bool(output_active[position]),
                "feature_component_cosine": float(
                    feature_component_cosine[position]
                ),
                "feature_component_conflict": bool(
                    feature_component_conflict[position]
                ),
                "baseline_feature_first_order": float(
                    baseline_alignment["first_order"][position]
                ),
                "candidate_feature_first_order": float(
                    candidate_alignment["first_order"][position]
                ),
                "candidate_minus_baseline_feature_first_order": float(
                    delta[position]
                ),
            }
        )
    _write_csv(oracle_path, oracle_rows)

    baseline_negative_burden = negative_first_order_burden(
        baseline_alignment["first_order"]
    )
    candidate_negative_burden = negative_first_order_burden(
        candidate_alignment["first_order"]
    )
    baseline_helpful = np.maximum(baseline_alignment["first_order"], 0.0).sum()
    candidate_helpful = np.maximum(candidate_alignment["first_order"], 0.0).sum()
    helpful_retention = float(
        candidate_helpful / baseline_helpful * 100.0
        if baseline_helpful > 0.0
        else 0.0
    )
    hard_indices = {name: CLASS_NAMES.index(name) for name in ("car", "person", "truck")}
    group_delta = {
        name: float(delta[labels == class_index].mean())
        for name, class_index in hard_indices.items()
    }
    other_mask = ~np.isin(labels, list(hard_indices.values()))
    group_delta["other_nine"] = float(delta[other_mask].mean())
    gate = evaluate_feature_jacobian_gate(
        input_contract_valid=(all(input_checks.values()) and prior_output_gate_passed),
        classifier_top1_reproduced=classifier_top1_reproduced,
        max_probability_replay_error=max_replay_error,
        overall_first_order=overall_comparison,
        active_first_order=active_comparison,
        baseline_negative_burden=baseline_negative_burden,
        candidate_negative_burden=candidate_negative_burden,
        helpful_retention_pct=helpful_retention,
        candidate_to_baseline_mean_norm_ratio=mean_norm_ratio,
        group_first_order_delta=group_delta,
    )
    summary = {
        "dataset": "VISDA-C",
        "seed": args.seed,
        "decision": gate["decision"],
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "candidate": lock["candidate"],
        "candidate_contract": lock["candidate_contract"],
        "provenance_limitation": lock["provenance_limitation"],
        "input_contract": {
            "passed": all(input_checks.values()) and prior_output_gate_passed,
            "checks": {
                **input_checks,
                "prior_output_gate_passed": bool(prior_output_gate_passed),
            },
        },
        "label_free_metrics": label_free_metrics,
        "oracle_metrics": {
            "overall_first_order": overall_comparison,
            "active_first_order": active_comparison,
            "baseline_negative_burden": baseline_negative_burden,
            "candidate_negative_burden": candidate_negative_burden,
            "helpful_first_order_retention_pct": helpful_retention,
            "group_first_order_delta": group_delta,
            "classwise": class_rows,
        },
        "gate": gate,
        "training_authorized": False,
        "proxy_authorized": False,
        "gpu_authorized": False,
        "target_images_loaded": False,
        "resnet_or_bottleneck_loaded": False,
        "clip_loaded": False,
        "classifier_checkpoint_loaded": True,
        "model_forward_calls": 0,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "next": (
            "design one exact arithmetic-DUET resident parameter-gradient audit"
            if gate["decision"] == "NEEDS_EXACT_CONTROL_PARAMETER_AUDIT"
            else "close PCGrad route without GPU work"
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote label-free feature signal: {signal_path}")
    print(f"Locked feature signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
