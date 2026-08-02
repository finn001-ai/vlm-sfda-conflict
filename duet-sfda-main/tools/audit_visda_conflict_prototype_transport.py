#!/usr/bin/env python3
"""CPU-only VisDA source-prototype/CLIP transport preflight.

Phase 1 reads the locked pre-cycle-1 DUET snapshot without accessing its
target-label key.  On task/CLIP conflicts, it combines two parameter-free
ordinal costs: target-feature cosine to the frozen source classifier rows and
frozen CLIP class probability.  A transportation LP then preserves the exact
fixed-CLIP class quota while changing which conflict sample occupies each
quota.  Label-free outputs are SHA256-locked before Phase 2 parses labels for
explicit oracle diagnostics.

No target image, ResNet, bottleneck, CLIP model, forward, backward, optimizer,
parameter update, proxy training, or full VisDA training is used.
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

from src.utils.conflict_boundary import paired_accuracy_bootstrap_ci  # noqa: E402
from src.utils.pcgrad_feature_jacobian_audit import (  # noqa: E402
    classifier_probability,
    effective_weight_normalized_linear,
)
from src.utils.prototype_transport_audit import (  # noqa: E402
    capacity_preserving_transport,
    classifier_replay_boundary_diagnostics,
    evaluate_prototype_transport_gate,
    prototype_cosine,
    row_ordinal_cost,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CONFLICTS = 7_070
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
STEM = "visda_conflict_prototype_transport"


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
        default=DEFAULT_BASE / "conflict_prototype_transport_audit",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2_020)
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


def _load_classifier(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("source_C checkpoint is not a state dictionary")
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
        raise RuntimeError(f"Unsupported source_C weight keys: {keys}")
    if "fc.bias" not in state:
        raise RuntimeError("source_C checkpoint is missing fc.bias")
    bias = np.asarray(state["fc.bias"].detach().cpu(), dtype=np.float64)
    return weight, bias, keys


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
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    candidate_correct = np.asarray(candidate) == labels
    baseline_correct = np.asarray(baseline) == labels
    interval = paired_accuracy_bootstrap_ci(
        candidate_correct,
        baseline_correct,
        repeats=repeats,
        seed=seed,
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


def _macro_accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    return float(
        np.mean(
            [
                (prediction[labels == class_index] == class_index).mean() * 100.0
                for class_index in range(EXPECTED_CLASSES)
            ]
        )
    )


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    oracle = summary["oracle_diagnostic"]
    best = oracle["best_baseline_name"]
    comparison = oracle["comparisons"][best]
    lines = [
        "# VisDA Conflict Prototype-Transport Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "## Evidence table",
        "",
        "| Evidence | Status | Source |",
        "|---|---|---|",
        (
            "| The candidate preserves fixed-CLIP conflict class counts | "
            f"`{summary['label_free_metrics']['quota_exact']}` | Locked LP output |"
        ),
        (
            "| The assignment is an integral sample-to-class transport | "
            f"max error `{summary['label_free_metrics']['integrality_max_error']:.3e}` "
            "| Locked LP output |"
        ),
        (
            "| Candidate beats the strongest matched conflict baseline | "
            f"`{comparison['gain_pp']:.6f}` pp, CI "
            f"`{comparison['paired_bootstrap_95_ci_pp']}` | Oracle diagnostic |"
        ),
        (
            "| Full proxy macro effect before any training | "
            f"`{oracle['full_macro_delta_pp']:.6f}` pp | Oracle diagnostic |"
        ),
        "",
        "## Label-free mechanism",
        "",
        "On cycle-1 task/CLIP conflicts, sum the within-sample ordinal ranks of",
        "(1) target bottleneck cosine to frozen source classifier rows and (2)",
        "frozen CLIP probabilities. Solve one exact transportation LP whose",
        "class quotas equal fixed-CLIP conflict counts. No fitted scale,",
        "temperature, threshold, or target label enters the rule.",
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
            "A PASS authorizes review of one matched proxy design only. This",
            "audit does not authorize or start proxy/full VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (
        args.snapshot,
        args.source_lock,
        args.source_classifier,
        args.target_list,
        args.class_names,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prototype-transport input: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")
    class_names = tuple(
        line.strip().replace("_", " ")
        for line in args.class_names.read_text().splitlines()
        if line.strip()
    )
    if class_names != CLASS_NAMES:
        raise RuntimeError("VisDA class-name contract changed")
    if args.bootstrap_repeats < 100:
        raise ValueError("bootstrap-repeats must be at least 100")

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
            raise ValueError(f"Snapshot is missing keys: {sorted(missing)}")
        cycle = int(np.asarray(snapshot["cycle"]).item())
        label_mask = np.asarray(snapshot["label_mask"], dtype=bool).copy()
        task_label = np.asarray(snapshot["source_label"], dtype=np.int64).copy()
        clip_label = np.asarray(snapshot["clip_label"], dtype=np.int64).copy()
        task_probability = np.asarray(snapshot["task_prob"], dtype=np.float64).copy()
        clip_probability = np.asarray(snapshot["clip_prob"], dtype=np.float64).copy()
        task_feature = np.asarray(snapshot["task_feature"], dtype=np.float64).copy()
        sample_index = np.asarray(snapshot["sample_index"], dtype=np.int64).copy()

    classifier_weight, classifier_bias, classifier_keys = _load_classifier(
        args.source_classifier
    )
    replay_probability = classifier_probability(
        task_feature, classifier_weight, classifier_bias
    )
    replay_diagnostic = classifier_replay_boundary_diagnostics(
        task_probability, replay_probability
    )
    replay_error = replay_diagnostic["max_probability_error"]
    conflict_mask = task_label != clip_label
    conflict_index = np.flatnonzero(conflict_mask)
    conflict_clip = clip_probability[conflict_mask]
    conflict_feature = task_feature[conflict_mask]
    source_cosine = prototype_cosine(conflict_feature, classifier_weight)
    source_rank = row_ordinal_cost(source_cosine)
    clip_rank = row_ordinal_cost(conflict_clip)
    transport_cost = source_rank + clip_rank
    clip_quota = np.bincount(
        clip_label[conflict_mask], minlength=EXPECTED_CLASSES
    ).astype(np.int64)
    transport = capacity_preserving_transport(transport_cost, clip_quota)
    candidate_prediction = transport["prediction"]

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
    baselines_full = {
        "fixed_task": task_label,
        "fixed_clip": clip_label,
        "confidence_choice": confidence_prediction,
        "arithmetic": arithmetic_prediction,
        "rms": rms_prediction,
    }
    full_candidate = clip_label.copy()
    full_candidate[conflict_mask] = candidate_prediction
    candidate_quota = np.bincount(candidate_prediction, minlength=EXPECTED_CLASSES)
    quota_exact = np.array_equal(candidate_quota, clip_quota)
    changed_fraction_pct = float(
        (candidate_prediction != clip_label[conflict_mask]).mean() * 100.0
    )

    input_checks = {
        "source_snapshot_matches_cycle_memory_lock": (
            snapshot_sha256 == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
        ),
        "source_lock_declares_labels_after_manifest": bool(
            source_lock.get("labels_read_after_this_manifest")
        ),
        "snapshot_is_pre_cycle1": cycle == 1,
        "expected_probability_shapes": (
            task_probability.shape
            == clip_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "expected_feature_shape": task_feature.shape
        == (EXPECTED_SAMPLES, EXPECTED_FEATURES),
        "probabilities_finite_normalized": bool(
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
        "duet_mask_is_cycle1_agreement": np.array_equal(
            label_mask, task_label == clip_label
        ),
        "expected_agreement_count": int(label_mask.sum())
        == EXPECTED_SAMPLES - EXPECTED_CONFLICTS,
        "expected_conflict_count": conflict_index.size == EXPECTED_CONFLICTS,
        "classifier_shape": classifier_weight.shape
        == (EXPECTED_CLASSES, EXPECTED_FEATURES),
        "frozen_classifier_replay_disagreements_are_boundary_ties": (
            replay_diagnostic["all_mismatches_within_2linf_margin"]
        ),
        "frozen_classifier_probability_error_at_most_5e_4": replay_error <= 5e-4,
        "transport_quota_exact": quota_exact,
        "transport_integral": transport["integrality_max_error"] <= 1e-6,
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Prototype-transport input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    signal_path = args.output_dir / f"{STEM}_label_free.npz"
    lock_path = args.output_dir / f"{STEM}_signal_lock.json"
    oracle_path = args.output_dir / f"{STEM}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{STEM}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{STEM}_summary.json"
    markdown_path = args.output_dir / f"{STEM}_summary.md"
    np.savez_compressed(
        signal_path,
        conflict_index=conflict_index,
        candidate_prediction=candidate_prediction,
        fixed_task_prediction=task_label[conflict_mask],
        fixed_clip_prediction=clip_label[conflict_mask],
        confidence_prediction=confidence_prediction[conflict_mask],
        arithmetic_prediction=arithmetic_prediction[conflict_mask],
        rms_prediction=rms_prediction[conflict_mask],
        source_prototype_cosine=source_cosine.astype(np.float32),
        source_ordinal_cost=source_rank.astype(np.int8),
        clip_ordinal_cost=clip_rank.astype(np.int8),
        transport_cost=transport_cost.astype(np.int8),
        fixed_clip_class_quota=clip_quota,
        candidate_class_count=candidate_quota,
    )
    lock = {
        "phase": "LABEL_FREE_VISDA_CONFLICT_PROTOTYPE_TRANSPORT_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "scope": "pre_cycle1_task_clip_top1_conflicts",
            "source_geometry": (
                "cosine(target_task_bottleneck_feature, frozen_source_C_class_row)"
            ),
            "semantic_prior": "frozen_clip_class_probability",
            "cost": "source_geometry_ordinal_rank_plus_clip_ordinal_rank",
            "transport": "unregularized_linear_program",
            "row_supply": "one_per_conflict_sample",
            "class_quota": "fixed_clip_top1_conflict_class_counts",
            "temperature": None,
            "fitted_parameters": False,
            "target_label_thresholds": False,
            "training_change_in_this_audit": False,
        },
        "literature_provenance": {
            "inspiration_not_replication": (
                "Vision-Language Model Guided Source-Free Domain Adaptation via "
                "Optimal Transport (CVPR 2026)"
            ),
            "primary_source": (
                "https://openaccess.thecvf.com/content/CVPR2026/html/"
                "Han_Vision-Language_Model_Guided_Source-Free_Domain_Adaptation_"
                "via_Optimal_Transport_CVPR_2026_paper.html"
            ),
            "difference_from_vsfot": (
                "offline fixed-feature exact-quota diagnostic only; no soft OT "
                "training or bidirectional distillation"
            ),
        },
        "predeclared_gate": {
            "min_changed_conflicts_pct": 5.0,
            "min_gain_vs_best_baseline_pp": 1.0,
            "best_baseline_paired_ci_lower_pp": "> 0",
            "min_full_proxy_macro_gain_pp": 0.2,
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": {
            "samples": EXPECTED_SAMPLES,
            "conflicts": EXPECTED_CONFLICTS,
            "agreements": EXPECTED_SAMPLES - EXPECTED_CONFLICTS,
            "changed_conflicts": int(
                (candidate_prediction != clip_label[conflict_mask]).sum()
            ),
            "changed_fraction_pct": changed_fraction_pct,
            "fixed_clip_class_quota": clip_quota.tolist(),
            "candidate_class_count": candidate_quota.tolist(),
            "quota_exact": quota_exact,
            "transport_objective": transport["objective"],
            "transport_integrality_max_error": transport["integrality_max_error"],
            "transport_row_sum_max_error": transport["row_sum_max_error"],
            "transport_class_sum_max_error": transport["class_sum_max_error"],
            "solver_status": transport["solver_status"],
            "solver_message": transport["solver_message"],
            "frozen_classifier_max_probability_replay_error": replay_error,
            "frozen_classifier_top1_mismatch_count": replay_diagnostic[
                "top1_mismatch_count"
            ],
            "frozen_classifier_top1_mismatch_fraction_pct": replay_diagnostic[
                "top1_mismatch_fraction_pct"
            ],
            "frozen_classifier_max_reference_margin_on_mismatch": (
                replay_diagnostic["max_reference_margin_on_mismatch"]
            ),
            "frozen_classifier_replay_disagreements_are_boundary_ties": (
                replay_diagnostic["all_mismatches_within_2linf_margin"]
            ),
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
            "source_classifier": {
                "path": str(args.source_classifier),
                "sha256": _sha256(args.source_classifier),
                "state_keys": classifier_keys,
            },
            "target_list_opaque_sha256": _sha256(args.target_list),
            "class_names_sha256": _sha256(args.class_names),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/prototype_transport_audit.py": _sha256(
                REPO_ROOT / "src/utils/prototype_transport_audit.py"
            ),
            "tools/audit_visda_conflict_prototype_transport.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: oracle diagnostic only, strictly after the label-free lock.
    target_hash_matches = (
        _sha256(args.target_list) == lock["inputs"]["target_list_opaque_sha256"]
    )
    labels = _parse_labels_after_lock(args.target_list)
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        embedded_labels = np.asarray(snapshot["target_label"], dtype=np.int64).copy()
    labels_match_snapshot = np.array_equal(labels[sample_index], embedded_labels)
    labels = labels[sample_index]
    conflict_labels = labels[conflict_mask]
    comparisons = {
        name: _comparison(
            candidate_prediction,
            prediction[conflict_mask],
            conflict_labels,
            repeats=args.bootstrap_repeats,
            seed=args.seed + offset,
        )
        for offset, (name, prediction) in enumerate(baselines_full.items())
    }
    best_baseline_name = max(
        comparisons,
        key=lambda name: comparisons[name]["baseline_accuracy_pct"],
    )
    best_full = baselines_full[best_baseline_name]
    candidate_macro = _macro_accuracy(full_candidate, labels)
    baseline_macro = _macro_accuracy(best_full, labels)
    full_macro_delta = candidate_macro - baseline_macro

    oracle_rows = []
    for local_index, global_index in enumerate(conflict_index):
        oracle_rows.append(
            {
                "proxy_index": int(global_index),
                "oracle_target_label": int(labels[global_index]),
                "candidate_prediction": int(candidate_prediction[local_index]),
                "candidate_correct": bool(
                    candidate_prediction[local_index] == labels[global_index]
                ),
                "best_baseline_name": best_baseline_name,
                "best_baseline_prediction": int(best_full[global_index]),
                "best_baseline_correct": bool(
                    best_full[global_index] == labels[global_index]
                ),
                "fixed_task_prediction": int(task_label[global_index]),
                "fixed_clip_prediction": int(clip_label[global_index]),
                "candidate_changed_from_clip": bool(
                    candidate_prediction[local_index] != clip_label[global_index]
                ),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(oracle_path, oracle_rows)

    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        full_mask = labels == class_index
        conflict_class_mask = conflict_labels == class_index
        candidate_class_accuracy = float(
            (full_candidate[full_mask] == class_index).mean() * 100.0
        )
        baseline_class_accuracy = float(
            (best_full[full_mask] == class_index).mean() * 100.0
        )
        if conflict_class_mask.any():
            conflict_candidate_accuracy = float(
                (
                    candidate_prediction[conflict_class_mask]
                    == conflict_labels[conflict_class_mask]
                ).mean()
                * 100.0
            )
            conflict_baseline_accuracy = float(
                (
                    best_full[conflict_mask][conflict_class_mask]
                    == conflict_labels[conflict_class_mask]
                ).mean()
                * 100.0
            )
        else:
            conflict_candidate_accuracy = 0.0
            conflict_baseline_accuracy = 0.0
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "full_samples": int(full_mask.sum()),
                "conflict_samples": int(conflict_class_mask.sum()),
                "candidate_full_accuracy_pct": candidate_class_accuracy,
                "best_baseline_name": best_baseline_name,
                "best_baseline_full_accuracy_pct": baseline_class_accuracy,
                "candidate_minus_best_full_pp": (
                    candidate_class_accuracy - baseline_class_accuracy
                ),
                "candidate_conflict_accuracy_pct": conflict_candidate_accuracy,
                "best_baseline_conflict_accuracy_pct": conflict_baseline_accuracy,
                "candidate_minus_best_conflict_pp": (
                    conflict_candidate_accuracy - conflict_baseline_accuracy
                ),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(class_path, class_rows)
    class_delta = np.asarray(
        [row["candidate_minus_best_full_pp"] for row in class_rows],
        dtype=np.float64,
    )
    car_delta = float(class_delta[3])
    truck_delta = float(class_delta[11])
    car_truck_mean = float(class_delta[[3, 11]].mean())
    other_ten_mean = float(np.delete(class_delta, [3, 11]).mean())
    gate = evaluate_prototype_transport_gate(
        input_contract_valid=all(input_checks.values())
        and target_hash_matches
        and labels_match_snapshot,
        quota_exact=quota_exact,
        integrality_max_error=transport["integrality_max_error"],
        changed_fraction_pct=changed_fraction_pct,
        comparisons=comparisons,
        best_baseline_name=best_baseline_name,
        full_macro_delta_pp=full_macro_delta,
        car_delta_pp=car_delta,
        truck_delta_pp=truck_delta,
        car_truck_mean_delta_pp=car_truck_mean,
        other_ten_mean_delta_pp=other_ten_mean,
    )
    summary = {
        "decision": gate["decision"],
        "checks": gate["checks"],
        "gate": gate,
        "method_status": (
            "single_cpu_offline_preflight; no proxy/full training authorized"
        ),
        "labels_used_only_after_signal_lock": True,
        "signal_lock_sha256": _sha256(lock_path),
        "input_contract_checks": input_checks,
        "label_free_metrics": {
            **lock["label_free_metrics"],
            "integrality_max_error": transport["integrality_max_error"],
        },
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "target_list_hash_matches_lock": target_hash_matches,
            "target_labels_match_embedded_snapshot_after_lock": (labels_match_snapshot),
            "best_baseline_name": best_baseline_name,
            "comparisons": comparisons,
            "candidate_full_macro_accuracy_pct": candidate_macro,
            "best_baseline_full_macro_accuracy_pct": baseline_macro,
            "full_macro_delta_pp": full_macro_delta,
            "car_delta_pp": car_delta,
            "truck_delta_pp": truck_delta,
            "car_truck_mean_delta_pp": car_truck_mean,
            "other_ten_mean_delta_pp": other_ten_mean,
        },
        "outputs": {
            "label_free_signal": str(signal_path),
            "signal_lock": str(lock_path),
            "oracle_diagnostic": str(oracle_path),
            "classwise_oracle_diagnostic": str(class_path),
            "markdown": str(markdown_path),
        },
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(
        json.dumps(
            {"decision": summary["decision"], "checks": gate["checks"]}, indent=2
        )
    )
    print(f"Wrote label-free transport: {signal_path}")
    print(f"Locked signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
