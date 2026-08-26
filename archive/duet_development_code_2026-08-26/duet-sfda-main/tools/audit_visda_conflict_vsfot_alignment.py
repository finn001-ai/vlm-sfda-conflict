#!/usr/bin/env python3
"""CPU-only audit of the isolated public VSFOT alignment component.

Phase 1 replays the public VSFOT ``loss_align`` direction from the locked
pre-cycle-1 DUET probabilities/features and frozen source classifier. Eight
fixed batch orders test the method's batch-coupling sensitivity. The label-free
directions are SHA256-locked before Phase 2 reads target labels for explicit
oracle feature-direction diagnostics.

This script loads no target image, ResNet, bottleneck, or CLIP model and runs no
forward, backward, optimizer, parameter update, proxy run, or full training.
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
    paired_mean_bootstrap_ci,
)
from src.utils.pcgrad_feature_jacobian_audit import (  # noqa: E402
    classifier_probability,
    effective_weight_normalized_linear,
)
from src.utils.prototype_transport_audit import (  # noqa: E402
    classifier_replay_boundary_diagnostics,
)
from src.utils.vsfot_alignment_audit import (  # noqa: E402
    clip_kl_feature_descent,
    evaluate_vsfot_alignment_gate,
    row_cosine,
    row_unit,
    vsfot_alignment_feature_descent,
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
STEM = "visda_conflict_vsfot_alignment"
VSFOT_REPOSITORY = "https://github.com/TangXu-Group/DomainAdaptation"
VSFOT_COMMIT = "c9fbc150dc6c769b2cdebfe87b60d101a3c44277"


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
        default=DEFAULT_BASE / "conflict_vsfot_alignment_audit",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2_020)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
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


def _parse_labels_after_lock(path: Path, expected_samples: int) -> np.ndarray:
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
    if result.shape != (expected_samples,):
        raise ValueError(f"Expected {expected_samples} labels, found {result.size}")
    if np.any(result < 0) or np.any(result >= EXPECTED_CLASSES):
        raise ValueError("Target label is outside the class range")
    return result


def _comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    difference = (
        np.asarray(candidate, dtype=np.float64)[mask]
        - np.asarray(baseline, dtype=np.float64)[mask]
    )
    interval = paired_mean_bootstrap_ci(difference, repeats=repeats, seed=seed)
    return {
        "samples": int(mask.sum()),
        "candidate_mean_oracle_cosine": float(np.mean(candidate[mask])),
        "baseline_mean_oracle_cosine": float(np.mean(baseline[mask])),
        "mean_difference": float(np.mean(difference)),
        "paired_bootstrap_95_ci": list(interval),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    oracle = summary["oracle_diagnostic"]
    comparisons = oracle["comparisons"]
    lines = [
        "# VisDA Isolated VSFOT Alignment Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Sinkhorn maximum marginal error | "
            f"`{summary['label_free_metrics']['max_sinkhorn_marginal_error']:.3e}` "
            "| Label-free replay |"
        ),
        (
            "| Minimum median direction cosine across fixed batch orders | "
            f"`{summary['label_free_metrics']['minimum_replay_median_cosine']:.6f}` "
            "| Label-free replay |"
        ),
        (
            "| Candidate minus DUET CLIP-KL, unresolved conflicts | "
            f"`{comparisons['clip_kl']['conflict']['mean_difference']:.6f}`; CI "
            f"`{comparisons['clip_kl']['conflict']['paired_bootstrap_95_ci']}` "
            "| Oracle diagnostic after lock |"
        ),
        (
            "| Candidate minus transported-CLIP classification only, conflicts | "
            f"`{comparisons['transport_classification_only']['conflict']['mean_difference']:.6f}`; CI "
            f"`{comparisons['transport_classification_only']['conflict']['paired_bootstrap_95_ci']}` "
            "| Oracle diagnostic after lock |"
        ),
        "",
        "## Isolated mechanism",
        "",
        "The candidate exactly isolates the public VSFOT `loss_align`: a CLIP",
        "soft Sinkhorn coupling weights the frozen task classifier's negative-log",
        "probability and source-prototype cosine costs. SimSiam, information",
        "maximization, CLIP adapter training, and all parameter updates are excluded.",
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
            "audit never authorizes or starts proxy/full VisDA training.",
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
        args.source_classifier,
        args.target_list,
        args.class_names,
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing VSFOT-alignment input: {input_path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")
    if args.bootstrap_repeats < 100:
        raise ValueError("bootstrap-repeats must be at least 100")
    if args.replays != 8 or args.batch_size != 64:
        raise ValueError("predeclared contract requires 8 replays and batch size 64")
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
            raise ValueError(f"Snapshot is missing keys: {sorted(missing)}")
        cycle = int(np.asarray(snapshot["cycle"]).item())
        label_mask = np.asarray(snapshot["label_mask"], dtype=bool).copy()
        task_label = np.asarray(snapshot["source_label"], dtype=np.int64).copy()
        clip_label = np.asarray(snapshot["clip_label"], dtype=np.int64).copy()
        task_probability = np.asarray(snapshot["task_prob"], dtype=np.float64).copy()
        clip_probability = np.asarray(snapshot["clip_prob"], dtype=np.float64).copy()
        task_feature = np.asarray(snapshot["task_feature"], dtype=np.float64).copy()
        sample_index = np.asarray(snapshot["sample_index"], dtype=np.int64).copy()

    sample_count, class_count = task_probability.shape
    feature_count = task_feature.shape[1]
    classifier_weight, classifier_bias, classifier_keys = _load_classifier(
        args.source_classifier
    )
    replay_probability = classifier_probability(
        task_feature, classifier_weight, classifier_bias
    )
    replay_diagnostic = classifier_replay_boundary_diagnostics(
        task_probability, replay_probability
    )
    conflict_mask = task_label != clip_label
    conflict_index = np.flatnonzero(conflict_mask)
    class_scale = 1.0 / np.maximum(task_probability.mean(axis=0), 1e-4)

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
        "classifier_shape": classifier_weight.shape
        == (EXPECTED_CLASSES, EXPECTED_FEATURES),
        "probabilities_finite_normalized": bool(
            np.isfinite(task_probability).all()
            and np.isfinite(clip_probability).all()
            and np.allclose(task_probability.sum(axis=1), 1.0, atol=1e-5)
            and np.allclose(clip_probability.sum(axis=1), 1.0, atol=1e-5)
        ),
        "features_finite": bool(np.isfinite(task_feature).all()),
        "sample_indices_are_proxy_order": np.array_equal(
            sample_index, np.arange(EXPECTED_SAMPLES)
        ),
        "saved_predictions_match_probabilities": (
            np.array_equal(task_label, task_probability.argmax(axis=1))
            and np.array_equal(clip_label, clip_probability.argmax(axis=1))
        ),
        "duet_mask_is_cycle1_agreement": np.array_equal(
            label_mask, task_label == clip_label
        ),
        "expected_conflict_count": conflict_index.size == EXPECTED_CONFLICTS,
        "frozen_classifier_replay_disagreements_are_boundary_ties": (
            replay_diagnostic["all_mismatches_within_2linf_margin"]
        ),
        "frozen_classifier_probability_error_at_most_5e_4": (
            replay_diagnostic["max_probability_error"] <= 5e-4
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"VSFOT-alignment input contract failed: {failed}")

    replay_results: list[dict[str, Any]] = []
    replay_orders = []
    for replay_index in range(args.replays):
        order = np.random.default_rng(args.seed + replay_index).permutation(
            sample_count
        )
        replay_orders.append(order)
        replay_results.append(
            vsfot_alignment_feature_descent(
                task_probability,
                clip_probability,
                task_feature,
                classifier_weight,
                class_scale,
                order,
                batch_size=args.batch_size,
                regularization=0.2,
            )
        )
    clip_descent = clip_kl_feature_descent(
        task_probability, clip_probability, classifier_weight
    )
    primary_unit = row_unit(replay_results[0]["combined_descent"])
    replay_median_cosines = [1.0]
    for result in replay_results[1:]:
        replay_median_cosines.append(
            float(np.median(row_cosine(primary_unit, result["combined_descent"])))
        )
    minimum_replay_median_cosine = float(min(replay_median_cosines))
    max_sinkhorn_marginal_error = float(
        max(result["max_sinkhorn_marginal_error"] for result in replay_results)
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    signal_path = args.output_dir / f"{STEM}_label_free.npz"
    lock_path = args.output_dir / f"{STEM}_signal_lock.json"
    oracle_path = args.output_dir / f"{STEM}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{STEM}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{STEM}_summary.json"
    markdown_path = args.output_dir / f"{STEM}_summary.md"
    np.savez_compressed(
        signal_path,
        conflict_mask=conflict_mask,
        sample_index=sample_index,
        class_scale=class_scale.astype(np.float32),
        clip_kl_direction=row_unit(clip_descent).astype(np.float16),
        vsfot_combined_direction=primary_unit.astype(np.float16),
        vsfot_classification_direction=row_unit(
            replay_results[0]["classification_descent"]
        ).astype(np.float16),
        vsfot_prototype_descent=replay_results[0]["prototype_descent"].astype(
            np.float16
        ),
        replay_median_cosines=np.asarray(replay_median_cosines, dtype=np.float64),
        replay_order_sha256=np.asarray(
            [hashlib.sha256(order.tobytes()).hexdigest() for order in replay_orders]
        ),
    )
    lock = {
        "phase": "LABEL_FREE_VISDA_ISOLATED_VSFOT_ALIGNMENT_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "scope": "all_pre_cycle1_proxy_samples_with_conflict_subgroup_audit",
            "isolated_public_component": "loss_align",
            "clip_coupling": "entropic_sinkhorn_over_class_by_batch",
            "clip_cost": "alpha_times_one_minus_clip_probability_plus_negative_log_clip_probability",
            "clip_source_marginal": "within_batch_mean_clip_probability",
            "sample_target_marginal": "uniform_within_batch",
            "regularization": 0.2,
            "batch_size": args.batch_size,
            "fixed_batch_order_replays": args.replays,
            "task_cost": "alpha_times_one_minus_source_prototype_cosine_plus_negative_log_task_probability",
            "global_class_weight": "inverse_global_mean_task_probability_clamped_at_1e-4",
            "matched_controls": [
                "original_duet_clip_kl_feature_direction",
                "same_transport_coupling_classification_term_only",
            ],
            "excluded_public_components": [
                "SimSiam",
                "information_maximization",
                "reverse_CLIP_adapter_distillation",
            ],
            "fitted_parameters": False,
            "target_label_thresholds": False,
            "training_change_in_this_audit": False,
        },
        "official_implementation_provenance": {
            "repository": VSFOT_REPOSITORY,
            "commit": VSFOT_COMMIT,
            "implementation_files": ["utils/otUse.py", "main.py"],
            "paper": (
                "https://openaccess.thecvf.com/content/CVPR2026/html/"
                "Han_Vision-Language_Model_Guided_Source-Free_Domain_Adaptation_"
                "via_Optimal_Transport_CVPR_2026_paper.html"
            ),
        },
        "predeclared_gate": {
            "max_sinkhorn_marginal_error": 1e-6,
            "min_replay_median_direction_cosine": 0.90,
            "candidate_minus_both_controls_overall_and_conflict_ci_lower": "> 0",
            "every_replay_conflict_gain_vs_clip": "> 0",
            "negative_burden_vs_clip": "not_worse",
            "car_person_truck_other_nine_delta_vs_clip": ">= 0",
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": {
            "samples": sample_count,
            "conflicts": int(conflict_mask.sum()),
            "agreements": int((~conflict_mask).sum()),
            "max_sinkhorn_marginal_error": max_sinkhorn_marginal_error,
            "maximum_sinkhorn_iterations": int(
                max(result["max_sinkhorn_iterations"] for result in replay_results)
            ),
            "replay_median_direction_cosines": replay_median_cosines,
            "minimum_replay_median_cosine": minimum_replay_median_cosine,
            "frozen_classifier_max_probability_replay_error": replay_diagnostic[
                "max_probability_error"
            ],
            "frozen_classifier_top1_mismatch_count": replay_diagnostic[
                "top1_mismatch_count"
            ],
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
            "src/utils/vsfot_alignment_audit.py": _sha256(
                REPO_ROOT / "src/utils/vsfot_alignment_audit.py"
            ),
            "tools/audit_visda_conflict_vsfot_alignment.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: explicit oracle diagnostic, strictly after the signal lock.
    target_hash_matches = (
        _sha256(args.target_list) == lock["inputs"]["target_list_opaque_sha256"]
    )
    labels = _parse_labels_after_lock(args.target_list, sample_count)
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        embedded_labels = np.asarray(snapshot["target_label"], dtype=np.int64).copy()
    labels_match_snapshot = np.array_equal(labels[sample_index], embedded_labels)
    labels = labels[sample_index]
    oracle_one_hot = np.eye(class_count, dtype=np.float64)[labels]
    oracle_descent = (oracle_one_hot - task_probability) @ classifier_weight

    clip_oracle_cosine = row_cosine(clip_descent, oracle_descent)
    candidate_replay_cosines = np.stack(
        [
            row_cosine(result["combined_descent"], oracle_descent)
            for result in replay_results
        ]
    )
    classification_replay_cosines = np.stack(
        [
            row_cosine(result["classification_descent"], oracle_descent)
            for result in replay_results
        ]
    )
    candidate_oracle_cosine = candidate_replay_cosines.mean(axis=0)
    classification_oracle_cosine = classification_replay_cosines.mean(axis=0)
    scopes = {
        "overall": np.ones(sample_count, dtype=bool),
        "conflict": conflict_mask,
    }
    baseline_score = {
        "clip_kl": clip_oracle_cosine,
        "transport_classification_only": classification_oracle_cosine,
    }
    comparisons = {
        baseline_name: {
            scope_name: _comparison(
                candidate_oracle_cosine,
                score,
                scope_mask,
                repeats=args.bootstrap_repeats,
                seed=args.seed + 100 * baseline_offset + scope_offset,
            )
            for scope_offset, (scope_name, scope_mask) in enumerate(scopes.items())
        }
        for baseline_offset, (baseline_name, score) in enumerate(baseline_score.items())
    }
    replay_conflict_gains = (
        candidate_replay_cosines[:, conflict_mask].mean(axis=1)
        - clip_oracle_cosine[conflict_mask].mean()
    )
    candidate_negative_burden = float(np.minimum(candidate_oracle_cosine, 0.0).mean())
    clip_negative_burden = float(np.minimum(clip_oracle_cosine, 0.0).mean())
    difference_vs_clip = candidate_oracle_cosine - clip_oracle_cosine
    group_masks = {
        "car": labels == 3,
        "person": labels == 7,
        "truck": labels == 11,
        "other_nine": ~np.isin(labels, [3, 7, 11]),
    }
    group_delta_vs_clip = {
        name: float(difference_vs_clip[mask].mean())
        for name, mask in group_masks.items()
    }

    oracle_rows = []
    for index in range(sample_count):
        oracle_rows.append(
            {
                "proxy_index": index,
                "is_task_clip_conflict": bool(conflict_mask[index]),
                "oracle_target_label": int(labels[index]),
                "vsfot_alignment_oracle_cosine": float(candidate_oracle_cosine[index]),
                "duet_clip_kl_oracle_cosine": float(clip_oracle_cosine[index]),
                "transport_classification_only_oracle_cosine": float(
                    classification_oracle_cosine[index]
                ),
                "candidate_minus_duet_clip_kl": float(difference_vs_clip[index]),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(oracle_path, oracle_rows)

    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_mask = labels == class_index
        class_conflict_mask = class_mask & conflict_mask
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "samples": int(class_mask.sum()),
                "conflict_samples": int(class_conflict_mask.sum()),
                "vsfot_alignment_mean_oracle_cosine": float(
                    candidate_oracle_cosine[class_mask].mean()
                ),
                "duet_clip_kl_mean_oracle_cosine": float(
                    clip_oracle_cosine[class_mask].mean()
                ),
                "candidate_minus_duet_clip_kl": float(
                    difference_vs_clip[class_mask].mean()
                ),
                "conflict_candidate_minus_duet_clip_kl": float(
                    difference_vs_clip[class_conflict_mask].mean()
                ),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(class_path, class_rows)

    gate = evaluate_vsfot_alignment_gate(
        input_contract_valid=(
            all(input_checks.values()) and target_hash_matches and labels_match_snapshot
        ),
        max_sinkhorn_marginal_error=max_sinkhorn_marginal_error,
        minimum_replay_median_cosine=minimum_replay_median_cosine,
        comparisons=comparisons,
        every_replay_conflict_gain_vs_clip_positive=bool(
            np.all(replay_conflict_gains > 0.0)
        ),
        candidate_negative_burden=candidate_negative_burden,
        clip_negative_burden=clip_negative_burden,
        group_delta_vs_clip=group_delta_vs_clip,
    )
    summary = {
        "decision": gate["decision"],
        "checks": gate["checks"],
        "gate": gate,
        "method_status": "single_cpu_offline_preflight; no proxy/full training authorized",
        "labels_used_only_after_signal_lock": True,
        "signal_lock_sha256": _sha256(lock_path),
        "input_contract_checks": input_checks,
        "label_free_metrics": lock["label_free_metrics"],
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "target_list_hash_matches_lock": target_hash_matches,
            "target_labels_match_embedded_snapshot_after_lock": labels_match_snapshot,
            "comparisons": comparisons,
            "replay_conflict_gain_vs_clip": replay_conflict_gains.tolist(),
            "candidate_negative_burden": candidate_negative_burden,
            "duet_clip_kl_negative_burden": clip_negative_burden,
            "group_delta_vs_clip": group_delta_vs_clip,
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
    print(f"Wrote label-free VSFOT directions: {signal_path}")
    print(f"Locked directions before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
