#!/usr/bin/env python3
"""CPU-only full-DUET audit of an agreement-only transport KL target.

The candidate changes one thing: on cycle-1 task/CLIP top-1 agreements, the
original CLIP KL target is replaced by a row-normalized CLIP Sinkhorn target.
Conflicts, pseudo-label CE, consistency, loss weights, and masks stay fixed.
The two matched controls are original DUET and replacing the same KL rows with
the already admitted hard pseudo label (duplicate hard CE).

Eight fixed batch replays are locked before target labels are parsed for
explicit oracle feature-gradient diagnostics. No image/model forward,
backward, optimizer, parameter update, proxy run, or full training is used.
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

from src.utils.agreement_transport_full_gradient_audit import (  # noqa: E402
    agreement_transport_joint_descents,
    evaluate_agreement_transport_gate,
)
from src.utils.candidate_set_gradient_audit import (  # noqa: E402
    oracle_ce_logit_descent,
    paired_mean_bootstrap_ci,
    rowwise_oracle_alignment,
)
from src.utils.pcgrad_feature_jacobian_audit import (  # noqa: E402
    classifier_probability,
    effective_weight_normalized_linear,
    map_joint_logit_descent_to_feature,
)
from src.utils.prototype_transport_audit import (  # noqa: E402
    classifier_replay_boundary_diagnostics,
)
from src.utils.vsfot_alignment_audit import (  # noqa: E402
    row_cosine,
    vsfot_transport_probability,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_AGREEMENTS = 6_777
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
STEM = "visda_agreement_transport_full_gradient"


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
        "--prior-vsfot-lock",
        type=Path,
        default=(
            DEFAULT_BASE
            / "conflict_vsfot_alignment_audit"
            / "visda_conflict_vsfot_alignment_signal_lock.json"
        ),
    )
    parser.add_argument(
        "--prior-vsfot-summary",
        type=Path,
        default=(
            DEFAULT_BASE
            / "conflict_vsfot_alignment_audit"
            / "visda_conflict_vsfot_alignment_summary.json"
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
        default=DEFAULT_BASE / "agreement_transport_full_gradient_audit",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2_020)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
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
        raise ValueError("Target label outside class range")
    return result


def _comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    difference = candidate[mask] - baseline[mask]
    interval = paired_mean_bootstrap_ci(difference, repeats=repeats, seed=seed)
    return {
        "samples": int(mask.sum()),
        "candidate_mean": float(candidate[mask].mean()),
        "baseline_mean": float(baseline[mask].mean()),
        "mean_difference": float(difference.mean()),
        "paired_bootstrap_95_ci": list(interval),
    }


def _mean_row_cosine(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    return float(row_cosine(first[mask], second[mask]).mean())


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    oracle = summary["oracle_diagnostic"]
    versus_duet = oracle["comparisons"]["original_duet"]["agreement"]
    versus_ce = oracle["comparisons"]["duplicate_hard_ce"]["agreement"]
    lines = [
        "# VisDA Agreement-Only Transport Full-Gradient Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Maximum Sinkhorn marginal error | "
            f"`{summary['label_free_metrics']['max_sinkhorn_marginal_error']:.3e}` "
            "| Label-free replay |"
        ),
        (
            "| Minimum transport-target replay median cosine | "
            f"`{summary['label_free_metrics']['minimum_target_replay_median_cosine']:.6f}` "
            "| Label-free replay |"
        ),
        (
            "| Agreement first-order gain vs original complete DUET | "
            f"`{versus_duet['first_order']['mean_difference']:.9f}`; CI "
            f"`{versus_duet['first_order']['paired_bootstrap_95_ci']}` "
            "| Oracle diagnostic after lock |"
        ),
        (
            "| Agreement first-order gain vs duplicate-hard-CE control | "
            f"`{versus_ce['first_order']['mean_difference']:.9f}`; CI "
            f"`{versus_ce['first_order']['paired_bootstrap_95_ci']}` "
            "| Oracle diagnostic after lock |"
        ),
        "",
        "## Changed variable",
        "",
        "Only the KL target on task/CLIP top-1 agreements changes. Conflicts,",
        "hard pseudo-label CE, consistency, masks, and all loss coefficients",
        "remain fixed. Row normalization makes the transported target a matched",
        "probability target without a fitted loss scale.",
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
            "Even NEEDS_EXACT_PARAMETER_AUDIT authorizes no proxy or full",
            "training. REJECT closes the route without GPU work.",
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
        args.prior_vsfot_lock,
        args.prior_vsfot_summary,
        args.source_classifier,
        args.target_list,
        args.class_names,
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing agreement-transport input: {input_path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")
    if args.replays != 8 or args.batch_size != 64:
        raise ValueError("predeclared contract requires 8 replays and batch size 64")
    if args.bootstrap_repeats < 100:
        raise ValueError("bootstrap-repeats must be at least 100")
    class_names = tuple(
        line.strip().replace("_", " ")
        for line in args.class_names.read_text().splitlines()
        if line.strip()
    )
    if class_names != CLASS_NAMES:
        raise RuntimeError("VisDA class-name contract changed")

    source_lock = json.loads(args.source_lock.read_text())
    prior_lock = json.loads(args.prior_vsfot_lock.read_text())
    snapshot_sha256 = _sha256(args.snapshot)
    required = {
        "cycle",
        "label_mask",
        "source_label",
        "clip_label",
        "task_prob",
        "clip_prob",
        "strong_task_prob",
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
        agreement = np.asarray(snapshot["label_mask"], dtype=bool).copy()
        task_label = np.asarray(snapshot["source_label"], dtype=np.int64).copy()
        clip_label = np.asarray(snapshot["clip_label"], dtype=np.int64).copy()
        weak_probability = np.asarray(snapshot["task_prob"], dtype=np.float64).copy()
        clip_probability = np.asarray(snapshot["clip_prob"], dtype=np.float64).copy()
        strong_probability = np.asarray(
            snapshot["strong_task_prob"], dtype=np.float64
        ).copy()
        task_feature = np.asarray(snapshot["task_feature"], dtype=np.float64).copy()
        sample_index = np.asarray(snapshot["sample_index"], dtype=np.int64).copy()

    classifier_weight, classifier_bias, classifier_keys = _load_classifier(
        args.source_classifier
    )
    replay_probability = classifier_probability(
        task_feature, classifier_weight, classifier_bias
    )
    classifier_replay = classifier_replay_boundary_diagnostics(
        weak_probability, replay_probability
    )
    input_checks = {
        "source_snapshot_matches_cycle_memory_lock": (
            snapshot_sha256 == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
        ),
        "prior_vsfot_uses_same_snapshot": (
            snapshot_sha256
            == prior_lock.get("inputs", {}).get("pre_cycle1_snapshot", {}).get("sha256")
        ),
        "prior_vsfot_lock_is_label_free": (
            prior_lock.get("contains_target_labels") is False
            and prior_lock.get("target_list_not_parsed_before_lock") is True
        ),
        "prior_vsfot_batch_direction_stable": (
            prior_lock.get("label_free_metrics", {}).get(
                "minimum_replay_median_cosine", 0.0
            )
            >= 0.90
        ),
        "snapshot_is_pre_cycle1": cycle == 1,
        "probability_shapes": (
            weak_probability.shape
            == strong_probability.shape
            == clip_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "feature_and_classifier_shapes": (
            task_feature.shape == (EXPECTED_SAMPLES, EXPECTED_FEATURES)
            and classifier_weight.shape == (EXPECTED_CLASSES, EXPECTED_FEATURES)
        ),
        "probabilities_finite_normalized": all(
            np.isfinite(value).all() and np.allclose(value.sum(axis=1), 1.0, atol=1e-5)
            for value in (weak_probability, strong_probability, clip_probability)
        ),
        "sample_indices_are_proxy_order": np.array_equal(
            sample_index, np.arange(EXPECTED_SAMPLES)
        ),
        "saved_predictions_match_probabilities": (
            np.array_equal(task_label, weak_probability.argmax(axis=1))
            and np.array_equal(clip_label, clip_probability.argmax(axis=1))
        ),
        "agreement_mask_matches_task_clip_top1": np.array_equal(
            agreement, task_label == clip_label
        ),
        "expected_agreement_and_conflict_counts": (
            int(agreement.sum()) == EXPECTED_AGREEMENTS
            and int((~agreement).sum()) == EXPECTED_CONFLICTS
        ),
        "classifier_replay_disagreements_are_boundary_ties": classifier_replay[
            "all_mismatches_within_2linf_margin"
        ],
        "classifier_probability_error_at_most_5e_4": (
            classifier_replay["max_probability_error"] <= 5e-4
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Agreement-transport input contract failed: {failed}")

    replay_orders = []
    replay_targets = []
    replay_descents = []
    max_sinkhorn_error = 0.0
    max_sinkhorn_iterations = 0
    dual_refined_batches = 0
    for replay_index in range(args.replays):
        order = np.random.default_rng(args.seed + replay_index).permutation(
            EXPECTED_SAMPLES
        )
        replay_orders.append(order)
        transported = vsfot_transport_probability(
            clip_probability,
            weak_probability,
            order,
            batch_size=args.batch_size,
            regularization=0.2,
        )
        target = transported["probability"]
        replay_targets.append(target)
        replay_descents.append(
            agreement_transport_joint_descents(
                weak_probability,
                strong_probability,
                clip_probability,
                task_label,
                agreement,
                target,
                order,
                batch_size=args.batch_size,
            )
        )
        max_sinkhorn_error = max(
            max_sinkhorn_error, transported["max_sinkhorn_marginal_error"]
        )
        max_sinkhorn_iterations = max(
            max_sinkhorn_iterations, transported["max_sinkhorn_iterations"]
        )
        dual_refined_batches += transported["dual_refined_batches"]

    target_replay_median_cosines = [1.0]
    for target in replay_targets[1:]:
        target_replay_median_cosines.append(
            float(
                np.median(row_cosine(replay_targets[0][agreement], target[agreement]))
            )
        )
    minimum_target_replay_median_cosine = float(min(target_replay_median_cosines))
    one_hot_task = np.eye(EXPECTED_CLASSES, dtype=np.float64)[task_label]
    transport_component = replay_targets[0] - weak_probability
    ce_component = one_hot_task - weak_probability
    mean_transport_ce_component_cosine = _mean_row_cosine(
        transport_component, ce_component, agreement
    )
    target_top1_match_pct = float(
        (replay_targets[0][agreement].argmax(axis=1) == task_label[agreement]).mean()
        * 100.0
    )
    routed_target = clip_probability.copy()
    routed_target[agreement] = replay_targets[0][agreement]
    class_mass_shift = routed_target.mean(axis=0) - clip_probability.mean(axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    signal_path = args.output_dir / f"{STEM}_label_free.npz"
    lock_path = args.output_dir / f"{STEM}_signal_lock.json"
    oracle_path = args.output_dir / f"{STEM}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{STEM}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{STEM}_summary.json"
    markdown_path = args.output_dir / f"{STEM}_summary.md"
    np.savez_compressed(
        signal_path,
        agreement_mask=agreement,
        sample_index=sample_index,
        primary_transport_target=replay_targets[0].astype(np.float32),
        duet_joint=np.stack(
            [result["duet_joint"] for result in replay_descents]
        ).astype(np.float32),
        agreement_transport_joint=np.stack(
            [result["agreement_transport_joint"] for result in replay_descents]
        ).astype(np.float32),
        duplicate_hard_ce_joint=np.stack(
            [result["duplicate_hard_ce_joint"] for result in replay_descents]
        ).astype(np.float32),
        target_replay_median_cosines=np.asarray(
            target_replay_median_cosines, dtype=np.float64
        ),
        replay_order_sha256=np.asarray(
            [hashlib.sha256(order.tobytes()).hexdigest() for order in replay_orders]
        ),
    )
    label_free_metrics = {
        "samples": EXPECTED_SAMPLES,
        "agreements": EXPECTED_AGREEMENTS,
        "conflicts_unchanged": EXPECTED_CONFLICTS,
        "max_sinkhorn_marginal_error": max_sinkhorn_error,
        "max_sinkhorn_iterations": max_sinkhorn_iterations,
        "dual_refined_batches_across_replays": dual_refined_batches,
        "target_replay_median_cosines": target_replay_median_cosines,
        "minimum_target_replay_median_cosine": minimum_target_replay_median_cosine,
        "agreement_transport_target_top1_matches_task_pct": target_top1_match_pct,
        "agreement_transport_component_mean_cosine_with_existing_ce": (
            mean_transport_ce_component_cosine
        ),
        "routed_target_class_mass_shift": class_mass_shift.tolist(),
        "max_absolute_routed_target_class_mass_shift_pp": float(
            np.max(np.abs(class_mass_shift)) * 100.0
        ),
    }
    lock = {
        "phase": "LABEL_FREE_VISDA_AGREEMENT_TRANSPORT_FULL_GRADIENT_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "changed_variable": "KL_target_on_task_clip_top1_agreements_only",
            "agreement_transport_target": "row_normalized_clip_sinkhorn_times_inverse_task_frequency",
            "conflict_KL_target": "original_clip_probability_unchanged",
            "pseudo_label_CE": "unchanged",
            "consistency": "unchanged",
            "loss_weights": {"pseudo_ce": 0.4, "consistency": 0.2, "kl": 0.4},
            "matched_controls": [
                "original_complete_DUET_gradient",
                "same_agreement_rows_replace_KL_with_duplicate_hard_CE",
            ],
            "fitted_parameters": False,
            "target_label_thresholds": False,
            "training_change_in_this_audit": False,
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "inputs": {
            "pre_cycle1_snapshot": {
                "path": str(args.snapshot),
                "sha256": snapshot_sha256,
            },
            "cycle2_memory_signal_lock": {
                "path": str(args.source_lock),
                "sha256": _sha256(args.source_lock),
            },
            "prior_vsfot_signal_lock": {
                "path": str(args.prior_vsfot_lock),
                "sha256": _sha256(args.prior_vsfot_lock),
            },
            "prior_vsfot_summary_opaque_sha256": _sha256(args.prior_vsfot_summary),
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
            "src/utils/agreement_transport_full_gradient_audit.py": _sha256(
                REPO_ROOT / "src/utils/agreement_transport_full_gradient_audit.py"
            ),
            "src/utils/vsfot_alignment_audit.py": _sha256(
                REPO_ROOT / "src/utils/vsfot_alignment_audit.py"
            ),
            "tools/audit_visda_agreement_transport_full_gradient.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: explicit oracle diagnostic, strictly after the signal lock.
    target_hash_matches = (
        _sha256(args.target_list) == lock["inputs"]["target_list_opaque_sha256"]
    )
    prior_summary_hash_matches = (
        _sha256(args.prior_vsfot_summary)
        == lock["inputs"]["prior_vsfot_summary_opaque_sha256"]
    )
    prior_summary = json.loads(args.prior_vsfot_summary.read_text())
    prior_reject_preserved = prior_summary.get(
        "decision"
    ) == "REJECT" and prior_summary.get("signal_lock_sha256") == _sha256(
        args.prior_vsfot_lock
    )
    labels = _parse_labels_after_lock(args.target_list)
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        embedded_labels = np.asarray(snapshot["target_label"], dtype=np.int64).copy()
    labels_match_snapshot = np.array_equal(labels[sample_index], embedded_labels)
    labels = labels[sample_index]
    oracle_joint = np.concatenate(
        (
            oracle_ce_logit_descent(weak_probability, labels),
            oracle_ce_logit_descent(strong_probability, labels),
        ),
        axis=1,
    )
    oracle_feature = map_joint_logit_descent_to_feature(oracle_joint, classifier_weight)

    score_by_method = {
        name: {"first_order": [], "cosine": [], "norm": []}
        for name in ("original_duet", "agreement_transport", "duplicate_hard_ce")
    }
    descent_key = {
        "original_duet": "duet_joint",
        "agreement_transport": "agreement_transport_joint",
        "duplicate_hard_ce": "duplicate_hard_ce_joint",
    }
    for replay in replay_descents:
        for method, key in descent_key.items():
            feature = map_joint_logit_descent_to_feature(replay[key], classifier_weight)
            alignment = rowwise_oracle_alignment(feature, oracle_feature)
            score_by_method[method]["first_order"].append(alignment["first_order"])
            score_by_method[method]["cosine"].append(alignment["cosine"])
            score_by_method[method]["norm"].append(np.linalg.norm(feature, axis=1))
    for method in score_by_method:
        for metric in score_by_method[method]:
            score_by_method[method][metric] = np.stack(score_by_method[method][metric])

    mean_score = {
        method: {metric: values.mean(axis=0) for metric, values in metrics.items()}
        for method, metrics in score_by_method.items()
    }
    scopes = {
        "overall": np.ones(EXPECTED_SAMPLES, dtype=bool),
        "agreement": agreement,
    }
    controls = ("original_duet", "duplicate_hard_ce")
    comparisons = {
        control: {
            scope_name: {
                metric: _comparison(
                    mean_score["agreement_transport"][metric],
                    mean_score[control][metric],
                    scope_mask,
                    repeats=args.bootstrap_repeats,
                    seed=(
                        args.seed
                        + 100 * control_index
                        + 10 * scope_index
                        + metric_index
                    ),
                )
                for metric_index, metric in enumerate(("first_order", "cosine"))
            }
            for scope_index, (scope_name, scope_mask) in enumerate(scopes.items())
        }
        for control_index, control in enumerate(controls)
    }
    every_replay_gain = {
        control: bool(
            np.all(
                score_by_method["agreement_transport"]["first_order"][
                    :, agreement
                ].mean(axis=1)
                > score_by_method[control]["first_order"][:, agreement].mean(axis=1)
            )
        )
        for control in controls
    }
    control_overall = {
        control: float(mean_score[control]["first_order"].mean())
        for control in controls
    }
    strongest_control = max(control_overall, key=control_overall.get)
    candidate_first_order = mean_score["agreement_transport"]["first_order"]
    strongest_first_order = mean_score[strongest_control]["first_order"]
    candidate_negative_burden = float(np.minimum(candidate_first_order, 0.0).mean())
    strongest_negative_burden = float(np.minimum(strongest_first_order, 0.0).mean())
    candidate_mean_norm = float(score_by_method["agreement_transport"]["norm"].mean())
    strongest_mean_norm = float(score_by_method[strongest_control]["norm"].mean())
    mean_norm_ratio = (
        candidate_mean_norm / strongest_mean_norm
        if strongest_mean_norm > 0.0
        else np.inf
    )
    first_order_delta = candidate_first_order - strongest_first_order
    group_masks = {
        "car": labels == 3,
        "person": labels == 7,
        "truck": labels == 11,
        "other_nine": ~np.isin(labels, [3, 7, 11]),
    }
    group_delta = {
        name: float(first_order_delta[mask].mean())
        for name, mask in group_masks.items()
    }

    oracle_rows = []
    for index in range(EXPECTED_SAMPLES):
        oracle_rows.append(
            {
                "proxy_index": index,
                "is_task_clip_agreement": bool(agreement[index]),
                "oracle_target_label": int(labels[index]),
                "candidate_mean_first_order": float(candidate_first_order[index]),
                "original_duet_mean_first_order": float(
                    mean_score["original_duet"]["first_order"][index]
                ),
                "duplicate_hard_ce_mean_first_order": float(
                    mean_score["duplicate_hard_ce"]["first_order"][index]
                ),
                "candidate_minus_strongest_control": float(first_order_delta[index]),
                "strongest_control": strongest_control,
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(oracle_path, oracle_rows)

    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_mask = labels == class_index
        class_agreement = class_mask & agreement
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "samples": int(class_mask.sum()),
                "agreement_samples": int(class_agreement.sum()),
                "candidate_mean_first_order": float(
                    candidate_first_order[class_mask].mean()
                ),
                "strongest_control": strongest_control,
                "strongest_control_mean_first_order": float(
                    strongest_first_order[class_mask].mean()
                ),
                "candidate_minus_strongest_first_order": float(
                    first_order_delta[class_mask].mean()
                ),
                "agreement_candidate_minus_strongest_first_order": float(
                    first_order_delta[class_agreement].mean()
                ),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(class_path, class_rows)

    gate = evaluate_agreement_transport_gate(
        input_contract_valid=(
            all(input_checks.values())
            and target_hash_matches
            and prior_summary_hash_matches
            and prior_reject_preserved
            and labels_match_snapshot
        ),
        max_sinkhorn_marginal_error=max_sinkhorn_error,
        minimum_target_replay_median_cosine=minimum_target_replay_median_cosine,
        mean_transport_ce_component_cosine=mean_transport_ce_component_cosine,
        comparisons=comparisons,
        every_replay_agreement_first_order_gain_positive=every_replay_gain,
        candidate_negative_burden=candidate_negative_burden,
        strongest_control_negative_burden=strongest_negative_burden,
        candidate_to_strongest_mean_norm_ratio=mean_norm_ratio,
        group_first_order_delta_vs_strongest=group_delta,
    )
    summary = {
        "decision": gate["decision"],
        "checks": gate["checks"],
        "gate": gate,
        "method_status": "cpu_only_full_gradient_preflight; no parameter/proxy/full training authorized",
        "labels_used_only_after_signal_lock": True,
        "signal_lock_sha256": _sha256(lock_path),
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "target_list_hash_matches_lock": target_hash_matches,
            "prior_summary_hash_matches_lock": prior_summary_hash_matches,
            "prior_vsfot_reject_preserved": prior_reject_preserved,
            "target_labels_match_embedded_snapshot_after_lock": labels_match_snapshot,
            "comparisons": comparisons,
            "every_replay_agreement_first_order_gain_positive": every_replay_gain,
            "strongest_control": strongest_control,
            "candidate_negative_burden": candidate_negative_burden,
            "strongest_control_negative_burden": strongest_negative_burden,
            "candidate_to_strongest_mean_norm_ratio": mean_norm_ratio,
            "group_first_order_delta_vs_strongest": group_delta,
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
    print(f"Wrote label-free full gradients: {signal_path}")
    print(f"Locked full gradients before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
