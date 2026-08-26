#!/usr/bin/env python3
"""CPU-only output-gradient interference audit for unresolved VisDA conflicts.

The audit replays the released DUET consistency and CLIP-KL logit descents
from the locked pre-cycle-2 probabilities.  It applies the deterministic
two-objective PCGrad projection only where the two output directions have a
negative dot product.  Signals are locked before target labels are parsed.

This is an output-space mechanism audit, not a parameter-gradient experiment.
Even a passing result only requests a later no-update parameter micro-audit;
it never authorizes proxy or full VisDA training.
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

from src.utils.candidate_set_gradient_audit import (  # noqa: E402
    oracle_ce_logit_descent,
    paired_mean_bootstrap_ci,
    rowwise_oracle_alignment,
)
from src.utils.conflict_pcgrad_audit import (  # noqa: E402
    decision_stability,
    direction_stability,
    duet_output_descent_components,
    evaluate_conflict_pcgrad_gate,
    symmetric_pcgrad,
)
from src.utils.support_conditioned_clip_audit import (  # noqa: E402
    negative_first_order_burden,
    normalize_probability_matrix,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_CYCLE = 2
CONSISTENCY_WEIGHT = 0.2
CLIP_WEIGHT = 0.4
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
PROBABILITY_FLOORS = {
    "float32_smallest_subnormal": float(
        np.nextafter(np.float32(0.0), np.float32(1.0))
    ),
    "float32_smallest_normal": float(np.finfo(np.float32).tiny),
    "one_e_minus_30": 1e-30,
}
PRIMARY_FLOOR = "float32_smallest_subnormal"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_BASE / "cycle2_conflict_memory_snapshots/pre_cycle02.npz",
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
        default=DEFAULT_BASE / "conflict_pcgrad_audit",
    )
    parser.add_argument("--expected-query-count", type=int, default=1_978)
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


def _load_label_free(path: Path) -> dict[str, np.ndarray]:
    required = {
        "cycle",
        "phase",
        "sample_index",
        "label_mask",
        "source_label",
        "clip_label",
        "task_prob",
        "clip_prob",
        "strong_task_prob",
    }
    with np.load(path, allow_pickle=False) as snapshot:
        missing = required.difference(snapshot.files)
        if missing:
            raise RuntimeError(f"Snapshot is missing keys: {sorted(missing)}")
        if "target_label" not in snapshot.files:
            raise RuntimeError("Snapshot is missing its later oracle diagnostic")
        return {key: np.asarray(snapshot[key]).copy() for key in required}


def _comparison(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != candidate["first_order"].shape:
        raise ValueError("comparison mask shape changed")
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
        "# VisDA Conflict PCGrad Output-Space Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The cycle-2 snapshot was read without target labels. Label-free",
        "component gradients and PCGrad decisions were SHA256-locked before",
        "the target list was parsed for oracle diagnostics.",
        "",
        "## What changed",
        "",
        "Only negative-dot consistency/CLIP-KL output directions receive the",
        "deterministic symmetric two-objective PCGrad projection. Pseudo labels,",
        "CLIP targets, masks, coefficients, and all non-conflicting rows stay fixed.",
        "",
        "## Label-free diagnostics",
        "",
        f"- Query samples: `{summary['label_free_metrics']['query_samples']}`.",
        f"- Gradient-conflict coverage: `{summary['label_free_metrics']['gradient_conflict_coverage_pct']:.6f}%`.",
        f"- Minimum underflow-floor decision stability: `{summary['label_free_metrics']['minimum_floor_decision_stability_pct']:.6f}%`.",
        "",
        "## Oracle diagnostic",
        "",
        f"- Overall first-order delta: `{metrics['overall_first_order']['mean_difference']:.9f}`.",
        f"- Overall 95% CI: `{metrics['overall_first_order']['paired_bootstrap_95_ci']}`.",
        f"- Conflicting-subset first-order delta: `{metrics['gradient_conflict_first_order']['mean_difference']:.9f}`.",
        f"- Conflicting-subset 95% CI: `{metrics['gradient_conflict_first_order']['paired_bootstrap_95_ci']}`.",
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
            "`NEEDS_PARAMETER_AUDIT` still does not authorize a proxy. Output-logit",
            "projection ignores the shared network Jacobian, so the next step would",
            "be a fixed no-update parameter-gradient micro-audit. `REJECT` closes",
            "this route without GPU work.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (args.snapshot, args.source_lock, args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing PCGrad audit input: {path}")
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
    source = _load_label_free(args.snapshot)
    cycle = int(np.asarray(source["cycle"]).item())
    phase = str(np.asarray(source["phase"]).item())
    sample_index = np.asarray(source["sample_index"], dtype=np.int64)
    label_mask = np.asarray(source["label_mask"], dtype=bool)
    task_label = np.asarray(source["source_label"], dtype=np.int64)
    clip_label = np.asarray(source["clip_label"], dtype=np.int64)
    task_probability = normalize_probability_matrix(
        source["task_prob"], name="task_probability"
    )
    strong_probability = normalize_probability_matrix(
        source["strong_task_prob"], name="strong_probability"
    )
    clip_probability = normalize_probability_matrix(
        source["clip_prob"], name="clip_probability"
    )
    query_mask = (~label_mask) & (task_label != clip_label)
    query = np.flatnonzero(query_mask)

    floor_results: dict[str, dict[str, np.ndarray]] = {}
    for name, floor in PROBABILITY_FLOORS.items():
        components = duet_output_descent_components(
            task_probability[query],
            strong_probability[query],
            clip_probability[query],
            consistency_weight=CONSISTENCY_WEIGHT,
            clip_weight=CLIP_WEIGHT,
            probability_floor=floor,
        )
        surgery = symmetric_pcgrad(
            components["consistency_joint"], components["clip_joint"]
        )
        floor_results[name] = {**components, **surgery}
    primary = floor_results[PRIMARY_FLOOR]
    floor_stability = {
        name: decision_stability(
            primary["gradient_conflict"], result["gradient_conflict"]
        )
        for name, result in floor_results.items()
        if name != PRIMARY_FLOOR
    }
    minimum_floor_stability = min(floor_stability.values(), default=100.0)
    floor_direction_stability = {
        name: direction_stability(
            primary["candidate_joint"], result["candidate_joint"]
        )
        for name, result in floor_results.items()
        if name != PRIMARY_FLOOR
    }
    minimum_floor_direction_stability = min(
        floor_direction_stability.values(), default=100.0
    )
    primary_candidate_mean_norm = float(
        np.linalg.norm(primary["candidate_joint"], axis=1).mean()
    )
    floor_candidate_mean_norm_ratio = {
        name: float(
            np.linalg.norm(result["candidate_joint"], axis=1).mean()
            / primary_candidate_mean_norm
            if primary_candidate_mean_norm > 0.0
            else 1.0
        )
        for name, result in floor_results.items()
        if name != PRIMARY_FLOOR
    }
    max_floor_norm_ratio_deviation = max(
        (abs(ratio - 1.0) for ratio in floor_candidate_mean_norm_ratio.values()),
        default=0.0,
    )
    baseline = primary["baseline_joint"]
    candidate = primary["candidate_joint"]
    gradient_conflict = primary["gradient_conflict"]
    baseline_norm = np.linalg.norm(baseline, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    mean_norm_ratio = float(
        candidate_norm.mean() / baseline_norm.mean()
        if baseline_norm.mean() > 0.0
        else np.inf
    )
    class_count = EXPECTED_CLASSES
    changed = np.linalg.norm(candidate - baseline, axis=1) > 1e-14
    projected_component_changed = (
        np.linalg.norm(
            primary["first_projected"] - primary["consistency_joint"], axis=1
        )
        > 1e-14
    ) | (
        np.linalg.norm(primary["second_projected"] - primary["clip_joint"], axis=1)
        > 1e-14
    )
    input_checks = {
        "snapshot_matches_cycle2_memory_lock": (
            snapshot_sha256
            == source_lock.get("inputs", {}).get("pre_cycle2_sha256")
        ),
        "source_lock_declares_labels_after_manifest": bool(
            source_lock.get("labels_read_after_this_manifest")
        ),
        "pre_cycle2_snapshot": cycle == EXPECTED_CYCLE and phase == "pre_cycle",
        "expected_proxy_sample_count": sample_index.shape == (EXPECTED_SAMPLES,),
        "sample_indices_are_proxy_order": np.array_equal(
            sample_index, np.arange(EXPECTED_SAMPLES)
        ),
        "probability_shapes_match": (
            task_probability.shape
            == strong_probability.shape
            == clip_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "saved_predictions_match_probabilities": (
            np.array_equal(task_label, task_probability.argmax(1))
            and np.array_equal(clip_label, clip_probability.argmax(1))
        ),
        "unresolved_conflicts_are_not_admitted": bool((~label_mask[query]).all()),
        "expected_query_count": query.size == args.expected_query_count,
        "projected_components_change_exactly_negative_dot_rows": np.array_equal(
            projected_component_changed, gradient_conflict
        ),
        "candidate_changes_only_on_negative_dot_rows": bool(
            (~changed | gradient_conflict).all()
        ),
        "baseline_weak_branch_zero_sum": np.allclose(
            baseline[:, :class_count].sum(1), 0.0, atol=1e-10
        ),
        "baseline_strong_branch_zero_sum": np.allclose(
            baseline[:, class_count:].sum(1), 0.0, atol=1e-10
        ),
        "candidate_weak_branch_zero_sum": np.allclose(
            candidate[:, :class_count].sum(1), 0.0, atol=1e-10
        ),
        "candidate_strong_branch_zero_sum": np.allclose(
            candidate[:, class_count:].sum(1), 0.0, atol=1e-10
        ),
        "loss_weights_match_released_duet": (
            CONSISTENCY_WEIGHT == 0.2 and CLIP_WEIGHT == 0.4
        ),
    }
    input_checks = {name: bool(passed) for name, passed in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"PCGrad label-free input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_pcgrad"
    signal_path = args.output_dir / f"{stem}_label_free.npz"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"
    np.savez_compressed(
        signal_path,
        query_index=query,
        weak_probability=task_probability[query].astype(np.float32),
        strong_probability=strong_probability[query].astype(np.float32),
        clip_probability=clip_probability[query].astype(np.float32),
        consistency_joint=primary["consistency_joint"].astype(np.float32),
        clip_joint=primary["clip_joint"].astype(np.float32),
        baseline_joint=baseline.astype(np.float32),
        candidate_joint=candidate.astype(np.float32),
        component_cosine=primary["component_cosine"].astype(np.float32),
        gradient_conflict=gradient_conflict,
        candidate_changed=changed,
    )
    label_free_metrics = {
        "query_samples": int(query.size),
        "query_fraction_of_proxy_pct": float(query.size / EXPECTED_SAMPLES * 100.0),
        "gradient_conflict_samples": int(gradient_conflict.sum()),
        "gradient_conflict_coverage_pct": float(
            gradient_conflict.mean() * 100.0
        ),
        "component_cosine_quantiles": {
            str(quantile): float(np.quantile(primary["component_cosine"], quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "probability_zero_counts": {
            "weak": int((task_probability[query] == 0.0).sum()),
            "strong": int((strong_probability[query] == 0.0).sum()),
            "clip": int((clip_probability[query] == 0.0).sum()),
        },
        "probability_floors": PROBABILITY_FLOORS,
        "floor_decision_stability_pct": floor_stability,
        "minimum_floor_decision_stability_pct": minimum_floor_stability,
        "floor_candidate_direction_stability_pct": floor_direction_stability,
        "minimum_floor_candidate_direction_stability_pct": (
            minimum_floor_direction_stability
        ),
        "floor_candidate_mean_norm_ratio": floor_candidate_mean_norm_ratio,
        "max_floor_candidate_mean_norm_ratio_deviation": (
            max_floor_norm_ratio_deviation
        ),
        "baseline_mean_descent_norm": float(baseline_norm.mean()),
        "candidate_mean_descent_norm": float(candidate_norm.mean()),
        "candidate_to_baseline_mean_norm_ratio": mean_norm_ratio,
    }
    lock = {
        "phase": "LABEL_FREE_CONFLICT_PCGRAD_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "labels_read_after_this_manifest": True,
        "candidate": "symmetric_two_objective_pcgrad_in_weak_strong_output_logit_space",
        "candidate_contract": {
            "scope": "cycle2_currently_unresolved_task_clip_top1_conflicts",
            "objectives": ["legacy_bidirectional_consistency", "clip_kl"],
            "activation": "component_dot_product_less_than_zero",
            "thresholds_fitted": False,
            "hyperparameters_added": False,
            "pseudo_labels_changed": False,
            "clip_targets_changed": False,
            "admission_mask_changed": False,
            "loss_coefficients_changed": False,
            "consistency_weight": CONSISTENCY_WEIGHT,
            "clip_weight": CLIP_WEIGHT,
            "space": "concatenated_weak_strong_output_logits",
            "parameter_jacobian_included": False,
        },
        "primary_reference": (
            "https://papers.nips.cc/paper/2020/hash/"
            "3fe78a8acf5fda99de95303940a2420c-Abstract.html"
        ),
        "inputs": {
            "snapshot_path": str(args.snapshot),
            "snapshot_sha256": snapshot_sha256,
            "source_lock_path": str(args.source_lock),
            "source_lock_sha256": _sha256(args.source_lock),
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "signal_path": str(signal_path),
        "signal_sha256": _sha256(signal_path),
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Oracle phase: labels are revealed only after the label-free signal lock.
    labels_from_list = _parse_labels_after_lock(args.target_list)
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        labels_from_snapshot = np.asarray(
            snapshot["target_label"], dtype=np.int64
        ).copy()
    if not np.array_equal(labels_from_list, labels_from_snapshot):
        raise RuntimeError("Target-list and snapshot oracle labels disagree")
    labels = labels_from_list[query]
    oracle = np.concatenate(
        (
            oracle_ce_logit_descent(task_probability[query], labels),
            oracle_ce_logit_descent(strong_probability[query], labels),
        ),
        axis=1,
    )
    baseline_alignment = rowwise_oracle_alignment(baseline, oracle)
    candidate_alignment = rowwise_oracle_alignment(candidate, oracle)
    all_rows = np.ones(query.size, dtype=bool)
    overall_comparison = _comparison(
        candidate_alignment,
        baseline_alignment,
        all_rows,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    conflict_comparison = _comparison(
        candidate_alignment,
        baseline_alignment,
        gradient_conflict,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 1,
    )
    delta = candidate_alignment["first_order"] - baseline_alignment["first_order"]
    class_rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_index
        class_delta = delta[mask]
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "samples": int(mask.sum()),
                "gradient_conflict_samples": int(gradient_conflict[mask].sum()),
                "gradient_conflict_coverage_pct": float(
                    gradient_conflict[mask].mean() * 100.0
                ),
                "baseline_mean_first_order": float(
                    baseline_alignment["first_order"][mask].mean()
                ),
                "candidate_mean_first_order": float(
                    candidate_alignment["first_order"][mask].mean()
                ),
                "candidate_minus_baseline_mean_first_order": float(
                    class_delta.mean()
                ),
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
                "task_top1": int(task_label[index]),
                "clip_top1": int(clip_label[index]),
                "component_cosine": float(primary["component_cosine"][position]),
                "gradient_conflict": bool(gradient_conflict[position]),
                "baseline_first_order": float(
                    baseline_alignment["first_order"][position]
                ),
                "candidate_first_order": float(
                    candidate_alignment["first_order"][position]
                ),
                "candidate_minus_baseline_first_order": float(delta[position]),
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
    gate = evaluate_conflict_pcgrad_gate(
        input_contract_valid=all(input_checks.values()),
        conflict_coverage_pct=label_free_metrics["gradient_conflict_coverage_pct"],
        floor_decision_stability_pct=minimum_floor_stability,
        floor_direction_stability_pct=minimum_floor_direction_stability,
        floor_mean_norm_ratio_max_deviation=max_floor_norm_ratio_deviation,
        overall_first_order=overall_comparison,
        conflict_first_order=conflict_comparison,
        baseline_negative_burden=baseline_negative_burden,
        candidate_negative_burden=candidate_negative_burden,
        helpful_retention_pct=helpful_retention,
        candidate_to_baseline_mean_norm_ratio=mean_norm_ratio,
        group_first_order_delta=group_delta,
    )
    summary = {
        "dataset": "VISDA-C",
        "seed": args.seed,
        "cycle": EXPECTED_CYCLE,
        "decision": gate["decision"],
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "candidate": lock["candidate"],
        "candidate_contract": lock["candidate_contract"],
        "input_contract": {"passed": all(input_checks.values()), "checks": input_checks},
        "label_free_metrics": label_free_metrics,
        "oracle_metrics": {
            "overall_first_order": overall_comparison,
            "gradient_conflict_first_order": conflict_comparison,
            "baseline_negative_burden": baseline_negative_burden,
            "candidate_negative_burden": candidate_negative_burden,
            "helpful_first_order_retention_pct": helpful_retention,
            "group_first_order_delta": group_delta,
            "classwise": class_rows,
        },
        "gate": gate,
        "training_authorized": False,
        "proxy_authorized": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "model_or_clip_loaded": False,
        "model_forward_calls": 0,
        "target_images_loaded": False,
        "scope_limit": (
            "Output-logit PCGrad does not include the shared network Jacobian. "
            "NEEDS_PARAMETER_AUDIT requests one fixed no-update parameter-gradient "
            "micro-audit only; it does not authorize proxy or full training."
        ),
        "next": (
            "design one fixed no-update parameter-gradient micro-audit"
            if gate["decision"] == "NEEDS_PARAMETER_AUDIT"
            else "close KL/consistency gradient-surgery route; do not start GPU"
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote label-free signal: {signal_path}")
    print(f"Locked signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
