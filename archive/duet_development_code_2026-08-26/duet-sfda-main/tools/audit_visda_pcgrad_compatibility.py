#!/usr/bin/env python
"""Offline audit of a parameter-free full-gradient PCGrad compatibility rule."""

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

from src.utils.pcgrad_compatibility import (
    compatibility_fraction_from_norms,
    reconstruct_fractional_metrics,
)
from src.utils.pcgrad_parameter_audit import (
    negative_burden,
    paired_metric_summary,
)


DEFAULT_PARAMETER_DIR = Path(
    "output/uda/VISDA-C/TV/plmatch_pcgrad_parameter_audit_seed2020/"
    "conflict_pcgrad_parameter_audit"
)
GROUPS = ("car", "person", "truck", "other_nine")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _vector(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype=np.float64)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["parameter_metrics"]
    comparisons = metrics["comparisons"]
    lines = [
        "# VisDA PCGrad Full-Gradient Compatibility Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The rejected raw PCGrad result remains rejected. This audit tests a new, "
        "parameter-free rule: retain only the non-negative projection fraction of "
        "the PCGrad correction along the complete DUET parameter gradient.",
        "",
        "Target labels are read only after the new fraction signal is locked. They "
        "are oracle diagnostics and never enter the rule.",
        "",
        "| Metric | Baseline | Candidate | Delta | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("cosine", "oracle_unit_projection", "first_order"):
        item = comparisons[name]
        low, high = item["paired_bootstrap_95_ci"]
        lines.append(
            f"| {name} | {item['baseline_mean']:.6f} | "
            f"{item['candidate_mean']:.6f} | {item['mean_difference']:.6f} | "
            f"[{low:.6f}, {high:.6f}] |"
        )
    lines.extend(
        [
            "",
            "| Group | First-order delta |",
            "|---|---:|",
        ]
    )
    for group, value in metrics["group_first_order_delta"].items():
        lines.append(f"| {group} | {value:.6f} |")
    lines.extend(
        [
            "",
            "This is exploratory mechanism evidence because the compatibility "
            "hypothesis was formed after diagnosing the rejected raw PCGrad run. "
            "A pass authorizes one matched proxy25 confirmation only, not full "
            "VisDA training or a seed sweep.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-dir", type=Path, default=DEFAULT_PARAMETER_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    parameter_dir = args.parameter_dir
    output_dir = args.output_dir or parameter_dir / "pcgrad_compatibility_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_summary_path = parameter_dir / "visda_conflict_pcgrad_parameter_summary.json"
    prior_lock_path = parameter_dir / "visda_conflict_pcgrad_parameter_signal_lock.json"
    batch_lock_dir = parameter_dir / "batch_signal_locks"
    batch_oracle_path = parameter_dir / "visda_conflict_pcgrad_parameter_oracle_diagnostic.csv"
    group_oracle_path = parameter_dir / "visda_conflict_pcgrad_parameter_groupwise_oracle_diagnostic.csv"
    required = (
        prior_summary_path,
        prior_lock_path,
        batch_lock_dir,
        batch_oracle_path,
        group_oracle_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing compatibility-audit inputs: {missing}")

    prior = json.loads(prior_summary_path.read_text())
    prior_lock = json.loads(prior_lock_path.read_text())
    batch_lock_paths = sorted(batch_lock_dir.glob("*.json"))
    label_free_checks = {
        "prior_parameter_audit_remains_reject": prior.get("decision") == "REJECT",
        "prior_input_contract_passed": prior.get("input_contract", {}).get("passed") is True,
        "prior_labels_locked": prior.get(
            "target_labels_used_only_for_post_lock_parameter_alignment"
        )
        is True,
        "prior_selection_lock_hash_matches": _sha256(prior_lock_path)
        == prior.get("artifacts", {}).get("selection_lock_sha256"),
        "prior_selection_is_label_free": prior_lock.get("contains_target_labels") is False
        and prior_lock.get("selection_uses_target_labels") is False,
        "ten_label_free_batch_locks": len(batch_lock_paths) == 10,
    }
    if not all(label_free_checks.values()):
        failed = [name for name, passed in label_free_checks.items() if not passed]
        raise RuntimeError(f"Label-free compatibility input contract failed: {failed}")

    label_free_rows: list[dict[str, Any]] = []
    for expected_batch, path in enumerate(batch_lock_paths, start=1):
        lock = json.loads(path.read_text())
        metrics = lock.get("label_free_gradient_metrics", {})
        if not (
            lock.get("batch") == expected_batch
            and lock.get("contains_target_labels") is False
            and lock.get("labels_read_after_this_manifest") is True
            and set(
                (
                    "baseline_parameter_gradient_norm",
                    "candidate_parameter_gradient_norm",
                    "correction_parameter_gradient_norm",
                )
            ).issubset(metrics)
        ):
            raise RuntimeError(f"Invalid prior batch signal lock: {path}")
        label_free_rows.append(
            {
                "batch": expected_batch,
                "baseline_norm": float(metrics["baseline_parameter_gradient_norm"]),
                "candidate_norm": float(metrics["candidate_parameter_gradient_norm"]),
                "correction_norm": float(metrics["correction_parameter_gradient_norm"]),
                "batch_lock_sha256": _sha256(path),
            }
        )

    baseline_norm = np.asarray(
        [row["baseline_norm"] for row in label_free_rows], dtype=np.float64
    )
    candidate_norm = np.asarray(
        [row["candidate_norm"] for row in label_free_rows], dtype=np.float64
    )
    correction_norm = np.asarray(
        [row["correction_norm"] for row in label_free_rows], dtype=np.float64
    )
    compatibility = compatibility_fraction_from_norms(
        baseline_norm, candidate_norm, correction_norm
    )
    fraction = compatibility["fraction"]
    stem = "visda_conflict_pcgrad_compatibility"
    signal_path = output_dir / f"{stem}_label_free.npz"
    lock_path = output_dir / f"{stem}_signal_lock.json"
    np.savez_compressed(
        signal_path,
        batch=np.arange(1, 11, dtype=np.int64),
        baseline_correction_dot=compatibility["baseline_correction_dot"],
        correction_norm_sq=compatibility["correction_norm_sq"],
        correction_fraction=fraction,
        batch_signal_lock_sha256=np.asarray(
            [row["batch_lock_sha256"] for row in label_free_rows], dtype="U64"
        ),
    )
    signal_lock = {
        "phase": "LABEL_FREE_PCGRAD_FULL_GRADIENT_COMPATIBILITY_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "labels_read_only_after_this_lock": True,
        "hypothesis_provenance": (
            "posthoc_mechanism_hypothesis_from_rejected_raw_pcgrad_audit"
        ),
        "confirmation_status": "exploratory_until_one_matched_proxy25",
        "candidate_contract": {
            "active_cycle": 2,
            "scope": "currently_unresolved_task_clip_conflicts",
            "raw_correction": "rowwise_symmetric_consistency_clip_PCGrad",
            "fraction_rule": (
                "clip(dot(full_duet_gradient,pcgrad_correction)/"
                "norm2(pcgrad_correction),0,1)"
            ),
            "target_labels_used_by_rule": False,
            "class_specific_route": False,
            "thresholds_fitted": False,
            "hyperparameters_added": 0,
            "unchanged_cycles": [1, 3, 4],
            "unchanged": [
                "pseudo_labels",
                "admission_mask",
                "clip_target",
                "loss_coefficients",
                "optimizer",
                "weak_strong_views",
            ],
        },
        "inputs": {
            "rejected_parameter_summary_sha256": _sha256(prior_summary_path),
            "prior_selection_lock_sha256": _sha256(prior_lock_path),
            "batch_signal_lock_sha256": [
                row["batch_lock_sha256"] for row in label_free_rows
            ],
        },
        "signal_path": str(signal_path),
        "signal_sha256": _sha256(signal_path),
    }
    lock_path.write_text(json.dumps(signal_lock, indent=2) + "\n")
    print(f"Wrote label-free compatibility fraction: {signal_path}")
    print(f"Locked fraction before oracle labels: {lock_path}")

    # Oracle diagnostic begins only after the new label-free signal is locked.
    if _sha256(batch_oracle_path) != prior.get("artifacts", {}).get(
        "batch_oracle_csv_sha256"
    ):
        raise RuntimeError("Prior batch oracle CSV hash changed")
    if _sha256(group_oracle_path) != prior.get("artifacts", {}).get(
        "group_oracle_csv_sha256"
    ):
        raise RuntimeError("Prior group oracle CSV hash changed")
    batch_rows = _read_csv(batch_oracle_path)
    group_rows = _read_csv(group_oracle_path)
    if len(batch_rows) != 10:
        raise RuntimeError("Expected ten locked parameter batches")
    for row, locked in zip(batch_rows, label_free_rows):
        if not (
            int(row["batch"]) == locked["batch"]
            and row["batch_signal_lock_sha256"] == locked["batch_lock_sha256"]
        ):
            raise RuntimeError("Oracle row does not match its prior batch lock")

    reconstructed = reconstruct_fractional_metrics(
        fraction=fraction,
        baseline_norm=baseline_norm,
        candidate_norm=candidate_norm,
        correction_norm=correction_norm,
        baseline_unit_projection=_vector(batch_rows, "baseline_oracle_unit_projection"),
        candidate_unit_projection=_vector(batch_rows, "candidate_oracle_unit_projection"),
        baseline_first_order=_vector(batch_rows, "baseline_first_order"),
        candidate_first_order=_vector(batch_rows, "candidate_first_order"),
    )
    baseline_cosine = _vector(batch_rows, "baseline_cosine")
    candidate_cosine = reconstructed["cosine"]
    candidate_cosine[fraction == 0.0] = baseline_cosine[fraction == 0.0]
    comparisons = {
        "cosine": paired_metric_summary(
            candidate_cosine, baseline_cosine, seed=2120
        ),
        "oracle_unit_projection": paired_metric_summary(
            reconstructed["oracle_unit_projection"],
            _vector(batch_rows, "baseline_oracle_unit_projection"),
            seed=2121,
        ),
        "first_order": paired_metric_summary(
            reconstructed["first_order"],
            _vector(batch_rows, "baseline_first_order"),
            seed=2122,
        ),
    }
    raw_comparisons = prior["parameter_metrics"]["comparisons"]
    baseline_first = _vector(batch_rows, "baseline_first_order")
    candidate_first = reconstructed["first_order"]
    baseline_positive = np.maximum(baseline_first, 0.0).sum()
    helpful_retention = float(
        np.maximum(candidate_first, 0.0).sum() / baseline_positive * 100.0
    )
    norm_ratio = reconstructed["norm"] / baseline_norm
    group_delta: dict[str, float] = {}
    for group in GROUPS:
        selected = [row for row in group_rows if row["group"] == group]
        if not selected:
            raise RuntimeError(f"Oracle group absent from audit: {group}")
        samples = np.asarray([int(row["samples"]) for row in selected])
        delta = np.asarray(
            [
                float(row["candidate_minus_baseline_first_order"])
                * fraction[int(row["batch"]) - 1]
                for row in selected
            ],
            dtype=np.float64,
        )
        group_delta[group] = float(np.average(delta, weights=samples))

    def positive_with_ci(name: str) -> bool:
        item = comparisons[name]
        return bool(
            item["mean_difference"] > 0.0
            and item["paired_bootstrap_95_ci"][0] > 0.0
        )

    checks = {
        "input_contract_valid": all(label_free_checks.values()),
        "raw_pcgrad_reject_preserved": prior.get("decision") == "REJECT",
        "ten_locked_batches_accounted_for": len(batch_rows) == 10,
        "fraction_has_no_labels_thresholds_or_hyperparameters": (
            signal_lock["candidate_contract"]["target_labels_used_by_rule"] is False
            and signal_lock["candidate_contract"]["thresholds_fitted"] is False
            and signal_lock["candidate_contract"]["hyperparameters_added"] == 0
        ),
        "correction_is_selective_not_degenerate": bool(
            0 < np.count_nonzero(fraction > 0.0) < fraction.size
        ),
        "cosine_gain_ci_lower_positive": positive_with_ci("cosine"),
        "projection_gain_ci_lower_positive": positive_with_ci(
            "oracle_unit_projection"
        ),
        "first_order_gain_ci_lower_positive": positive_with_ci("first_order"),
        "beats_raw_pcgrad_on_all_three_mean_deltas": all(
            comparisons[name]["mean_difference"]
            > raw_comparisons[name]["mean_difference"]
            for name in ("cosine", "oracle_unit_projection", "first_order")
        ),
        "candidate_negative_burden_not_worse": negative_burden(candidate_first)
        >= negative_burden(baseline_first),
        "helpful_first_order_retention_at_least_99pct": helpful_retention >= 99.0,
        "mean_descent_norm_inflation_at_most_1_5x": float(norm_ratio.mean())
        <= 1.5,
        "car_first_order_delta_nonnegative": group_delta["car"] >= 0.0,
        "person_first_order_delta_nonnegative": group_delta["person"] >= 0.0,
        "truck_first_order_delta_nonnegative": group_delta["truck"] >= 0.0,
        "other_nine_first_order_delta_nonnegative": group_delta["other_nine"]
        >= 0.0,
    }
    passed = all(checks.values())
    decision = "PASS_EXPLORATORY_COMPATIBILITY_PREFLIGHT" if passed else "REJECT"
    summary = {
        "dataset": "VISDA-C",
        "seed": 2020,
        "decision": decision,
        "oracle_diagnostic": True,
        "labels_used_only_after_compatibility_signal_lock": True,
        "raw_pcgrad_decision_unchanged": "REJECT",
        "candidate": "cycle2_full_gradient_compatibility_fraction_for_conflict_pcgrad",
        "hypothesis_provenance": signal_lock["hypothesis_provenance"],
        "parameter_metrics": {
            "batches": 10,
            "nonzero_fraction_batches": int(np.count_nonzero(fraction > 0.0)),
            "mean_correction_fraction": float(fraction.mean()),
            "correction_fraction": [float(value) for value in fraction],
            "comparisons": comparisons,
            "raw_pcgrad_comparisons": raw_comparisons,
            "baseline_negative_burden": negative_burden(baseline_first),
            "candidate_negative_burden": negative_burden(candidate_first),
            "helpful_first_order_retention_pct": helpful_retention,
            "candidate_to_baseline_mean_norm_ratio": float(norm_ratio.mean()),
            "group_first_order_delta": group_delta,
        },
        "gate": {
            "decision": decision,
            "checks": checks,
            "matched_proxy_authorized": passed,
            "full_training_authorized": False,
            "seed_sweep_authorized": False,
        },
        "matched_proxy_authorized": passed,
        "full_training_authorized": False,
        "seed_sweep_authorized": False,
        "estimated_next_cost": (
            "one matched proxy25 run; four DUET cycles with one extra backward "
            "only in cycle 2; expected about 40-50 GPU minutes"
        ),
        "next": (
            "run exactly one matched proxy25 confirmation"
            if passed
            else "stop compatibility-controlled PCGrad without GPU work"
        ),
        "artifacts": {
            "signal_sha256": _sha256(signal_path),
            "signal_lock_sha256": _sha256(lock_path),
            "prior_parameter_summary_sha256": _sha256(prior_summary_path),
            "batch_oracle_csv_sha256": _sha256(batch_oracle_path),
            "group_oracle_csv_sha256": _sha256(group_oracle_path),
        },
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, summary_path.with_suffix(".md"))
    print(json.dumps({"decision": decision, "checks": checks}, indent=2))
    print(f"Wrote compatibility summary: {summary_path}")


if __name__ == "__main__":
    main()
