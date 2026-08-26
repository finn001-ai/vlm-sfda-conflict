#!/usr/bin/env python3
"""CPU-only exploratory risk control for locked patch-to-CLS rescues.

Phase 1 reads only the prior label-free NPZ and lock.  It keeps the upper
median of stable full-head margins, then greedily enforces the already declared
1% full-proxy pseudo-class mass cap.  The prior oracle files and summary are
read strictly after the new signal lock.  This same-proxy analysis is
exploratory: even PASS authorizes only one independent held-out full-target
audit, never a parameter audit, proxy run, or training.
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

from src.utils.conflict_boundary import paired_accuracy_bootstrap_ci  # noqa: E402
from src.utils.patch_cls_risk_control_audit import (  # noqa: E402
    evaluate_patch_cls_risk_control_gate,
    select_upper_median_mass_capped_rescues,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CONFLICTS = 7_070
EXPECTED_CLASSES = 12
MAX_CLASS_MASS_SHIFT_FRACTION = 0.01
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
    "duet_support_conditioned_clip_cycle2_memory_audit_seed2020/"
    "patch_cls_contribution_audit"
)
STEM = "visda_conflict_patch_cls_risk_control"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-signal",
        type=Path,
        default=DEFAULT_BASE / "visda_conflict_patch_cls_contribution_label_free.npz",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_BASE / "visda_conflict_patch_cls_contribution_signal_lock.json",
    )
    parser.add_argument(
        "--source-oracle",
        type=Path,
        default=DEFAULT_BASE
        / "visda_conflict_patch_cls_contribution_oracle_diagnostic.csv",
    )
    parser.add_argument(
        "--source-classwise",
        type=Path,
        default=(
            DEFAULT_BASE
            / "visda_conflict_patch_cls_contribution_classwise_oracle_diagnostic.csv"
        ),
    )
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=DEFAULT_BASE / "visda_conflict_patch_cls_contribution_summary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_BASE / "risk_control_audit"
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


def _comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    candidate_correct = candidate == labels
    baseline_correct = baseline == labels
    interval = paired_accuracy_bootstrap_ci(
        candidate_correct, baseline_correct, repeats=repeats, seed=seed
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


def _read_oracle_after_lock(path: Path, query_index: np.ndarray) -> np.ndarray:
    labels: list[int] = []
    indices: list[int] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("oracle_usage") != "diagnostic_only_after_label_free_lock":
                raise ValueError("source oracle row lacks diagnostic-only provenance")
            indices.append(int(row["sample_index"]))
            labels.append(int(row["oracle_target_label"]))
    observed_index = np.asarray(indices, dtype=np.int64)
    result = np.asarray(labels, dtype=np.int64)
    if not np.array_equal(observed_index, query_index):
        raise ValueError("source oracle order differs from locked query order")
    if result.shape != (EXPECTED_CONFLICTS,):
        raise ValueError("source oracle has an unexpected number of rows")
    if np.any(result < 0) or np.any(result >= EXPECTED_CLASSES):
        raise ValueError("oracle label outside class range")
    return result


def _read_class_sizes_after_lock(path: Path) -> np.ndarray:
    sizes = np.zeros(EXPECTED_CLASSES, dtype=np.int64)
    names: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_index = int(row["class_index"])
            sizes[class_index] = int(row["full_proxy_samples"])
            names.append(row["class"])
    if tuple(names) != CLASS_NAMES or np.any(sizes <= 0):
        raise ValueError("source classwise file violates the VisDA class contract")
    if int(sizes.sum()) != EXPECTED_SAMPLES:
        raise ValueError("source class sizes do not sum to the proxy sample count")
    return sizes


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    oracle = summary["oracle_diagnostic"]
    best_name = oracle["best_baseline_name"]
    best = oracle["comparisons"][best_name]
    lines = [
        "# VisDA Patch-to-CLS Exploratory Risk-Control Audit",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Selected conflict coverage | "
            f"`{summary['label_free_metrics']['selected_coverage_pct']:.6f}%` "
            "| New label-free lock |"
        ),
        (
            "| Maximum full-proxy class-mass shift | "
            f"`{summary['label_free_metrics']['max_class_mass_shift_pp']:.6f}` pp "
            "| New label-free lock |"
        ),
        (
            f"| Conflict gain vs `{best_name}` | `{best['gain_pp']:.6f}` pp; "
            f"CI `{best['paired_bootstrap_95_ci_pp']}` "
            "| Oracle diagnostic after lock |"
        ),
        (
            "| Full-proxy macro gain | "
            f"`{oracle['full_proxy_macro_gain_pp']:.6f}` pp "
            "| Oracle diagnostic after lock |"
        ),
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "## Label-free rule",
        "",
        "Among the prior full/even/odd-head unanimous positive rescues, retain",
        "only samples whose full-head margin is at least the median of that",
        "stable set. Visit candidates in descending margin order and accept a",
        "task rescue only while every task-minus-CLIP pseudo-class count shift",
        "remains within 1% of the full proxy. No target label, class route,",
        "searched fraction, optimizer, model forward, or training enters.",
        "",
        "## Confirmatory limitation",
        "",
        "This rule was designed after inspecting mechanism diagnostics on the",
        "same proxy. Therefore PASS is exploratory and authorizes only one",
        "independent held-out full-target audit. It cannot authorize a parameter",
        "audit, proxy experiment, or training.",
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for required in (
        args.source_signal,
        args.source_lock,
        args.source_oracle,
        args.source_classwise,
        args.source_summary,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    source_lock = json.loads(args.source_lock.read_text())
    if source_lock.get("contains_target_labels", True):
        raise RuntimeError("source signal lock is not label-free")

    # Phase 1: read only locked label-free arrays.  Oracle artifacts are only
    # hashed opaquely until the new signal lock has been written.
    with np.load(args.source_signal, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    query_index = np.asarray(arrays["query_index"], dtype=np.int64)
    task_candidate = np.asarray(arrays["task_candidate"], dtype=np.int64)
    clip_candidate = np.asarray(arrays["clip_candidate"], dtype=np.int64)
    full_margin = np.asarray(arrays["full_task_peak"], dtype=np.float64) - np.asarray(
        arrays["full_clip_peak"], dtype=np.float64
    )
    stable_rescue = np.asarray(arrays["rescue_task"], dtype=bool)
    result = select_upper_median_mass_capped_rescues(
        task_candidate,
        clip_candidate,
        full_margin,
        stable_rescue,
        full_sample_count=EXPECTED_SAMPLES,
        max_class_mass_shift_fraction=MAX_CLASS_MASS_SHIFT_FRACTION,
        class_count=EXPECTED_CLASSES,
    )
    expected_source_prediction = np.where(stable_rescue, task_candidate, clip_candidate)
    input_checks = {
        "source_lock_phase": (
            source_lock.get("phase") == "LABEL_FREE_VISDA_PATCH_CLS_CONTRIBUTION_LOCK"
        ),
        "source_signal_hash_matches_lock": (
            _sha256(args.source_signal)
            == source_lock.get("signal_npz", {}).get("sha256")
        ),
        "expected_rows": query_index.shape == (EXPECTED_CONFLICTS,),
        "query_indices_unique": np.unique(query_index).size == EXPECTED_CONFLICTS,
        "candidate_vectors_align": (
            task_candidate.shape == clip_candidate.shape == query_index.shape
        ),
        "task_clip_top1_conflict": bool(np.all(task_candidate != clip_candidate)),
        "source_prediction_reproduced": np.array_equal(
            np.asarray(arrays["candidate_prediction"], dtype=np.int64),
            expected_source_prediction,
        ),
        "source_stable_mask_reproduced": np.array_equal(
            stable_rescue,
            np.asarray(arrays["full_choose_task"], dtype=bool)
            & np.asarray(arrays["even_choose_task"], dtype=bool)
            & np.asarray(arrays["odd_choose_task"], dtype=bool),
        ),
        "source_mass_limit_reused_exactly": (
            float(
                source_lock.get("predeclared_gate", {}).get(
                    "max_class_mass_shift_pp", -1.0
                )
            )
            == 1.0
        ),
        "label_free_values_finite": bool(np.isfinite(full_margin).all()),
        "candidate_only_task_or_clip": bool(
            np.all(
                (result["prediction"] == task_candidate)
                | (result["prediction"] == clip_candidate)
            )
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Patch risk-control input contract failed: {failed}")

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
        full_task_margin=full_margin.astype(np.float32),
        source_stable_rescue=stable_rescue,
        upper_median=result["upper_median"],
        selected=result["selected"],
        rejected_by_mass_cap=result["rejected_by_mass_cap"],
        candidate_prediction=result["prediction"],
        median_threshold=np.asarray(result["threshold"], dtype=np.float64),
        class_count_shift=result["class_count_shift"],
        fixed_task_prediction=np.asarray(arrays["fixed_task_prediction"]),
        fixed_clip_prediction=np.asarray(arrays["fixed_clip_prediction"]),
        confidence_prediction=np.asarray(arrays["confidence_prediction"]),
        arithmetic_prediction=np.asarray(arrays["arithmetic_prediction"]),
        rms_prediction=np.asarray(arrays["rms_prediction"]),
    )
    class_mass_shift_pp = result["class_mass_shift_fraction"] * 100.0
    label_free_metrics = {
        "source_stable_rescues": int(stable_rescue.sum()),
        "upper_median_candidates": int(result["upper_median"].sum()),
        "selected_samples": int(result["selected"].sum()),
        "selected_coverage_pct": float(result["selected"].mean() * 100.0),
        "rejected_by_mass_cap": int(result["rejected_by_mass_cap"].sum()),
        "median_full_head_margin": float(result["threshold"]),
        "class_count_cap": int(result["count_cap"]),
        "class_count_shift": result["class_count_shift"].tolist(),
        "class_mass_shift_pp": {
            name: float(class_mass_shift_pp[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "max_class_mass_shift_pp": float(np.abs(class_mass_shift_pp).max()),
    }
    lock = {
        "phase": "LABEL_FREE_VISDA_PATCH_CLS_RISK_CONTROL_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_oracle_not_read_before_lock": True,
        "source_classwise_oracle_not_read_before_lock": True,
        "source_summary_not_read_before_lock": True,
        "oracle_artifacts_read_after_this_manifest": True,
        "confirmatory_status": "exploratory_same_proxy_requires_independent_holdout",
        "candidate_contract": {
            "source": "locked_patch_cls_full_even_odd_unanimous_positive_rescues",
            "confidence_control": "full_head_margin_at_or_above_stable_set_median",
            "class_mass_control": "descending_margin_greedy_acceptance_with_1pct_full_proxy_cap",
            "default_prediction": "fixed_clip_top1",
            "alternative_prediction": "fixed_task_top1",
            "searched_fraction": False,
            "numerical_margin_threshold": False,
            "class_specific_route": False,
            "fitted_target_labels": False,
            "training_change_in_this_audit": False,
        },
        "predeclared_gate": {
            "selected_coverage_pct": [2.0, 10.0],
            "min_paired_adjudication_precision_pct": 60.0,
            "min_accuracy_gain_vs_best_baseline_pp": 1.0,
            "best_baseline_paired_ci_lower": "> 0",
            "min_full_proxy_macro_gain_pp": 0.20,
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
            "max_class_mass_shift_pp": 1.0,
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
            "opaque_source_classwise_sha256": _sha256(args.source_classwise),
            "opaque_source_summary_sha256": _sha256(args.source_summary),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/patch_cls_risk_control_audit.py": _sha256(
                REPO_ROOT / "src/utils/patch_cls_risk_control_audit.py"
            ),
            "tools/audit_visda_conflict_patch_cls_risk_control.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: explicit oracle diagnostic after the new signal lock.
    for path, expected_hash in (
        (args.source_oracle, lock["inputs"]["opaque_source_oracle_sha256"]),
        (
            args.source_classwise,
            lock["inputs"]["opaque_source_classwise_sha256"],
        ),
        (args.source_summary, lock["inputs"]["opaque_source_summary_sha256"]),
    ):
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"Oracle artifact changed after lock: {path}")
    labels = _read_oracle_after_lock(args.source_oracle, query_index)
    full_class_sizes = _read_class_sizes_after_lock(args.source_classwise)
    source_summary = json.loads(args.source_summary.read_text())
    source_reject_preserved = source_summary.get("decision") == "REJECT"

    predictions = {
        "candidate": result["prediction"],
        "fixed_task": np.asarray(arrays["fixed_task_prediction"], dtype=np.int64),
        "fixed_clip": np.asarray(arrays["fixed_clip_prediction"], dtype=np.int64),
        "confidence_choice": np.asarray(
            arrays["confidence_prediction"], dtype=np.int64
        ),
        "arithmetic": np.asarray(arrays["arithmetic_prediction"], dtype=np.int64),
        "rms": np.asarray(arrays["rms_prediction"], dtype=np.int64),
    }
    comparison_order = (
        "fixed_task",
        "fixed_clip",
        "confidence_choice",
        "arithmetic",
        "rms",
    )
    comparisons = {
        name: _comparison(
            predictions["candidate"],
            predictions[name],
            labels,
            repeats=args.bootstrap_repeats,
            seed=args.seed + offset,
        )
        for offset, name in enumerate(comparison_order)
    }
    best_baseline_name = max(
        comparisons,
        key=lambda name: (comparisons[name]["baseline_accuracy_pct"], name),
    )
    selected = result["selected"]
    task_correct = predictions["fixed_task"] == labels
    clip_correct = predictions["fixed_clip"] == labels
    paired_resolved = selected & (task_correct | clip_correct)
    paired_adjudication_precision = float(task_correct[paired_resolved].mean() * 100.0)

    class_rows: list[dict[str, Any]] = []
    class_delta = np.empty(EXPECTED_CLASSES, dtype=np.float64)
    candidate_correct = predictions["candidate"] == labels
    best_correct = predictions[best_baseline_name] == labels
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_query = labels == class_index
        net_corrections = int(
            candidate_correct[class_query].sum() - best_correct[class_query].sum()
        )
        class_delta[class_index] = (
            net_corrections / float(full_class_sizes[class_index]) * 100.0
        )
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "full_proxy_samples": int(full_class_sizes[class_index]),
                "conflict_samples": int(class_query.sum()),
                "selected_samples": int((selected & class_query).sum()),
                "net_corrections_vs_best_baseline": net_corrections,
                "full_proxy_accuracy_delta_pp": float(class_delta[class_index]),
                "oracle_usage": "diagnostic_only_after_risk_control_signal_lock",
            }
        )
    _write_csv(class_path, class_rows)
    oracle_rows: list[dict[str, Any]] = []
    for row, global_index in enumerate(query_index):
        oracle_rows.append(
            {
                "sample_index": int(global_index),
                "oracle_target_label": int(labels[row]),
                "upper_median_candidate": bool(result["upper_median"][row]),
                "selected_by_risk_control": bool(selected[row]),
                "rejected_by_mass_cap": bool(result["rejected_by_mass_cap"][row]),
                "candidate_prediction": int(predictions["candidate"][row]),
                "candidate_correct": bool(candidate_correct[row]),
                "fixed_clip_correct": bool(clip_correct[row]),
                "candidate_minus_fixed_clip_correct": int(candidate_correct[row])
                - int(clip_correct[row]),
                "oracle_usage": "diagnostic_only_after_risk_control_signal_lock",
            }
        )
    _write_csv(oracle_path, oracle_rows)

    car_delta = float(class_delta[3])
    truck_delta = float(class_delta[11])
    car_truck_mean_delta = float((car_delta + truck_delta) / 2.0)
    other_indices = [index for index in range(EXPECTED_CLASSES) if index not in (3, 11)]
    other_ten_mean_delta = float(class_delta[other_indices].mean())
    full_proxy_macro_gain = float(class_delta.mean())
    gate = evaluate_patch_cls_risk_control_gate(
        input_contract_valid=all(input_checks.values()),
        source_reject_preserved=source_reject_preserved,
        selected_coverage_pct=label_free_metrics["selected_coverage_pct"],
        paired_adjudication_precision_pct=paired_adjudication_precision,
        comparisons=comparisons,
        full_proxy_macro_gain_pp=full_proxy_macro_gain,
        car_delta_pp=car_delta,
        truck_delta_pp=truck_delta,
        car_truck_mean_delta_pp=car_truck_mean_delta,
        other_ten_mean_delta_pp=other_ten_mean_delta,
        max_class_mass_shift_pp=label_free_metrics["max_class_mass_shift_pp"],
    )
    summary = {
        "decision": gate["decision"],
        "method": "patch_cls_upper_median_with_label_free_class_mass_risk_control",
        "confirmatory_status": "exploratory_same_proxy_requires_independent_holdout",
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "source_reject_preserved": source_reject_preserved,
            "selected_task_correct": int((selected & task_correct).sum()),
            "selected_clip_correct": int((selected & clip_correct).sum()),
            "selected_neither_correct": int(
                (selected & ~(task_correct | clip_correct)).sum()
            ),
            "paired_adjudication_precision_pct": paired_adjudication_precision,
            "comparisons": comparisons,
            "best_baseline_name": best_baseline_name,
            "full_proxy_macro_gain_pp": full_proxy_macro_gain,
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
        "safety": {
            "target_images_loaded": False,
            "model_or_checkpoint_loaded": False,
            "forward_calls": 0,
            "backward_calls": 0,
            "optimizer_constructed": False,
            "parameter_updates": 0,
            "parameter_audit_authorized": False,
            "proxy_authorized": False,
            "training_authorized": False,
        },
        "scope_limit": (
            "Exploratory PASS authorizes one independent held-out full-target "
            "audit only; it never authorizes parameter/proxy/full training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(
        json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2)
    )
    print(f"Wrote label-free risk-control signal: {signal_path}")
    print(f"Locked risk-control signal before oracle files: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
