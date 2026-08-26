#!/usr/bin/env python3
"""Finalize the matched pure-DUET PCGrad parameter preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.pcgrad_parameter_audit import (  # noqa: E402
    AUDIT_BATCH_COUNT,
    AUDITED_CONFLICTS,
    evaluate_exact_parameter_gate,
    negative_burden,
    paired_metric_summary,
)


ACCURACY_PATTERN = re.compile(
    r"Task:\s*TV,\s*Iter:(\d+)/(\d+);\s*Cycle:\s*(\d+)/(\d+);\s*"
    r"Accuracy\s*=\s*([0-9.]+)%"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--feature-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _cycle_one(path: Path) -> list[tuple[int, float]]:
    records = []
    for iteration, _maximum, cycle, _cycles, accuracy in ACCURACY_PATTERN.findall(
        path.read_text(errors="ignore")
    ):
        if int(cycle) == 1:
            records.append((int(iteration), float(accuracy)))
    return records


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["parameter_metrics"]
    lines = [
        "# VisDA Exact PCGrad Parameter Preflight",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "This run replayed exactly one pure arithmetic-DUET cycle, measured",
        "the complete ResNet101+bottleneck parameter direction at the cycle-2",
        "boundary, restored all buffers, and took zero cycle-2 optimizer steps.",
        "",
        "## Matched replay",
        "",
        f"- Maximum cycle-1 control error: `{summary['matched_control']['max_accuracy_error_pp']:.6f}` pp.",
        f"- Audited conflicts: `{summary['parameter_metrics']['audited_conflicts']}` / `{summary['parameter_metrics']['unresolved_conflicts']}`.",
        "",
        "## Oracle diagnostic",
        "",
        f"- First-order delta: `{metrics['comparisons']['first_order']['mean_difference']:.9f}`.",
        f"- First-order 95% CI: `{metrics['comparisons']['first_order']['paired_bootstrap_95_ci']}`.",
        f"- Cosine delta: `{metrics['comparisons']['cosine']['mean_difference']:.9f}`.",
        f"- Projection delta: `{metrics['comparisons']['oracle_unit_projection']['mean_difference']:.9f}`.",
        f"- Group deltas: `{metrics['group_first_order_delta']}`.",
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- {name}: `{value}`" for name, value in summary["gate"]["checks"].items()
    )
    lines.extend(
        [
            "",
            "PASS authorizes exactly one matched proxy25 experiment. It does",
            "not authorize a full VisDA run, seed sweep, or parallel candidates.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    raw_path = args.audit_dir / "visda_conflict_pcgrad_parameter_runtime_raw.json"
    batch_path = args.audit_dir / "visda_conflict_pcgrad_parameter_oracle_diagnostic.csv"
    group_path = (
        args.audit_dir
        / "visda_conflict_pcgrad_parameter_groupwise_oracle_diagnostic.csv"
    )
    lock_path = args.audit_dir / "visda_conflict_pcgrad_parameter_signal_lock.json"
    batch_lock_dir = args.audit_dir / "batch_signal_locks"
    for path in (
        raw_path,
        batch_path,
        group_path,
        lock_path,
        args.audit_log,
        args.control_log,
        args.control_summary,
        args.feature_summary,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing parameter-audit finalizer input: {path}")

    raw = json.loads(raw_path.read_text())
    feature = json.loads(args.feature_summary.read_text())
    control = json.loads(args.control_summary.read_text())
    batch_rows = _read_csv(batch_path)
    group_rows = _read_csv(group_path)
    batch_locks = sorted(batch_lock_dir.glob("*.json"))
    control_cycle = _cycle_one(args.control_log)
    audit_cycle = _cycle_one(args.audit_log)
    if len(control_cycle) != 4 or len(audit_cycle) != 4:
        raise RuntimeError("Matched control and audit must each expose four cycle-1 records")
    if [record[0] for record in control_cycle] != [record[0] for record in audit_cycle]:
        raise RuntimeError("Matched cycle-1 checkpoint iterations changed")
    cycle_errors = [
        abs(candidate[1] - baseline[1])
        for baseline, candidate in zip(control_cycle, audit_cycle)
    ]
    max_cycle_error = max(cycle_errors)

    input_checks = {
        "runtime_evidence_captured": raw.get("decision")
        == "EXACT_PARAMETER_EVIDENCE_CAPTURED",
        "feature_jacobian_gate_passed": feature.get("decision")
        == "NEEDS_EXACT_CONTROL_PARAMETER_AUDIT",
        "feature_labels_locked": bool(feature.get("labels_used_only_after_signal_lock")),
        "matched_control_final_is_87_93": control.get("final", {}).get("accuracy")
        == 87.93,
        "ten_batch_rows": len(batch_rows) == AUDIT_BATCH_COUNT,
        "ten_batch_signal_locks": len(batch_locks) == AUDIT_BATCH_COUNT,
        "selection_lock_hash_matches_runtime": _sha256(lock_path)
        == raw.get("selection_signal_lock_sha256"),
        "cycle2_optimizer_steps_zero": raw.get("cycle2_optimizer_steps") == 0,
        "audit_updates_no_parameters": raw.get("parameters_updated_by_audit") is False,
        "audit_log_stops_before_cycle2_optimization": (
            "PCGrad exact parameter audit stop: after_pre_cycle=2; "
            "cycle2_optimizer_steps=0; parameters_updated_by_audit=False"
            in args.audit_log.read_text(errors="ignore")
        ),
        "audit_log_has_no_cycle2_accuracy_checkpoint": not any(
            int(cycle) == 2
            for _iteration, _maximum, cycle, _cycles, _accuracy in ACCURACY_PATTERN.findall(
                args.audit_log.read_text(errors="ignore")
            )
        ),
        "all_batch_locks_precede_oracle_labels": all(
            json.loads(path.read_text()).get("labels_read_after_this_manifest") is True
            for path in batch_locks
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    if not all(input_checks.values()):
        failed = [name for name, passed in input_checks.items() if not passed]
        raise RuntimeError(f"Exact parameter input contract failed: {failed}")

    def vector(column: str) -> np.ndarray:
        return np.asarray([float(row[column]) for row in batch_rows], dtype=np.float64)

    comparisons = {
        "cosine": paired_metric_summary(
            vector("candidate_cosine"), vector("baseline_cosine"), seed=2020
        ),
        "oracle_unit_projection": paired_metric_summary(
            vector("candidate_oracle_unit_projection"),
            vector("baseline_oracle_unit_projection"),
            seed=2021,
        ),
        "first_order": paired_metric_summary(
            vector("candidate_first_order"),
            vector("baseline_first_order"),
            seed=2022,
        ),
    }
    baseline_first_order = vector("baseline_first_order")
    candidate_first_order = vector("candidate_first_order")
    baseline_positive = np.maximum(baseline_first_order, 0.0).sum()
    candidate_positive = np.maximum(candidate_first_order, 0.0).sum()
    helpful_retention = float(
        candidate_positive / baseline_positive * 100.0
        if baseline_positive > 0.0
        else 0.0
    )
    norm_ratio = vector("candidate_norm") / vector("baseline_norm")
    active_conflicts = int(
        sum(int(row["output_pcgrad_active_conflicts"]) for row in batch_rows)
    )
    group_delta = {}
    for group in ("car", "person", "truck", "other_nine"):
        selected = [row for row in group_rows if row["group"] == group]
        if not selected:
            raise RuntimeError(f"Oracle group absent from audit: {group}")
        sample_count = np.asarray([int(row["samples"]) for row in selected])
        delta = np.asarray(
            [float(row["candidate_minus_baseline_first_order"]) for row in selected]
        )
        group_delta[group] = float(np.average(delta, weights=sample_count))

    baseline_burden = negative_burden(baseline_first_order)
    candidate_burden = negative_burden(candidate_first_order)
    positive_batch_fraction = float(
        np.mean(candidate_first_order > baseline_first_order) * 100.0
    )
    audited_coverage = float(raw["audited_conflict_coverage_pct"])
    output_active_coverage = float(active_conflicts / AUDITED_CONFLICTS * 100.0)
    gate = evaluate_exact_parameter_gate(
        input_contract_valid=all(input_checks.values()),
        cycle1_max_accuracy_error_pp=max_cycle_error,
        audited_conflict_coverage_pct=audited_coverage,
        output_active_coverage_pct=output_active_coverage,
        comparisons=comparisons,
        baseline_negative_burden=baseline_burden,
        candidate_negative_burden=candidate_burden,
        helpful_retention_pct=helpful_retention,
        mean_norm_ratio=float(norm_ratio.mean()),
        positive_batch_fraction_pct=positive_batch_fraction,
        group_first_order_delta=group_delta,
    )
    summary = {
        "dataset": "VISDA-C",
        "seed": 2020,
        "decision": gate["decision"],
        "oracle_diagnostic": True,
        "target_labels_used_only_for_post_lock_parameter_alignment": True,
        "candidate": "rowwise_symmetric_PCGrad_for_unresolved_consistency_clip_output_gradients",
        "matched_control": {
            "control_final": 87.93,
            "control_cycle1": control_cycle,
            "audit_cycle1": audit_cycle,
            "absolute_accuracy_errors_pp": cycle_errors,
            "max_accuracy_error_pp": max_cycle_error,
        },
        "input_contract": {"passed": all(input_checks.values()), "checks": input_checks},
        "parameter_metrics": {
            "unresolved_conflicts": int(raw["unresolved_conflicts"]),
            "audited_conflicts": int(raw["audited_conflicts"]),
            "audited_conflict_coverage_pct": audited_coverage,
            "output_pcgrad_active_conflicts": active_conflicts,
            "output_pcgrad_active_coverage_pct": output_active_coverage,
            "comparisons": comparisons,
            "baseline_negative_burden": baseline_burden,
            "candidate_negative_burden": candidate_burden,
            "helpful_first_order_retention_pct": helpful_retention,
            "candidate_to_baseline_mean_norm_ratio": float(norm_ratio.mean()),
            "positive_batch_fraction_pct": positive_batch_fraction,
            "group_first_order_delta": group_delta,
        },
        "gate": gate,
        "matched_proxy_authorized": gate["matched_proxy_authorized"],
        "full_training_authorized": False,
        "seed_sweep_authorized": False,
        "estimated_gpu_cost_paid": "one pure-DUET cycle plus the no-update gradient audit",
        "next": (
            "design exactly one matched proxy25 PCGrad experiment"
            if gate["decision"] == "PASS_EXACT_PARAMETER_PREFLIGHT"
            else "close PCGrad without proxy or full training"
        ),
        "artifacts": {
            "runtime_raw_sha256": _sha256(raw_path),
            "selection_lock_sha256": _sha256(lock_path),
            "batch_oracle_csv_sha256": _sha256(batch_path),
            "group_oracle_csv_sha256": _sha256(group_path),
            "feature_summary_sha256": _sha256(args.feature_summary),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, args.output.with_suffix(".md"))
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote exact parameter summary: {args.output}")


if __name__ == "__main__":
    main()
