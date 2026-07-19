#!/usr/bin/env python
"""Gate the Stage20 AC/PA/RA covariance-transport preflight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ["AC", "PA", "RA"]
MATCHED_BASE = {"AC": 73.59, "PA": 83.11, "RA": 83.52}
DUET = {"AC": 73.6, "PA": 82.7, "RA": 83.6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, str]], flow: dict) -> dict:
    if len(rows) != len(TASKS):
        raise ValueError(f"Expected three Stage20 rows, found {len(rows)}")
    by_task = {row["task"]: row for row in rows}
    if set(by_task) != set(TASKS):
        raise ValueError(f"Expected tasks {TASKS}, found {sorted(by_task)}")
    flow_by_task = {item["task"]: item for item in flow.get("tasks", [])}
    if set(flow_by_task) != set(TASKS):
        raise ValueError("Stage20 flow diagnostics do not match preflight tasks")

    accuracies = []
    task_summaries = []
    config_valid = True
    for task in TASKS:
        row = by_task[task]
        accuracy = float(row["accuracy"])
        row_config_valid = (
            row["selection"] == "peak"
            and row["record_type"] == "standard"
            and row["target_head_variant"] == "blend"
            and row["cov_transport_adapt"].lower() == "true"
            and int(row["cov_transport_min_anchors"]) == 8
            and int(row["cov_transport_rank"]) == 4
            and float(row["cov_transport_max_gate"]) == 0.05
        )
        config_valid = config_valid and row_config_valid
        accuracies.append(accuracy)
        task_summaries.append(
            {
                "task": task,
                "accuracy": round(accuracy, 4),
                "matched_base": MATCHED_BASE[task],
                "delta_vs_matched_base": round(
                    accuracy - MATCHED_BASE[task], 4
                ),
                "duet": DUET[task],
                "delta_vs_duet": round(accuracy - DUET[task], 4),
                "active_classes": flow_by_task[task]["active_classes"],
                "eligible_coverage": flow_by_task[task]["eligible_coverage"],
                "mean_relative_shift": flow_by_task[task][
                    "mean_relative_shift"
                ],
                "config_valid": row_config_valid,
            }
        )

    accuracy_mean = mean(accuracies)
    matched_mean = mean(list(MATCHED_BASE.values()))
    duet_mean = mean(list(DUET.values()))
    checks = {
        "config_valid": config_valid,
        "mechanism_valid": flow.get("decision") == "pass_transport_diagnostics",
        "mean_beats_matched_base": accuracy_mean > matched_mean,
        "mean_beats_duet_subset": accuracy_mean > duet_mean,
        "no_task_collapse": all(
            item["delta_vs_matched_base"] >= -0.5 for item in task_summaries
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "pass_covariance_preflight"
            if passed
            else "fail_covariance_preflight"
        ),
        "selection": "peak",
        "warning": "peak accuracy uses target labels and is diagnostic only",
        "mean_accuracy": round(accuracy_mean, 4),
        "matched_base_mean": round(matched_mean, 4),
        "delta_vs_matched_base": round(accuracy_mean - matched_mean, 4),
        "duet_subset_mean": round(duet_mean, 4),
        "delta_vs_duet_subset": round(accuracy_mean - duet_mean, 4),
        "checks": checks,
        "tasks": task_summaries,
        "next": (
            "run the complete 12-task Stage20 seed-2022 gate"
            if passed
            else "replace candidate-conditioned transport with global agreement-whitened optimal transport"
        ),
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    flow = json.loads(Path(args.flow).read_text())
    summary = summarize(rows, flow)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
