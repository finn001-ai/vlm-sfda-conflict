#!/usr/bin/env python
"""Gate the complete Stage20 seed-2022 covariance-transport run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ["AC", "AP", "AR", "CA", "CP", "CR", "PA", "PC", "PR", "RA", "RC", "RP"]
DUET = dict(zip(TASKS, [73.6, 90.4, 91.0, 83.6, 90.7, 90.9, 82.7, 73.7, 91.2, 83.6, 74.0, 91.2]))
REQUIRED_MEAN = 84.7225


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def summarize(rows: list[dict[str, str]], flow: dict) -> dict:
    if len(rows) != len(TASKS):
        raise ValueError(f"Expected 12 Stage20 rows, found {len(rows)}")
    by_task = {row["task"]: row for row in rows}
    if set(by_task) != set(TASKS):
        raise ValueError("Stage20 CSV has missing or duplicate tasks")
    flow_by_task = {item["task"]: item for item in flow.get("tasks", [])}
    if set(flow_by_task) != set(TASKS):
        raise ValueError("Stage20 flow diagnostics do not match all tasks")

    tasks = []
    accuracies = []
    config_valid = True
    for task in TASKS:
        row = by_task[task]
        accuracy = float(row["accuracy"])
        row_config_valid = (
            row["selection"] == "peak"
            and row["record_type"] == "standard"
            and row["cov_transport_adapt"].lower() == "true"
            and int(row["cov_transport_rank"]) == 4
        )
        config_valid = config_valid and row_config_valid
        accuracies.append(accuracy)
        tasks.append(
            {
                "task": task,
                "accuracy": round(accuracy, 4),
                "duet": DUET[task],
                "delta_vs_duet": round(accuracy - DUET[task], 4),
                "active_classes": flow_by_task[task]["active_classes"],
                "eligible_coverage": flow_by_task[task]["eligible_coverage"],
                "mean_relative_shift": flow_by_task[task]["mean_relative_shift"],
            }
        )

    mean_accuracy = sum(accuracies) / len(accuracies)
    duet_mean = sum(DUET.values()) / len(DUET)
    worst_delta = min(item["delta_vs_duet"] for item in tasks)
    checks = {
        "config_valid": config_valid,
        "mechanism_valid": flow.get("decision") == "pass_transport_diagnostics",
        "mean_passes": mean_accuracy > REQUIRED_MEAN,
        "collapse_passes": worst_delta >= -1.5,
    }
    passed = all(checks.values())
    return {
        "decision": "pass_seed2022_gate" if passed else "fail_seed2022_gate",
        "selection": "peak",
        "required_mean": REQUIRED_MEAN,
        "duet_mean": round(duet_mean, 4),
        "mean_accuracy": round(mean_accuracy, 4),
        "delta_vs_duet": round(mean_accuracy - duet_mean, 4),
        "task_wins_vs_duet": sum(item["delta_vs_duet"] >= 0 for item in tasks),
        "worst_task_delta_vs_duet": round(worst_delta, 4),
        "checks": checks,
        "tasks": tasks,
        "next_if_fail": (
            "replace candidate-conditioned covariance transport with global "
            "agreement-whitened optimal transport"
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
