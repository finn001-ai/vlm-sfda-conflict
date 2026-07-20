#!/usr/bin/env python
"""Gate the complete Stage22 seed-2022 run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ["AC", "AP", "AR", "CA", "CP", "CR", "PA", "PC", "PR", "RA", "RC", "RP"]
DUET = {
    "AC": 73.6, "AP": 90.4, "AR": 91.0, "CA": 83.6,
    "CP": 90.7, "CR": 90.9, "PA": 82.7, "PC": 73.7,
    "PR": 91.2, "RA": 83.6, "RC": 74.0, "RP": 91.2,
}
MATCHED_STAGE14 = 84.7225
EXPECTED_CONFIG = {
    "start_cycle": 1,
    "steps": 5,
    "dirichlet": 5.0,
    "min_class_anchors": 3,
    "par": 0.05,
    "gradient_scope": "target_head_only",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    flow = json.loads(Path(args.flow).read_text())
    if len(rows) != len(TASKS) or len(flow.get("tasks", [])) != len(TASKS):
        raise ValueError("Stage22 seed-2022 gate requires 12 task records")
    by_task = {row["task"]: row for row in rows}
    flow_by_task = {item["task"]: item for item in flow["tasks"]}
    if set(by_task) != set(TASKS) or set(flow_by_task) != set(TASKS):
        raise ValueError("Stage22 seed-2022 tasks are missing or duplicated")

    tasks = []
    for task in TASKS:
        accuracy = float(by_task[task]["accuracy"])
        tasks.append(
            {
                "task": task,
                "accuracy": round(accuracy, 4),
                "duet": DUET[task],
                "delta_vs_duet": round(accuracy - DUET[task], 4),
                "mechanism_valid": flow_by_task[task]["mechanism_valid"],
            }
        )
    accuracy_mean = mean([item["accuracy"] for item in tasks])
    duet_mean = mean(list(DUET.values()))
    checks = {
        "peak_selection": all(row["selection"] == "peak" for row in rows),
        "mechanism_valid_12_tasks": all(item["mechanism_valid"] for item in tasks),
        "fixed_config_12_tasks": all(
            item["config"] == EXPECTED_CONFIG for item in flow_by_task.values()
        ),
        "mean_beats_matched_stage14": accuracy_mean > MATCHED_STAGE14,
        "mean_beats_duet": accuracy_mean > duet_mean,
        "no_task_collapse": min(item["delta_vs_duet"] for item in tasks) >= -1.5,
    }
    passed = all(checks.values())
    summary = {
        "decision": (
            "pass_three_view_em_seed2022_gate"
            if passed
            else "fail_three_view_em_seed2022_gate"
        ),
        "selection": "peak",
        "mean_accuracy": round(accuracy_mean, 4),
        "duet_mean": round(duet_mean, 4),
        "delta_vs_duet": round(accuracy_mean - duet_mean, 4),
        "matched_stage14_mean": MATCHED_STAGE14,
        "delta_vs_matched_stage14": round(accuracy_mean - MATCHED_STAGE14, 4),
        "checks": checks,
        "tasks": tasks,
        "next": (
            "run Stage22 seeds 2020 and 2021 for the three-seed stability gate"
            if passed
            else "archive the valid failure and retain Stage14"
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
