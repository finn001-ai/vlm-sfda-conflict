#!/usr/bin/env python
"""Gate the complete Stage21 seed-2022 Office-Home run."""

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
REQUIRED_MEAN = 84.7225


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def summarize(rows: list[dict[str, str]], flow: dict) -> dict:
    if len(rows) != len(TASKS):
        raise ValueError(f"Expected 12 Stage21 rows, found {len(rows)}")
    by_task = {row["task"]: row for row in rows}
    flow_by_task = {item["task"]: item for item in flow.get("tasks", [])}
    if set(by_task) != set(TASKS) or set(flow_by_task) != set(TASKS):
        raise ValueError("Stage21 complete tasks are missing or duplicated")
    tasks = []
    config_valid = True
    for task in TASKS:
        row = by_task[task]
        accuracy = float(row["accuracy"])
        diagnostic = flow_by_task[task]
        row_valid = (
            row["selection"] == "peak"
            and row["record_type"] == "standard"
            and row["target_head_variant"] == "blend"
            and row["cov_transport_adapt"].lower() == "true"
        )
        config_valid = config_valid and row_valid
        tasks.append(
            {
                "task": task,
                "accuracy": round(accuracy, 4),
                "duet": DUET[task],
                "delta_vs_duet": round(accuracy - DUET[task], 4),
                "selected_strength": diagnostic["selected_strength"],
                "mechanism_valid": diagnostic["mechanism_valid"],
                "config_valid": row_valid,
            }
        )
    mean_accuracy = sum(item["accuracy"] for item in tasks) / len(tasks)
    checks = {
        "config_valid": config_valid,
        "mean_passes": mean_accuracy > REQUIRED_MEAN,
        "mechanism_valid": flow.get("decision") == "pass_whitened_diagnostics",
        "collapse_passes": min(item["delta_vs_duet"] for item in tasks) >= -1.5,
    }
    passed = all(checks.values())
    return {
        "decision": "pass_seed2022_gate" if passed else "fail_seed2022_gate",
        "selection": "peak",
        "required_mean": REQUIRED_MEAN,
        "mean_accuracy": round(mean_accuracy, 4),
        "delta_vs_required": round(mean_accuracy - REQUIRED_MEAN, 4),
        "task_wins_vs_duet": sum(item["delta_vs_duet"] >= 0 for item in tasks),
        "checks": checks,
        "tasks": tasks,
        "next": (
            "run the three-seed Stage21 stability gate"
            if passed
            else "replace geometric transport with three-view class-conditional noise EM consensus"
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
