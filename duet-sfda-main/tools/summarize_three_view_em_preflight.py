#!/usr/bin/env python
"""Gate the Stage22 AC/PA/RA preflight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ["AC", "PA", "RA"]
MATCHED_BASE = {"AC": 73.59, "PA": 83.11, "RA": 83.52}
DUET = {"AC": 73.6, "PA": 82.7, "RA": 83.6}
EXPECTED_CONFIG = {
    "start_cycle": 1,
    "steps": 5,
    "dirichlet": 5.0,
    "min_class_anchors": 3,
    "par": 0.05,
    "gradient_scope": "target_head_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, str]], flow: dict) -> dict:
    if len(rows) != len(TASKS) or len(flow.get("tasks", [])) != len(TASKS):
        raise ValueError("Stage22 preflight requires three unique task records")
    by_task = {row["task"]: row for row in rows}
    flow_by_task = {item["task"]: item for item in flow.get("tasks", [])}
    if set(by_task) != set(TASKS) or set(flow_by_task) != set(TASKS):
        raise ValueError("Stage22 preflight requires exactly AC/PA/RA")

    task_summaries = []
    for task in TASKS:
        row = by_task[task]
        diagnostic = flow_by_task[task]
        accuracy = float(row["accuracy"])
        task_summaries.append(
            {
                "task": task,
                "accuracy": round(accuracy, 4),
                "matched_base": MATCHED_BASE[task],
                "delta_vs_matched_base": round(accuracy - MATCHED_BASE[task], 4),
                "duet": DUET[task],
                "delta_vs_duet": round(accuracy - DUET[task], 4),
                "mechanism_valid": diagnostic["mechanism_valid"],
                "config_valid": diagnostic["config"] == EXPECTED_CONFIG,
                "final_head_loss": diagnostic["final_head_loss"],
            }
        )

    accuracy_mean = mean([item["accuracy"] for item in task_summaries])
    matched_mean = mean(list(MATCHED_BASE.values()))
    duet_mean = mean(list(DUET.values()))
    checks = {
        "peak_selection": all(row["selection"] == "peak" for row in rows),
        "mechanism_valid": all(item["mechanism_valid"] for item in task_summaries),
        "config_valid": all(item["config_valid"] for item in task_summaries),
        "mean_beats_matched_base": accuracy_mean > matched_mean,
        "mean_beats_duet_subset": accuracy_mean > duet_mean,
        "no_task_collapse": all(
            item["delta_vs_matched_base"] >= -0.5 for item in task_summaries
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "pass_three_view_em_preflight"
            if passed
            else "fail_three_view_em_preflight"
        ),
        "selection": "peak",
        "mean_accuracy": round(accuracy_mean, 4),
        "matched_base_mean": round(matched_mean, 4),
        "delta_vs_matched_base": round(accuracy_mean - matched_mean, 4),
        "duet_subset_mean": round(duet_mean, 4),
        "delta_vs_duet_subset": round(accuracy_mean - duet_mean, 4),
        "checks": checks,
        "tasks": task_summaries,
        "next": (
            "run the complete 12-task Stage22 seed-2022 gate"
            if passed
            else "archive the valid failure and retain Stage14"
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
