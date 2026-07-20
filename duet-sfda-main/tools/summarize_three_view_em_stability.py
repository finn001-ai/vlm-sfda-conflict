#!/usr/bin/env python
"""Gate the complete three-seed Stage22 stability run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TASKS = ["AC", "AP", "AR", "CA", "CP", "CR", "PA", "PC", "PR", "RA", "RC", "RP"]
SEEDS = ["2020", "2021", "2022"]
DUET = {
    "AC": 73.6, "AP": 90.4, "AR": 91.0, "CA": 83.6,
    "CP": 90.7, "CR": 90.9, "PA": 82.7, "PC": 73.7,
    "PR": 91.2, "RA": 83.6, "RC": 74.0, "RP": 91.2,
}
EXPECTED_CONFIG = {
    "start_cycle": 1,
    "steps": 5,
    "dirichlet": 5.0,
    "min_class_anchors": 3,
    "par": 0.05,
    "gradient_scope": "target_head_only",
}
MAX_SEED_STD = 0.10
MIN_OVERALL_MEAN = 84.7825


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return (
        sum((value - average) ** 2 for value in values) / (len(values) - 1)
    ) ** 0.5


def seed_from_method(method: str) -> str:
    match = re.search(r"seed(\d+)", method)
    if match is None:
        raise ValueError(f"Stage22 method has no seed: {method}")
    return match.group(1)


def summarize(rows: list[dict[str, str]], flow: dict) -> dict:
    diagnostics = flow.get("tasks", [])
    expected_records = len(SEEDS) * len(TASKS)
    if len(rows) != expected_records or len(diagnostics) != expected_records:
        raise ValueError("Stage22 stability requires exactly 36 accuracy and flow records")

    accuracy_by_key = {}
    for row in rows:
        key = (seed_from_method(row["method"]), row["task"])
        if key in accuracy_by_key:
            raise ValueError(f"Duplicate Stage22 accuracy record: {key}")
        accuracy_by_key[key] = row
    flow_by_key = {}
    for item in diagnostics:
        key = (seed_from_method(item["method"]), item["task"])
        if key in flow_by_key:
            raise ValueError(f"Duplicate Stage22 flow record: {key}")
        flow_by_key[key] = item

    expected_keys = {(seed, task) for seed in SEEDS for task in TASKS}
    if set(accuracy_by_key) != expected_keys or set(flow_by_key) != expected_keys:
        raise ValueError("Stage22 stability records do not cover the fixed seed/task grid")

    duet_mean = mean(list(DUET.values()))
    seed_summaries = []
    task_values = {task: [] for task in TASKS}
    for seed in SEEDS:
        values = []
        for task in TASKS:
            value = float(accuracy_by_key[(seed, task)]["accuracy"])
            values.append(value)
            task_values[task].append(value)
        seed_mean = mean(values)
        seed_summaries.append(
            {
                "seed": seed,
                "mean": round(seed_mean, 4),
                "delta_vs_duet": round(seed_mean - duet_mean, 4),
                "beats_duet_mean": seed_mean > duet_mean,
                "task_wins_vs_duet": sum(
                    float(accuracy_by_key[(seed, task)]["accuracy"]) >= DUET[task]
                    for task in TASKS
                ),
            }
        )

    seed_means = [item["mean"] for item in seed_summaries]
    overall_mean = mean(seed_means)
    seed_std = sample_std(seed_means)
    checks = {
        "peak_selection": all(row["selection"] == "peak" for row in rows),
        "mechanism_valid_36_tasks": all(
            item["mechanism_valid"] for item in diagnostics
        ),
        "fixed_config_36_tasks": all(
            item["config"] == EXPECTED_CONFIG for item in diagnostics
        ),
        "all_seeds_beat_duet": min(seed_means) > duet_mean,
        "seed_std_at_most_0_10": seed_std <= MAX_SEED_STD,
        "overall_mean_beats_stage14": overall_mean > MIN_OVERALL_MEAN,
    }
    task_summaries = []
    for task in TASKS:
        values = task_values[task]
        task_summaries.append(
            {
                "task": task,
                "mean": round(mean(values), 4),
                "std": round(sample_std(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "duet": DUET[task],
                "delta_mean_vs_duet": round(mean(values) - DUET[task], 4),
            }
        )
    passed = all(checks.values())
    return {
        "decision": (
            "pass_three_view_em_stability_gate"
            if passed
            else "fail_three_view_em_stability_gate"
        ),
        "selection": "peak",
        "duet_mean": round(duet_mean, 4),
        "mean_over_seed_means": round(overall_mean, 4),
        "std_over_seed_means": round(seed_std, 4),
        "min_seed_mean": round(min(seed_means), 4),
        "max_seed_mean": round(max(seed_means), 4),
        "stage14_peak_mean": MIN_OVERALL_MEAN,
        "checks": checks,
        "seeds": seed_summaries,
        "tasks": task_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    flow = json.loads(Path(args.flow).read_text())
    summary = summarize(rows, flow)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
