#!/usr/bin/env python
"""Gate the Stage18 seed-2022 class-pair flow experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ["AC", "AP", "AR", "CA", "CP", "CR", "PA", "PC", "PR", "RA", "RC", "RP"]
DUET = {
    "AC": 73.6,
    "AP": 90.4,
    "AR": 91.0,
    "CA": 83.6,
    "CP": 90.7,
    "CR": 90.9,
    "PA": 82.7,
    "PC": 73.7,
    "PR": 91.2,
    "RA": 83.6,
    "RC": 74.0,
    "RP": 91.2,
}
HISTORICAL_STAGE14_SEED2022 = 84.6783
MATCHED_STAGE17_ONLINE_SEED2022 = 84.7225


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--matched-online-mean",
        type=float,
        default=MATCHED_STAGE17_ONLINE_SEED2022,
    )
    parser.add_argument("--min-active-tasks", type=int, default=10)
    parser.add_argument("--min-task-delta", type=float, default=-1.5)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_rows(
    rows: list[dict[str, str]],
    matched_online_mean: float,
    min_active_tasks: int,
    min_task_delta: float,
) -> dict:
    if len(rows) != len(TASKS):
        raise ValueError(f"Expected 12 task rows, found {len(rows)}")
    by_task = {row["task"]: row for row in rows}
    missing = [task for task in TASKS if task not in by_task]
    if missing or len(by_task) != len(rows):
        raise ValueError(f"Missing or duplicate tasks: {missing}")

    methods = {row["method"] for row in rows}
    selections = {row["selection"] for row in rows}
    record_types = {row["record_type"] for row in rows}
    variants = {row["target_head_variant"] for row in rows}
    if len(methods) != 1:
        raise ValueError(f"Expected one method, found {sorted(methods)}")
    if selections != {"peak"}:
        raise ValueError(f"Stage18 requires peak selection, found {sorted(selections)}")
    if record_types != {"standard"}:
        raise ValueError(f"Stage18 requires standard records, found {sorted(record_types)}")
    if variants != {"pair_flow"}:
        raise ValueError(f"Expected pair_flow target head, found {sorted(variants)}")

    task_summaries = []
    accuracies = []
    active_tasks = 0
    for task in TASKS:
        row = by_task[task]
        accuracy = float(row["accuracy"])
        active_rank = int(row["pair_flow_active_rank"] or 0)
        final_gate = float(row["pair_flow_gate_final"] or 0.0)
        delta = accuracy - DUET[task]
        accuracies.append(accuracy)
        active_tasks += active_rank > 0
        task_summaries.append(
            {
                "task": task,
                "accuracy": round(accuracy, 4),
                "duet": DUET[task],
                "delta_vs_duet": round(delta, 4),
                "pair_flow_active_rank": active_rank,
                "pair_flow_gate_final": round(final_gate, 6),
            }
        )

    duet_mean = mean([DUET[task] for task in TASKS])
    accuracy_mean = mean(accuracies)
    worst_task_delta = min(item["delta_vs_duet"] for item in task_summaries)
    required_mean = max(duet_mean, matched_online_mean)
    mean_passes = accuracy_mean > required_mean
    mechanism_passes = active_tasks >= min_active_tasks
    collapse_passes = worst_task_delta >= min_task_delta
    passed = mean_passes and mechanism_passes and collapse_passes
    return {
        "decision": "pass_seed2022_gate" if passed else "fail_seed2022_gate",
        "gate": (
            f"peak mean > {required_mean:.4f}; active pair-flow basis on at least "
            f"{min_active_tasks}/12 tasks; worst task delta vs DUET >= {min_task_delta:.2f}"
        ),
        "method": next(iter(methods)),
        "selection": "peak",
        "duet_mean": round(duet_mean, 4),
        "historical_stage14_seed2022_mean": HISTORICAL_STAGE14_SEED2022,
        "matched_stage17_online_seed2022_mean": round(matched_online_mean, 4),
        "mean_accuracy": round(accuracy_mean, 4),
        "delta_vs_duet": round(accuracy_mean - duet_mean, 4),
        "delta_vs_historical_stage14": round(
            accuracy_mean - HISTORICAL_STAGE14_SEED2022, 4
        ),
        "delta_vs_matched_online": round(accuracy_mean - matched_online_mean, 4),
        "task_wins_vs_duet": sum(
            item["delta_vs_duet"] >= 0.0 for item in task_summaries
        ),
        "active_pair_flow_tasks": active_tasks,
        "worst_task_delta_vs_duet": round(worst_task_delta, 4),
        "checks": {
            "mean_passes": mean_passes,
            "mechanism_passes": mechanism_passes,
            "collapse_passes": collapse_passes,
        },
        "next_if_fail": (
            "replace classifier-only adaptation with class-pair-specific low-rank "
            "feature adapters softly routed by the aggregate conflict-flow posterior"
        ),
        "tasks": task_summaries,
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    summary = summarize_rows(
        rows,
        matched_online_mean=args.matched_online_mean,
        min_active_tasks=args.min_active_tasks,
        min_task_delta=args.min_task_delta,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
