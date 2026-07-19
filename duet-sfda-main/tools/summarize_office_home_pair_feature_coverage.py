#!/usr/bin/env python
"""Gate the complete Stage19-C coverage-protected seed-2022 run."""

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
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-task-delta", type=float, default=-1.5)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_rows(rows: list[dict[str, str]], min_task_delta: float = -1.5) -> dict:
    if len(rows) != len(TASKS):
        raise ValueError(f"Expected 12 task rows, found {len(rows)}")
    by_task = {row["task"]: row for row in rows}
    if set(by_task) != set(TASKS):
        raise ValueError(f"Missing or duplicate tasks: {sorted(set(TASKS) - set(by_task))}")

    policy_valid = True
    active_tasks = 0
    fallback_tasks = 0
    accuracies = []
    task_summaries = []
    for task in TASKS:
        row = by_task[task]
        rank = int(row["pair_flow_active_rank"] or 0)
        min_rank = int(row["pair_feature_min_active_rank"] or 0)
        effective = row["pair_feature_effective"].lower() == "true"
        router_norm = float(row["pair_feature_router_norm"] or 0.0)
        expected_effective = rank >= min_rank
        row_policy_valid = (
            row["record_type"] == "standard"
            and row["selection"] == "peak"
            and row["target_head_variant"] == "blend"
            and row["pair_feature_adapt"].lower() == "true"
            and min_rank == 8
            and effective == expected_effective
            and (router_norm > 0.0 if effective else router_norm == 0.0)
        )
        policy_valid = policy_valid and row_policy_valid
        active_tasks += effective
        fallback_tasks += not effective
        accuracy = float(row["accuracy"])
        accuracies.append(accuracy)
        task_summaries.append(
            {
                "task": task,
                "accuracy": round(accuracy, 4),
                "duet": DUET[task],
                "delta_vs_duet": round(accuracy - DUET[task], 4),
                "active_rank": rank,
                "coverage_fallback": not effective,
                "router_norm": round(router_norm, 6),
                "policy_valid": row_policy_valid,
            }
        )

    accuracy_mean = mean(accuracies)
    worst_delta = min(item["delta_vs_duet"] for item in task_summaries)
    checks = {
        "mean_passes": accuracy_mean > REQUIRED_MEAN,
        "coverage_policy_valid": policy_valid,
        "collapse_passes": worst_delta >= min_task_delta,
    }
    passed = all(checks.values())
    return {
        "decision": "pass_seed2022_gate" if passed else "fail_seed2022_gate",
        "gate": (
            f"peak mean > {REQUIRED_MEAN:.4f}; rank>=8 routes through pair-feature "
            f"adapter and rank<8 falls back exactly; worst task delta vs DUET >= {min_task_delta:.2f}"
        ),
        "selection": "peak",
        "duet_mean": round(mean(list(DUET.values())), 4),
        "required_mean": REQUIRED_MEAN,
        "mean_accuracy": round(accuracy_mean, 4),
        "delta_vs_duet": round(accuracy_mean - mean(list(DUET.values())), 4),
        "active_pair_feature_tasks": active_tasks,
        "coverage_fallback_tasks": fallback_tasks,
        "task_wins_vs_duet": sum(item["delta_vs_duet"] >= 0 for item in task_summaries),
        "worst_task_delta_vs_duet": round(worst_delta, 4),
        "checks": checks,
        "tasks": task_summaries,
        "next_if_fail": (
            "replace learned routing with agreement-anchor class-conditional "
            "covariance subspace transport"
        ),
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    summary = summarize_rows(rows, args.min_task_delta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
