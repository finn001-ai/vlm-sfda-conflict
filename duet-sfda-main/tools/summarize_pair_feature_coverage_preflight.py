#!/usr/bin/env python
"""Gate the Stage19-C target-Art coverage fallback preflight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ["CA", "PA", "RA"]
STAGE19 = {"CA": 83.60, "PA": 82.49, "RA": 83.07}
MATCHED_ONLINE = {"CA": 83.68, "PA": 83.11, "RA": 83.52}
STAGE19_MEAN = 84.6908
FULL_GATE = 84.7225


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_rows(rows: list[dict[str, str]]) -> dict:
    if len(rows) != len(TASKS):
        raise ValueError(f"Expected three target-Art rows, found {len(rows)}")
    by_task = {row["task"]: row for row in rows}
    if set(by_task) != set(TASKS):
        raise ValueError(f"Expected tasks {TASKS}, found {sorted(by_task)}")

    task_summaries = []
    policy_valid = True
    accuracies = []
    for task in TASKS:
        row = by_task[task]
        rank = int(row["pair_flow_active_rank"] or 0)
        min_rank = int(row["pair_feature_min_active_rank"] or 0)
        effective = row["pair_feature_effective"].lower() == "true"
        router_norm = float(row["pair_feature_router_norm"] or 0.0)
        accuracy = float(row["accuracy"])
        row_policy_valid = (
            row["selection"] == "peak"
            and row["target_head_variant"] == "blend"
            and row["pair_feature_adapt"].lower() == "true"
            and min_rank == 8
            and rank < min_rank
            and not effective
            and router_norm == 0.0
        )
        policy_valid = policy_valid and row_policy_valid
        accuracies.append(accuracy)
        task_summaries.append(
            {
                "task": task,
                "accuracy": round(accuracy, 4),
                "stage19": STAGE19[task],
                "delta_vs_stage19": round(accuracy - STAGE19[task], 4),
                "matched_online": MATCHED_ONLINE[task],
                "delta_vs_matched_online": round(
                    accuracy - MATCHED_ONLINE[task], 4
                ),
                "active_rank": rank,
                "min_active_rank": min_rank,
                "coverage_fallback": not effective,
                "router_norm": router_norm,
                "policy_valid": row_policy_valid,
            }
        )

    old_art_mean = mean([STAGE19[task] for task in TASKS])
    new_art_mean = mean(accuracies)
    projected_mean = STAGE19_MEAN + sum(
        float(by_task[task]["accuracy"]) - STAGE19[task] for task in TASKS
    ) / 12.0
    checks = {
        "coverage_policy_valid": policy_valid,
        "target_art_recovery": new_art_mean >= old_art_mean + 0.20,
        "projected_full_mean_passes": projected_mean > FULL_GATE,
    }
    passed = all(checks.values())
    return {
        "decision": (
            "pass_coverage_preflight" if passed else "fail_coverage_preflight"
        ),
        "selection": "peak",
        "stage19_target_art_mean": round(old_art_mean, 4),
        "coverage_target_art_mean": round(new_art_mean, 4),
        "target_art_recovery": round(new_art_mean - old_art_mean, 4),
        "projected_full_mean_diagnostic": round(projected_mean, 4),
        "projected_delta_vs_required": round(projected_mean - FULL_GATE, 4),
        "warning": (
            "projected_full_mean reuses nine archived Stage19 task results and "
            "is not a final 12-task result"
        ),
        "checks": checks,
        "tasks": task_summaries,
        "next": (
            "run the complete 12-task Stage19-C seed-2022 gate"
            if passed
            else "stop learned pair routing and implement agreement-anchor covariance transport"
        ),
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    summary = summarize_rows(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
