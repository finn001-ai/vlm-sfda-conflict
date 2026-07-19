#!/usr/bin/env python
"""Gate Stage19-G graph-temporal-only pair-feature training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

if __package__:
    from tools.summarize_office_home_pair_feature import summarize_rows
else:
    from summarize_office_home_pair_feature import summarize_rows


PREFLIGHT_TASKS = ["AC", "PA", "RA"]
STAGE19 = {"AC": 73.77, "PA": 82.49, "RA": 83.07}
MATCHED_ONLINE = {"AC": 73.59, "PA": 83.11, "RA": 83.52}
STAGE19_MEAN = 84.6908
FULL_GATE = 84.7225


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode", choices=["preflight", "full", "route"], required=True
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def route_is_valid(row: dict[str, str]) -> bool:
    rank = int(row.get("pair_flow_active_rank") or 0)
    min_rank = int(row.get("pair_feature_min_active_rank") or 1)
    return (
        row.get("selection") == "peak"
        and row.get("target_head_variant") == "blend"
        and row.get("pair_feature_adapt", "").lower() == "true"
        and row.get("pair_feature_gradient_mode") == "gtr_only"
        and row.get("pair_feature_effective", "").lower() == "true"
        and rank >= min_rank
        and float(row.get("pair_feature_router_norm") or 0.0) > 0.0
        and int(row.get("pair_feature_gtr_active") or 0) > 0
        and int(row.get("pair_feature_gtr_batches") or 0) > 0
    )


def route_summary(rows: list[dict[str, str]]) -> dict:
    tasks = []
    for row in rows:
        tasks.append(
            {
                "method": row.get("method", ""),
                "task": row.get("task", ""),
                "gradient_mode": row.get("pair_feature_gradient_mode", ""),
                "active_rank": int(row.get("pair_flow_active_rank") or 0),
                "router_norm": float(row.get("pair_feature_router_norm") or 0.0),
                "gtr_active": int(row.get("pair_feature_gtr_active") or 0),
                "gtr_loss": float(row.get("pair_feature_gtr_loss") or 0.0),
                "gtr_batches": int(row.get("pair_feature_gtr_batches") or 0),
                "route_valid": route_is_valid(row),
            }
        )
    return {
        "decision": (
            "pass_gtr_route" if tasks and all(item["route_valid"] for item in tasks)
            else "fail_gtr_route"
        ),
        "num_rows": len(tasks),
        "valid_rows": sum(item["route_valid"] for item in tasks),
        "tasks": tasks,
    }


def summarize_preflight(rows: list[dict[str, str]]) -> dict:
    if len(rows) != len(PREFLIGHT_TASKS):
        raise ValueError(f"Expected three preflight rows, found {len(rows)}")
    by_task = {row["task"]: row for row in rows}
    if set(by_task) != set(PREFLIGHT_TASKS):
        raise ValueError(
            f"Expected tasks {PREFLIGHT_TASKS}, found {sorted(by_task)}"
        )

    route = route_summary(rows)
    accuracies = {task: float(by_task[task]["accuracy"]) for task in PREFLIGHT_TASKS}
    subset_mean = mean(list(accuracies.values()))
    matched_mean = mean([MATCHED_ONLINE[task] for task in PREFLIGHT_TASKS])
    projected_mean = STAGE19_MEAN + sum(
        accuracies[task] - STAGE19[task] for task in PREFLIGHT_TASKS
    ) / 12.0
    worst_delta = min(
        accuracies[task] - MATCHED_ONLINE[task] for task in PREFLIGHT_TASKS
    )
    checks = {
        "gtr_route_passes": route["decision"] == "pass_gtr_route",
        "beats_matched_subset": subset_mean > matched_mean,
        "projected_full_mean_passes": projected_mean > FULL_GATE,
        "task_regression_passes": worst_delta >= -0.30,
    }
    passed = all(checks.values())
    return {
        "decision": "pass_gtr_preflight" if passed else "fail_gtr_preflight",
        "selection": "peak",
        "gate": (
            "valid gtr_only route on AC/PA/RA; subset mean > matched online; "
            "projected full mean > 84.7225; no task loses >0.30"
        ),
        "stage19_subset_mean": round(
            mean([STAGE19[task] for task in PREFLIGHT_TASKS]), 4
        ),
        "matched_online_subset_mean": round(matched_mean, 4),
        "gtr_subset_mean": round(subset_mean, 4),
        "projected_full_mean_diagnostic": round(projected_mean, 4),
        "projected_delta_vs_required": round(projected_mean - FULL_GATE, 4),
        "warning": (
            "projected_full_mean reuses nine archived Stage19 rows and is not "
            "a final 12-task result"
        ),
        "checks": checks,
        "route": route,
        "next": (
            "run the complete Stage19-G seed-2022 gate"
            if passed
            else "archive the valid failure and run the deferred Stage20 covariance preflight"
        ),
        "tasks": [
            {
                "task": task,
                "accuracy": round(accuracies[task], 4),
                "stage19": STAGE19[task],
                "delta_vs_stage19": round(accuracies[task] - STAGE19[task], 4),
                "matched_online": MATCHED_ONLINE[task],
                "delta_vs_matched_online": round(
                    accuracies[task] - MATCHED_ONLINE[task], 4
                ),
                "route_valid": route_is_valid(by_task[task]),
            }
            for task in PREFLIGHT_TASKS
        ],
    }


def summarize_full(rows: list[dict[str, str]]) -> dict:
    summary = summarize_rows(rows, FULL_GATE, 10, -1.5)
    route = route_summary(rows)
    summary["checks"]["gtr_route_passes"] = (
        route["decision"] == "pass_gtr_route" and route["valid_rows"] == 12
    )
    passed = all(summary["checks"].values())
    summary["decision"] = (
        "pass_gtr_seed2022_gate" if passed else "fail_gtr_seed2022_gate"
    )
    summary["route"] = route
    summary["next_if_fail"] = (
        "archive the valid failure and run the deferred Stage20 covariance preflight"
    )
    return summary


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    if args.mode == "preflight":
        summary = summarize_preflight(rows)
    elif args.mode == "full":
        summary = summarize_full(rows)
    else:
        summary = route_summary(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
