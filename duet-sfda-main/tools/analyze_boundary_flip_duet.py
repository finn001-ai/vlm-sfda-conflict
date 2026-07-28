#!/usr/bin/env python
"""Summarize the matched Boundary-Flip DUET preflight without label selection."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


EXPECTED_TASKS = {"AC", "PC", "RC"}


def read_final_accuracy(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, float] = {}
    for row in rows:
        task = row.get("task", "")
        if task:
            result[task] = float(row["final_accuracy"])
    return result


def read_mechanism(pattern: str) -> dict[str, object]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    by_task: dict[str, dict[str, int]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            required = (
                "boundary_flip_candidate_mask",
                "boundary_flip_stable_mask",
                "boundary_flip_active_mask",
            )
            if any(key not in payload for key in required):
                continue
            task = str(payload["task"])
            task_stats = by_task.setdefault(
                task, {"cycles": 0, "candidates": 0, "stable": 0, "active": 0}
            )
            task_stats["cycles"] += 1
            task_stats["candidates"] += int(
                payload["boundary_flip_candidate_mask"].sum()
            )
            task_stats["stable"] += int(
                payload["boundary_flip_stable_mask"].sum()
            )
            task_stats["active"] += int(
                payload["boundary_flip_active_mask"].sum()
            )
    return {
        "snapshots": len(paths),
        "tasks": by_task,
        "all_tasks_active": EXPECTED_TASKS.issubset(by_task)
        and all(by_task[task]["active"] > 0 for task in EXPECTED_TASKS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--control-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-control", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = read_final_accuracy(args.candidate_csv)
    control = read_final_accuracy(args.control_csv)
    mechanism = read_mechanism(args.diagnostics_glob)
    common_tasks = sorted(set(candidate) & set(control))
    deltas = {
        task: round(candidate[task] - control[task], 4) for task in common_tasks
    }
    candidate_mean = (
        float(np.mean(list(candidate.values()))) if candidate else None
    )
    control_mean = float(np.mean(list(control.values()))) if control else None
    mean_delta = (
        candidate_mean - control_mean
        if candidate_mean is not None and control_mean is not None
        else None
    )
    accuracy_gate = (
        set(common_tasks) == EXPECTED_TASKS
        and set(candidate) == EXPECTED_TASKS
        and mean_delta is not None
        and mean_delta >= 0.20
        and sum(delta >= 0.0 for delta in deltas.values()) >= 2
        and min(deltas.values(), default=-999.0) >= -0.30
    )
    control_ready = set(common_tasks) == EXPECTED_TASKS or not args.require_control
    if not control and not args.require_control:
        decision = (
            "candidate_complete_control_pending"
            if mechanism["all_tasks_active"] and set(candidate) == EXPECTED_TASKS
            else "fail_boundary_flip_mechanism"
        )
    else:
        decision = (
            "pass_boundary_flip_preflight"
            if control_ready and mechanism["all_tasks_active"] and accuracy_gate
            else "fail_boundary_flip_preflight"
        )
    report = {
        "decision": decision,
        "candidate_accuracy": candidate,
        "control_accuracy": control,
        "delta_by_task": deltas,
        "candidate_mean": (
            round(candidate_mean, 4) if candidate_mean is not None else None
        ),
        "control_mean": (
            round(control_mean, 4) if control_mean is not None else None
        ),
        "mean_delta": round(mean_delta, 4) if mean_delta is not None else None,
        "mechanism": mechanism,
        "gates": {
            "control_ready": control_ready,
            "mechanism_active_on_all_tasks": mechanism["all_tasks_active"],
            "mean_delta_at_least_0_20": bool(
                mean_delta is not None and mean_delta >= 0.20
            ),
            "wins_at_least_2_of_3": sum(
                delta >= 0.0 for delta in deltas.values()
            )
            >= 2,
            "worst_delta_at_least_minus_0_30": bool(deltas)
            and min(deltas.values()) >= -0.30,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
