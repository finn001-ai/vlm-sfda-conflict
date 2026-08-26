#!/usr/bin/env python
"""Summarize Office-Home adaptation-seed sweep accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import re
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Accuracy CSV from extract_final_accuracy.py")
    parser.add_argument("--out", required=True, help="Output JSON summary path")
    parser.add_argument(
        "--max-seed-std",
        type=float,
        default=None,
        help="Optional maximum sample std across seed means",
    )
    parser.add_argument(
        "--min-overall-mean",
        type=float,
        default=None,
        help="Optional strict lower bound for the mean over seed means",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def seed_from_method(method: str) -> str:
    match = re.search(r"seed(\d+)", method)
    if not match:
        return method
    return match.group(1)


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    by_seed: dict[str, dict[str, float]] = {}
    for row in rows:
        seed = seed_from_method(row["method"])
        by_seed.setdefault(seed, {})[row["task"]] = float(row["accuracy"])

    duet_mean = mean([DUET[task] for task in TASKS])
    seed_summaries = []
    task_values = {task: [] for task in TASKS}
    for seed in sorted(by_seed, key=lambda item: int(item) if item.isdigit() else item):
        missing = [task for task in TASKS if task not in by_seed[seed]]
        if missing:
            raise ValueError(f"Seed {seed} is missing tasks: {missing}")
        values = [by_seed[seed][task] for task in TASKS]
        for task, value in by_seed[seed].items():
            task_values[task].append(value)
        seed_mean = mean(values)
        seed_summaries.append(
            {
                "seed": seed,
                "mean": round(seed_mean, 4),
                "delta_vs_duet": round(seed_mean - duet_mean, 4),
                "beats_duet_mean": seed_mean > duet_mean,
                "task_wins_vs_duet": sum(by_seed[seed][task] >= DUET[task] for task in TASKS),
            }
        )

    seed_means = [item["mean"] for item in seed_summaries]
    task_summaries = []
    for task in TASKS:
        values = task_values[task]
        task_mean = mean(values)
        task_summaries.append(
            {
                "task": task,
                "mean": round(task_mean, 4),
                "std": round(sample_std(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "duet": DUET[task],
                "delta_mean_vs_duet": round(task_mean - DUET[task], 4),
            }
        )

    seed_std = sample_std(seed_means)
    all_beat_duet = bool(seed_means) and min(seed_means) > duet_mean
    std_passes = args.max_seed_std is None or seed_std <= args.max_seed_std
    overall_mean = mean(seed_means)
    overall_mean_passes = (
        args.min_overall_mean is None or overall_mean > args.min_overall_mean
    )
    summary = {
        "decision": "pass_stability_gate"
        if all_beat_duet and std_passes and overall_mean_passes
        else "fail_stability_gate",
        "gate": (
            "all adaptation seeds must beat DUET mean"
            + (
                f" and seed-mean std must be <= {args.max_seed_std:.4f}"
                if args.max_seed_std is not None
                else ""
            )
            + (
                f" and mean over seed means must exceed {args.min_overall_mean:.4f}"
                if args.min_overall_mean is not None
                else ""
            )
        ),
        "duet_mean": round(duet_mean, 4),
        "num_seeds": len(seed_summaries),
        "mean_over_seed_means": round(overall_mean, 4),
        "std_over_seed_means": round(seed_std, 4),
        "min_seed_mean": round(min(seed_means), 4) if seed_means else 0.0,
        "max_seed_mean": round(max(seed_means), 4) if seed_means else 0.0,
        "checks": {
            "all_seeds_beat_duet": all_beat_duet,
            "seed_std_passes": std_passes,
            "overall_mean_passes": overall_mean_passes,
        },
        "seeds": seed_summaries,
        "tasks": task_summaries,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
