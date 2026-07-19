#!/usr/bin/env python
"""Summarize final-versus-oracle-peak accuracy without treating peak as valid selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ["AC", "AP", "AR", "CA", "CP", "CR", "PA", "PC", "PR", "RA", "RC", "RP"]
DUET_MEAN = 84.7167


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--peak-is-primary",
        action="store_true",
        help="Treat target-label-selected peak accuracy as the primary protocol.",
    )
    parser.add_argument(
        "--next-on-no-headroom",
        default="move to the source-anchored zero-initialized residual classifier",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["task"]] = row
    for method, task_rows in by_method.items():
        missing = [task for task in TASKS if task not in task_rows]
        if missing:
            raise ValueError(f"Method {method} is missing tasks: {missing}")

    final_values = [
        float(row.get("final_accuracy") or row["accuracy"])
        for row in rows
    ]
    peak_values = [float(row["peak_accuracy"]) for row in rows]
    final_mean = sum(final_values) / len(final_values)
    peak_mean = sum(peak_values) / len(peak_values)

    if args.peak_is_primary and peak_mean > DUET_MEAN:
        decision = "peak_pass"
        next_method = "validate peak-selected stability across seeds"
    elif args.peak_is_primary:
        decision = "no_oracle_headroom"
        next_method = args.next_on_no_headroom
    elif final_mean > DUET_MEAN:
        decision = "final_pass"
        next_method = "validate stability without oracle checkpoint selection"
    elif peak_mean > DUET_MEAN:
        decision = "oracle_headroom_requires_unlabeled_selection"
        next_method = "develop a label-free cycle selector; do not report oracle peak as the main result"
    else:
        decision = "no_oracle_headroom"
        next_method = "move to the source-anchored zero-initialized residual classifier"

    summary = {
        "decision": decision,
        "warning": "peak_accuracy uses target labels and is diagnostic only",
        "primary_protocol": "peak" if args.peak_is_primary else "final",
        "duet_mean": DUET_MEAN,
        "final_mean": round(final_mean, 4),
        "oracle_peak_mean": round(peak_mean, 4),
        "oracle_peak_minus_final": round(peak_mean - final_mean, 4),
        "next_method": next_method,
        "methods": [
            {
                "method": method,
                "final_mean": round(
                    sum(
                        float(task_rows[task].get("final_accuracy") or task_rows[task]["accuracy"])
                        for task in TASKS
                    ) / len(TASKS),
                    4,
                ),
                "oracle_peak_mean": round(
                    sum(float(task_rows[task]["peak_accuracy"]) for task in TASKS)
                    / len(TASKS),
                    4,
                ),
            }
            for method, task_rows in sorted(by_method.items())
        ],
        "tasks": [
            {
                "method": row["method"],
                "task": row["task"],
                "final_accuracy": float(
                    row.get("final_accuracy") or row["accuracy"]
                ),
                "oracle_peak_accuracy": float(row["peak_accuracy"]),
                "peak_cycle": row["peak_cycle"],
                "peak_iter": row["peak_iter"],
                "peak_minus_final": round(
                    float(row["peak_accuracy"])
                    - float(
                        row.get("final_accuracy") or row["accuracy"]
                    ),
                    4,
                ),
            }
            for row in rows
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
