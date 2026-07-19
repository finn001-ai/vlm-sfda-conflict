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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    by_task = {row["task"]: row for row in rows}
    missing = [task for task in TASKS if task not in by_task]
    if missing:
        raise ValueError(f"Missing tasks: {missing}")

    final_values = [float(by_task[task]["accuracy"]) for task in TASKS]
    peak_values = [float(by_task[task]["peak_accuracy"]) for task in TASKS]
    final_mean = sum(final_values) / len(final_values)
    peak_mean = sum(peak_values) / len(peak_values)

    if final_mean > DUET_MEAN:
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
        "duet_mean": DUET_MEAN,
        "final_mean": round(final_mean, 4),
        "oracle_peak_mean": round(peak_mean, 4),
        "oracle_peak_minus_final": round(peak_mean - final_mean, 4),
        "next_method": next_method,
        "tasks": [
            {
                "task": task,
                "final_accuracy": float(by_task[task]["accuracy"]),
                "oracle_peak_accuracy": float(by_task[task]["peak_accuracy"]),
                "peak_cycle": by_task[task]["peak_cycle"],
                "peak_iter": by_task[task]["peak_iter"],
                "peak_minus_final": round(
                    float(by_task[task]["peak_accuracy"])
                    - float(by_task[task]["accuracy"]),
                    4,
                ),
            }
            for task in TASKS
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
