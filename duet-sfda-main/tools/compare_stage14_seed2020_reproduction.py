#!/usr/bin/env python
"""Compare a Stage14 seed-2020 rerun with the archived oracle peaks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED = {
    "AC": 73.81, "AP": 91.12, "AR": 91.23, "CA": 83.60,
    "CP": 91.17, "CR": 90.73, "PA": 83.44, "PC": 73.47,
    "PR": 91.21, "RA": 83.56, "RC": 74.20, "RP": 91.26,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    if len(rows) != len(EXPECTED):
        raise ValueError(
            "Stage14 reproduction requires exactly 12 records; use a clean "
            "isolated output directory"
        )
    if any(row.get("selection") != "peak" for row in rows):
        raise ValueError("Stage14 reproduction comparison requires peak selection")
    observed = {row["task"]: float(row["accuracy"]) for row in rows}
    if set(observed) != set(EXPECTED):
        raise ValueError("Stage14 reproduction requires all 12 unique tasks")
    tasks = [
        {
            "task": task,
            "observed_peak": observed[task],
            "archived_peak": EXPECTED[task],
            "delta": round(observed[task] - EXPECTED[task], 4),
        }
        for task in EXPECTED
    ]
    observed_mean = sum(observed.values()) / len(observed)
    archived_mean = sum(EXPECTED.values()) / len(EXPECTED)
    exact = all(item["delta"] == 0 for item in tasks)
    close = (
        abs(observed_mean - archived_mean) <= 0.10
        and max(abs(item["delta"]) for item in tasks) <= 0.50
    )
    summary = {
        "decision": (
            "exact_reproduction"
            if exact
            else "close_reproduction" if close else "reproduction_mismatch"
        ),
        "selection": "peak",
        "observed_mean": round(observed_mean, 4),
        "archived_mean": round(archived_mean, 4),
        "mean_delta": round(observed_mean - archived_mean, 4),
        "warning": (
            "peak uses target labels; cuDNN benchmark mode can prevent bitwise "
            "reproduction even with the same seed"
        ),
        "tasks": tasks,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
