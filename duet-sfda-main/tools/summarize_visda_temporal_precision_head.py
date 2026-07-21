#!/usr/bin/env python
"""Extract VisDA-C mean-class and per-class Stage14 checkpoint results."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path


DEFAULT_CLASSES = [
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
]
RECORD_PATTERN = re.compile(
    r"Task:\s*TV,\s*Iter:(\d+)/(\d+);\s*Cycle:\s*(\d+)/(\d+);\s*"
    r"Accuracy\s*=\s*([0-9.]+)%"
)
FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--class-names")
    return parser.parse_args()


def parse_records(text: str) -> list[dict]:
    lines = text.splitlines()
    records = []
    for index, line in enumerate(lines):
        match = RECORD_PATTERN.search(line)
        if match is None:
            continue
        if index + 1 >= len(lines):
            raise ValueError("VisDA-C accuracy record is missing its per-class line")
        class_values = [float(value) for value in FLOAT_PATTERN.findall(lines[index + 1])]
        if len(class_values) != 12:
            raise ValueError(
                f"Expected 12 VisDA-C class accuracies, found {len(class_values)}"
            )
        iteration, max_iteration, cycle, max_cycle, accuracy = match.groups()
        records.append(
            {
                "iteration": int(iteration),
                "max_iteration": int(max_iteration),
                "cycle": int(cycle),
                "max_cycle": int(max_cycle),
                "accuracy": float(accuracy),
                "class_accuracy": class_values,
            }
        )
    if not records:
        raise ValueError("No VisDA-C Stage14 accuracy records found")
    return records


def load_class_names(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_CLASSES
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(names) != 12:
        raise ValueError("VisDA-C classname file must contain 12 non-empty lines")
    return names


def summarize(records: list[dict], class_names: list[str]) -> dict:
    final = records[-1]
    peak = max(records, key=lambda item: item["accuracy"])
    return {
        "decision": "visda_stage14_run_complete",
        "metric": "mean per-class accuracy",
        "selection_warning": "peak reads validation labels and is oracle-only",
        "num_checkpoints": len(records),
        "final": final,
        "oracle_peak": peak,
        "peak_minus_final": round(peak["accuracy"] - final["accuracy"], 4),
        "classes": [
            {
                "class": name,
                "final_accuracy": final["class_accuracy"][index],
                "peak_checkpoint_accuracy": peak["class_accuracy"][index],
            }
            for index, name in enumerate(class_names)
        ],
        "next": "compare with same-source VisDA baselines, then run adaptation seeds 2021/2022",
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one clean VisDA-C log, found {len(paths)}")
    class_names = load_class_names(args.class_names)
    summary = summarize(parse_records(paths[0].read_text(errors="ignore")), class_names)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class", "final_accuracy", "peak_checkpoint_accuracy"],
        )
        writer.writeheader()
        writer.writerows(summary["classes"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
