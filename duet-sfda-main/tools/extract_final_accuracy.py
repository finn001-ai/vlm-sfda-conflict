#!/usr/bin/env python
"""从训练日志提取最终/峰值精度和当前方法配置。"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path


ACCURACY_PATTERN = re.compile(
    r"Task:\s*([A-Z]{2}),\s*Iter:\s*(\d+)/(\d+);\s*"
    r"Cycle:\s*(\d+)/(\d+);\s*Accuracy\s*=\s*([0-9.]+)%"
)

CONFIG_PATTERNS = {
    "duet_fcp_power": r"POWER",
}

FIELDS = [
    "method",
    "task",
    "selection",
    "cycle",
    "iter",
    "accuracy",
    "final_accuracy",
    "final_cycle",
    "final_iter",
    "peak_accuracy",
    "peak_cycle",
    "peak_iter",
    "peak_minus_final",
    *CONFIG_PATTERNS,
    "log",
]


def select_final_and_peak(text: str):
    matches = ACCURACY_PATTERN.findall(text)
    if not matches:
        return None, None
    return matches[-1], max(matches, key=lambda item: float(item[5]))


def select_primary(final, peak, selection: str):
    if selection == "final":
        return final
    if selection == "peak":
        return peak
    raise ValueError("selection must be final or peak")


def read_config_value(text: str, key: str) -> str:
    match = re.search(
        rf"^\s*{key}:\s*([^\s#]+)", text, flags=re.MULTILINE
    )
    return match.group(1) if match else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/*/*.txt",
        help="训练日志 glob。",
    )
    parser.add_argument(
        "--selection",
        choices=("final", "peak"),
        default="final",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for path_string in sorted(glob.glob(args.glob)):
        path = Path(path_string)
        text = path.read_text(errors="ignore")
        final, peak = select_final_and_peak(text)
        if final is None:
            continue
        selected = select_primary(final, peak, args.selection)
        task, selected_iter, _, selected_cycle, _, selected_accuracy = selected
        _, final_iter, _, final_cycle, _, final_accuracy = final
        _, peak_iter, _, peak_cycle, _, peak_accuracy = peak
        row = {
            "method": path.parent.name,
            "task": task,
            "selection": args.selection,
            "cycle": selected_cycle,
            "iter": selected_iter,
            "accuracy": selected_accuracy,
            "final_accuracy": final_accuracy,
            "final_cycle": final_cycle,
            "final_iter": final_iter,
            "peak_accuracy": peak_accuracy,
            "peak_cycle": peak_cycle,
            "peak_iter": peak_iter,
            "peak_minus_final": round(
                float(peak_accuracy) - float(final_accuracy), 6
            ),
            "log": str(path),
        }
        row.update(
            {
                name: read_config_value(text, key)
                for name, key in CONFIG_PATTERNS.items()
            }
        )
        writer.writerow(row)


if __name__ == "__main__":
    main()
