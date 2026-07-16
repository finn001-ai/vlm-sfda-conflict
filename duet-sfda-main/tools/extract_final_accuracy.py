#!/usr/bin/env python
"""Extract final task accuracy from DUET/DCCL log files."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/*/*.txt",
        help="Glob for log txt files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(p) for p in glob.glob(args.glob))
    pattern = re.compile(r"Task:\s*([A-Z]{2}),\s*Iter:\s*(\d+)/(\d+);\s*Cycle:\s*(\d+)/(\d+);\s*Accuracy\s*=\s*([0-9.]+)%")
    print("method,task,cycle,iter,accuracy,log")
    for path in paths:
        matches = pattern.findall(path.read_text(errors="ignore"))
        if not matches:
            continue
        task, iter_num, max_iter, cycle, max_cycle, acc = matches[-1]
        parts = path.parts
        method = parts[-2] if len(parts) >= 2 else "unknown"
        print(f"{method},{task},{cycle}/{max_cycle},{iter_num}/{max_iter},{acc},{path}")


if __name__ == "__main__":
    main()
