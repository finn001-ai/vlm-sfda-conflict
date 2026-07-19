#!/usr/bin/env python
"""Extract Stage20 agreement-covariance transport diagnostics from logs."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


GEOMETRY_PATTERN = re.compile(
    r"agreement covariance geometry frozen: anchors=(\d+); "
    r"active_classes=(\d+); fixed_conflicts=(\d+); "
    r"eligible_conflicts=(\d+); eligible_coverage=([0-9.]+); "
    r"mean_relative_shift=([0-9.]+)"
)
TASK_PATTERN = re.compile(r"Task:\s*([A-Z]{2}),")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-active-classes", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=0.25)
    parser.add_argument("--max-shift", type=float, default=0.05)
    return parser.parse_args()


def summarize_log(
    text: str,
    method: str,
    min_active_classes: int = 20,
    min_coverage: float = 0.25,
    max_shift: float = 0.05,
) -> dict:
    task_matches = TASK_PATTERN.findall(text)
    geometry_matches = GEOMETRY_PATTERN.findall(text)
    if not task_matches or len(geometry_matches) != 1:
        raise ValueError(f"Missing or repeated Stage20 geometry in {method}")
    anchors, active, conflicts, eligible, coverage, shift = geometry_matches[0]
    active = int(active)
    coverage = float(coverage)
    shift = float(shift)
    checks = {
        "class_geometry_present": active >= min_active_classes,
        "conflict_coverage_present": coverage >= min_coverage,
        "transport_nonzero_and_bounded": 0.0 < shift <= max_shift + 1e-6,
    }
    return {
        "method": method,
        "task": task_matches[-1],
        "anchors": int(anchors),
        "active_classes": active,
        "fixed_conflicts": int(conflicts),
        "eligible_conflicts": int(eligible),
        "eligible_coverage": coverage,
        "mean_relative_shift": shift,
        "checks": checks,
        "mechanism_valid": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.glob))
    tasks = [
        summarize_log(
            path.read_text(errors="ignore"),
            path.parent.name,
            args.min_active_classes,
            args.min_coverage,
            args.max_shift,
        )
        for path in paths
    ]
    summary = {
        "decision": (
            "pass_transport_diagnostics"
            if tasks and all(task["mechanism_valid"] for task in tasks)
            else "fail_transport_diagnostics"
        ),
        "num_logs": len(tasks),
        "valid_tasks": sum(task["mechanism_valid"] for task in tasks),
        "tasks": tasks,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
