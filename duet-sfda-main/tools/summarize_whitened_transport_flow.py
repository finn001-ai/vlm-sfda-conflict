#!/usr/bin/env python
"""Extract Stage21 agreement-whitened transport diagnostics."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


ENABLED_PATTERN = re.compile(
    r"agreement-whitened transport enabled: min_anchors=(\d+); "
    r"shrinkage=([0-9.]+); holdout_ratio=([0-9.]+); max_gate=([0-9.]+); "
    r"min_improvement=([0-9.]+); start_cycle=(\d+)"
)
FROZEN_PATTERN = re.compile(
    r"agreement-whitened geometry frozen: anchors=(\d+); "
    r"train_anchors=(\d+); heldout_anchors=(\d+); active_classes=(\d+); "
    r"selected_strength=([0-9.]+); heldout_loss_improvement=([0-9.]+); "
    r"heldout_accuracy_delta=(-?[0-9.]+); mean_relative_shift=([0-9.]+)"
)
TASK_PATTERN = re.compile(r"Task:\s*([A-Z]{2}),")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-active-classes", type=int, default=40)
    parser.add_argument("--min-heldout", type=int, default=100)
    return parser.parse_args()


def summarize_log(
    text: str,
    method: str,
    min_active_classes: int = 40,
    min_heldout: int = 100,
) -> dict:
    task_matches = TASK_PATTERN.findall(text)
    enabled_matches = ENABLED_PATTERN.findall(text)
    frozen_matches = FROZEN_PATTERN.findall(text)
    if not task_matches or len(enabled_matches) != 1 or len(frozen_matches) != 1:
        raise ValueError(f"Missing or repeated Stage21 diagnostics in {method}")
    min_anchors, shrinkage, holdout_ratio, max_gate, min_improvement, start = (
        enabled_matches[0]
    )
    anchors, train, heldout, active, strength, improvement, acc_delta, shift = (
        frozen_matches[0]
    )
    config = {
        "min_anchors": int(min_anchors),
        "shrinkage": float(shrinkage),
        "holdout_ratio": float(holdout_ratio),
        "max_gate": float(max_gate),
        "min_improvement": float(min_improvement),
        "start_cycle": int(start),
    }
    anchors = int(anchors)
    heldout = int(heldout)
    active = int(active)
    strength = float(strength)
    improvement = float(improvement)
    acc_delta = float(acc_delta)
    shift = float(shift)
    checks = {
        "anchor_support": anchors >= config["min_anchors"],
        "heldout_support": heldout >= min_heldout,
        "class_support": active >= min_active_classes,
        "label_free_selection_active": 0.0 < strength <= config["max_gate"],
        "heldout_objective_improves": improvement >= config["min_improvement"],
        "heldout_consensus_preserved": acc_delta >= -1e-6,
        "global_shift_nonzero_and_bounded": 0.0 < shift <= strength + 1e-6,
    }
    return {
        "method": method,
        "task": task_matches[-1],
        "anchors": anchors,
        "train_anchors": int(train),
        "heldout_anchors": heldout,
        "active_classes": active,
        "selected_strength": strength,
        "heldout_loss_improvement": improvement,
        "heldout_accuracy_delta": acc_delta,
        "mean_relative_shift": shift,
        "config": config,
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
            args.min_heldout,
        )
        for path in paths
    ]
    summary = {
        "decision": (
            "pass_whitened_diagnostics"
            if tasks and all(task["mechanism_valid"] for task in tasks)
            else "fail_whitened_diagnostics"
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
