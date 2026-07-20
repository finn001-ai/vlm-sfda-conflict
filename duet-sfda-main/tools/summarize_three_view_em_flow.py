#!/usr/bin/env python
"""Summarize Stage22 three-view EM activation from training logs."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


TASK_PATTERN = re.compile(r"Task:\s*([A-Z]{2}),")
CONFIG_PATTERN = re.compile(
    r"DCCL three-view EM enabled: start_cycle=(\d+); steps=(\d+); "
    r"dirichlet=([0-9.]+); min_class_anchors=(\d+); par=([0-9.]+); "
    r"gradient_scope=([^\s]+)"
)
CONSENSUS_PATTERN = re.compile(
    r"DCCL three-view EM consensus: cycle=(\d+); anchors=(\d+); "
    r"active_classes=(\d+); conflicts=(\d+); weighted_conflicts=(\d+); "
    r"mean_conflict_weight=([0-9.]+); changed_top1=(\d+); "
    r"source_diag=([0-9.]+); clip_diag=([0-9.]+); graph_diag=([0-9.]+)"
)
LOSS_PATTERN = re.compile(
    r"three_view_em_loss=([0-9.]+); three_view_em_batches=(\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def summarize_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    tasks = TASK_PATTERN.findall(text)
    config_match = CONFIG_PATTERN.search(text)
    cycle_matches = CONSENSUS_PATTERN.findall(text)
    loss_matches = LOSS_PATTERN.findall(text)
    if not tasks or config_match is None:
        raise ValueError(f"Missing Stage22 task/config log records in {path}")
    if not cycle_matches:
        raise ValueError(f"Missing Stage22 consensus records in {path}")

    start_cycle, steps, dirichlet, min_anchors, par, gradient_scope = (
        config_match.groups()
    )
    cycles = []
    for match in cycle_matches:
        (
            cycle,
            anchors,
            active_classes,
            conflicts,
            weighted_conflicts,
            mean_weight,
            changed_top1,
            source_diag,
            clip_diag,
            graph_diag,
        ) = match
        cycles.append(
            {
                "cycle": int(cycle),
                "anchors": int(anchors),
                "active_classes": int(active_classes),
                "conflicts": int(conflicts),
                "weighted_conflicts": int(weighted_conflicts),
                "mean_conflict_weight": float(mean_weight),
                "changed_top1": int(changed_top1),
                "source_transition_diagonal": float(source_diag),
                "clip_transition_diagonal": float(clip_diag),
                "graph_transition_diagonal": float(graph_diag),
            }
        )
    final_loss, final_batches = loss_matches[-1] if loss_matches else ("0", "0")
    checks = {
        "three_em_cycles": len({item["cycle"] for item in cycles}) >= 3,
        "anchor_support": min(item["anchors"] for item in cycles) >= 512,
        "class_support": min(item["active_classes"] for item in cycles) >= 40,
        "conflict_support": min(item["weighted_conflicts"] for item in cycles) >= 100,
        "continuous_weight": min(item["mean_conflict_weight"] for item in cycles) > 0,
        "head_loss_active": int(final_batches) > 0 and float(final_loss) > 0,
        "head_only_gradient": gradient_scope == "target_head_only",
    }
    return {
        "method": path.parent.name,
        "task": tasks[-1],
        "config": {
            "start_cycle": int(start_cycle),
            "steps": int(steps),
            "dirichlet": float(dirichlet),
            "min_class_anchors": int(min_anchors),
            "par": float(par),
            "gradient_scope": gradient_scope,
        },
        "cycles": cycles,
        "final_head_loss": float(final_loss),
        "final_head_loss_batches": int(final_batches),
        "checks": checks,
        "mechanism_valid": all(checks.values()),
    }


def summarize_paths(paths: list[Path]) -> dict:
    tasks = [summarize_log(path) for path in paths]
    valid_tasks = sum(item["mechanism_valid"] for item in tasks)
    return {
        "decision": (
            "pass_three_view_em_diagnostics"
            if tasks and valid_tasks == len(tasks)
            else "fail_three_view_em_diagnostics"
        ),
        "num_logs": len(tasks),
        "valid_tasks": valid_tasks,
        "tasks": tasks,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(item) for item in glob.glob(args.glob))
    if not paths:
        raise ValueError("No Stage22 logs matched")
    summary = summarize_paths(paths)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
