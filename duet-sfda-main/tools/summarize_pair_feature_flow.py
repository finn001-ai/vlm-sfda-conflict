#!/usr/bin/env python
"""Extract Stage19 class-pair flow activation diagnostics from logs."""

from __future__ import annotations

import argparse
import ast
import glob
import json
import re
from pathlib import Path


FLOW_PATTERN = re.compile(
    r"fixed-candidate pair flow: cycle=(\d+); valid=(\d+); "
    r"candidate_mass=([0-9.]+); active_rank=(\d+); frozen=(True|False); "
    r"resolved_flow_mass=([0-9.]+)"
)
PAIR_PATTERN = re.compile(r"pair-feature directions frozen: pairs=(\[[^\n]+\])")
TASK_PATTERN = re.compile(r"Task:\s*([A-Z]{2}),")
ROUTER_PATTERN = re.compile(r"pair_feature_router_norm=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def summarize_log(text: str, method: str) -> dict:
    task_matches = TASK_PATTERN.findall(text)
    flow_matches = FLOW_PATTERN.findall(text)
    if not task_matches or not flow_matches:
        raise ValueError(f"Missing Stage19 diagnostics for method {method}")

    cycles = [
        {
            "cycle": int(cycle),
            "fixed_conflicts": int(valid),
            "candidate_mass": float(candidate_mass),
            "active_rank": int(active_rank),
            "frozen": frozen == "True",
            "cumulative_flow_mass": float(flow_mass),
        }
        for cycle, valid, candidate_mass, active_rank, frozen, flow_mass in flow_matches
    ]
    frozen_cycles = [item["cycle"] for item in cycles if item["frozen"]]
    pair_matches = PAIR_PATTERN.findall(text)
    pairs = ast.literal_eval(pair_matches[-1]) if pair_matches else []
    router_matches = ROUTER_PATTERN.findall(text)
    router_norm = float(router_matches[-1]) if router_matches else 0.0
    final_rank = cycles[-1]["active_rank"]
    return {
        "method": method,
        "task": task_matches[-1],
        "activation_cycle": min(frozen_cycles) if frozen_cycles else None,
        "final_active_rank": final_rank,
        "frozen_pairs": [list(pair) for pair in pairs],
        "pair_feature_router_norm": router_norm,
        "mechanism_active": final_rank > 0 and router_norm > 0.0,
        "cycles": cycles,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.glob))
    tasks = [summarize_log(path.read_text(errors="ignore"), path.parent.name) for path in paths]
    summary = {
        "decision": (
            "pass_flow_diagnostics"
            if tasks and all(item["mechanism_active"] for item in tasks)
            else "fail_flow_diagnostics"
        ),
        "num_logs": len(tasks),
        "active_tasks": sum(item["mechanism_active"] for item in tasks),
        "tasks": tasks,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
