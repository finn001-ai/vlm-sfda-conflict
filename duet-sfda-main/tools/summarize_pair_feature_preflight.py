#!/usr/bin/env python
"""Verify that Stage19 activates before running all Office-Home tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def summarize_rows(rows: list[dict[str, str]]) -> dict:
    if len(rows) != 1 or rows[0]["task"] != "AC":
        raise ValueError("Stage19 preflight expects exactly the AC task")
    row = rows[0]
    active_rank = int(row["pair_flow_active_rank"] or 0)
    router_norm = float(row["pair_feature_router_norm"] or 0.0)
    config_valid = (
        row["target_head_variant"] == "blend"
        and row["pair_feature_adapt"].lower() == "true"
        and row["selection"] == "peak"
    )
    mechanism_active = active_rank > 0 and router_norm > 0.0
    passed = config_valid and mechanism_active
    return {
        "decision": "pass_mechanism_preflight" if passed else "fail_mechanism_preflight",
        "task": "AC",
        "accuracy_diagnostic_only": float(row["accuracy"]),
        "pair_flow_active_rank": active_rank,
        "pair_feature_gate_final": float(row["pair_feature_gate_final"] or 0.0),
        "pair_feature_router_norm": router_norm,
        "checks": {
            "config_valid": config_valid,
            "mechanism_active": mechanism_active,
        },
        "next": (
            "run the 12-task seed-2022 gate"
            if passed
            else "stop before the 12-task run and audit the activation log"
        ),
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.csv).open()))
    summary = summarize_rows(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
