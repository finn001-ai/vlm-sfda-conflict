#!/usr/bin/env python
"""汇总 VisDA-C Boundary-Flip DUET 的最终精度与机制激活状态。"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

try:
    from tools.analyze_boundary_flip_duet import read_final_accuracy
except ModuleNotFoundError:
    # 支持从仓库根目录直接执行：
    # python tools/analyze_visda_boundary_flip_duet.py
    from analyze_boundary_flip_duet import read_final_accuracy


def read_visda_mechanism(
    diagnostics_pattern: str, log_pattern: str
) -> dict[str, int | float | bool]:
    snapshots = sorted(Path(path) for path in glob.glob(diagnostics_pattern))
    candidates = 0
    stable = 0
    active = 0
    boundary_snapshots = 0
    for path in snapshots:
        with np.load(path, allow_pickle=False) as payload:
            required = (
                "boundary_flip_candidate_mask",
                "boundary_flip_stable_mask",
                "boundary_flip_active_mask",
            )
            if any(key not in payload for key in required):
                continue
            boundary_snapshots += 1
            candidates += int(payload["boundary_flip_candidate_mask"].sum())
            stable += int(payload["boundary_flip_stable_mask"].sum())
            active += int(payload["boundary_flip_active_mask"].sum())

    loss_batches = 0
    positive_loss_records = 0
    for path_string in glob.glob(log_pattern):
        text = Path(path_string).read_text(errors="ignore")
        loss_batches += sum(
            int(value)
            for value in re.findall(r"boundary_flip_batches=(\d+)", text)
        )
        positive_loss_records += sum(
            float(value) > 0.0
            for value in re.findall(r"boundary_flip_loss=([0-9.]+)", text)
        )

    return {
        "snapshots": boundary_snapshots,
        "candidates": candidates,
        "stable": stable,
        "active": active,
        "loss_batches": loss_batches,
        "positive_loss_records": positive_loss_records,
        "valid": active > 0 and loss_batches > 0 and positive_loss_records > 0,
    }


def summarize(
    candidate_accuracy: dict[str, float],
    control_accuracy: dict[str, float],
    mechanism: dict[str, int | float | bool],
    *,
    min_delta: float = 0.20,
) -> dict[str, object]:
    candidate = candidate_accuracy.get("TV")
    control = control_accuracy.get("TV")
    delta = (
        candidate - control
        if candidate is not None and control is not None
        else None
    )

    if control is None:
        decision = (
            "candidate_complete_control_pending"
            if candidate is not None and mechanism["valid"]
            else "fail_visda_boundary_flip_mechanism"
        )
    else:
        decision = (
            "pass_visda_boundary_flip_preflight"
            if candidate is not None
            and mechanism["valid"]
            and delta is not None
            and delta >= min_delta
            else "fail_visda_boundary_flip_preflight"
        )
    return {
        "decision": decision,
        "candidate_final_accuracy": candidate,
        "control_final_accuracy": control,
        "delta": round(delta, 4) if delta is not None else None,
        "minimum_delta": min_delta,
        "mechanism": mechanism,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--control-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-glob", required=True)
    parser.add_argument("--log-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-delta", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize(
        read_final_accuracy(args.candidate_csv),
        read_final_accuracy(args.control_csv),
        read_visda_mechanism(args.diagnostics_glob, args.log_glob),
        min_delta=args.min_delta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
