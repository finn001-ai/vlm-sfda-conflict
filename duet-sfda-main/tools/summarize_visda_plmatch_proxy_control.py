#!/usr/bin/env python
"""Compare a VisDA PLMatch proxy control with the matched DCCL P1 result."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

try:
    from tools.summarize_visda_temporal_precision_head import parse_records
except ModuleNotFoundError:
    from summarize_visda_temporal_precision_head import parse_records


HARD_CLASS_INDICES = (3, 7, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dccl-final", type=float, default=87.83)
    parser.add_argument("--dccl-hard-mean", type=float, default=73.79)
    parser.add_argument("--dccl-other9-mean", type=float, default=92.51)
    parser.add_argument("--tie-margin", type=float, default=0.20)
    return parser.parse_args()


def mean_at(values: list[float], indices: tuple[int, ...]) -> float:
    return sum(values[index] for index in indices) / len(indices)


def summarize_control(
    records: list[dict],
    *,
    dccl_final: float = 87.83,
    dccl_hard_mean: float = 73.79,
    dccl_other9_mean: float = 92.51,
    tie_margin: float = 0.20,
) -> dict:
    if not records:
        raise ValueError("PLMatch proxy control has no accuracy records")
    if tie_margin < 0:
        raise ValueError("tie_margin must be non-negative")

    final = records[-1]
    if final["cycle"] != final["max_cycle"]:
        raise ValueError("PLMatch proxy control did not finish its cycle budget")

    peak = max(records, key=lambda row: row["accuracy"])
    final_classes = final["class_accuracy"]
    hard_mean = mean_at(final_classes, HARD_CLASS_INDICES)
    other_indices = tuple(
        index for index in range(len(final_classes)) if index not in HARD_CLASS_INDICES
    )
    other9_mean = mean_at(final_classes, other_indices)
    delta = final["accuracy"] - dccl_final

    if delta > tie_margin:
        decision = "plmatch_above_dccl"
        next_step = "run DCCL component ablations before proposing another module"
    elif delta < -tie_margin:
        decision = "dccl_above_plmatch"
        next_step = "preserve DCCL and diagnose a dataset-agnostic class-balanced objective"
    else:
        decision = "matched_within_margin"
        next_step = "treat the external 91.4 gap as potentially environment-dependent"

    return {
        "decision": decision,
        "metric": "VisDA mean per-class accuracy",
        "selection_warning": "oracle peak reads validation labels and is diagnostic only",
        "tie_margin": tie_margin,
        "num_checkpoints": len(records),
        "plmatch_final": round(final["accuracy"], 4),
        "plmatch_oracle_peak": round(peak["accuracy"], 4),
        "plmatch_peak_cycle": peak["cycle"],
        "plmatch_peak_iter": peak["iteration"],
        "plmatch_hard_mean": round(hard_mean, 4),
        "plmatch_other9_mean": round(other9_mean, 4),
        "dccl_p1_final": dccl_final,
        "dccl_p1_hard_mean": dccl_hard_mean,
        "dccl_p1_other9_mean": dccl_other9_mean,
        "final_delta_vs_dccl": round(delta, 4),
        "hard_mean_delta_vs_dccl": round(hard_mean - dccl_hard_mean, 4),
        "other9_mean_delta_vs_dccl": round(other9_mean - dccl_other9_mean, 4),
        "next": next_step,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one clean PLMatch log, found {len(paths)}")
    records = parse_records(paths[0].read_text(errors="ignore"))
    result = summarize_control(
        records,
        dccl_final=args.dccl_final,
        dccl_hard_mean=args.dccl_hard_mean,
        dccl_other9_mean=args.dccl_other9_mean,
        tie_margin=args.tie_margin,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
