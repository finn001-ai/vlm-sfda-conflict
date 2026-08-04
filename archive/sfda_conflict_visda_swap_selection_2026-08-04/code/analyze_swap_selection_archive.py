#!/usr/bin/env python3
"""Reproduce the swap-selection decision-level evidence archived 2026-08-04.

Input: per-cycle conflict_samples.csv from the TV (VISDA-C train->validation,
seed 2020) Top-k conflict probe run.
Outputs (written under --out):
  decision_curve.csv   D threshold x direction gate x early-stop grid:
                       decisions / correct / precision / net-correct

The rule and the offline-locked direction-accuracy table are imported from the
archived implementation copy ``swap_conflict_selection.py`` (same file as the
mainline training-path module).  Ground truth is used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from swap_conflict_selection import CYCLE0_DIRECTION_ACCURACY  # noqa: E402


EPS = 1e-9


def read_swap_rows(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    return [r for r in rows if r["bidirectional_cross_support"] == "True"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dir containing cycle_000..cycle_007")
    parser.add_argument("--out", required=True, help="output dir")
    args = parser.parse_args()
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-cycle evidence columns from the archived CSV rows.
    evidence = []
    for cycle in range(8):
        rows = read_swap_rows(data_dir / f"cycle_{cycle:03d}" / "conflict_samples.csv")
        evidence.append(
            {
                "cycle": cycle,
                "pA": [float(r["task_top1_prob"]) for r in rows],
                "pB": [float(r["task_top2_prob"]) for r in rows],
                "qA": [float(r["clip_top2_score"]) for r in rows],
                "qB": [float(r["clip_top1_score"]) for r in rows],
                "gt": [int(r["gt_label_probe"]) for r in rows],
                "A": [int(r["task_top1_id"]) for r in rows],
                "B": [int(r["clip_top1_id"]) for r in rows],
            }
        )

    curve: list[dict[str, object]] = []
    for gate_D in (2.0, 4.0):
        for dir_thr in (0.0, 0.8):
            for last_cycle in (6, 8):
                decisions = correct = 0
                for cycle, ev in enumerate(evidence):
                    if cycle + 1 > last_cycle:
                        continue
                    for i in range(len(ev["A"])):
                        if cycle == 0:
                            chosen = ev["B"][i]
                        else:
                            log_t = math.log(max(ev["pA"][i], EPS)) - math.log(
                                max(ev["pB"][i], EPS)
                            )
                            log_c = math.log(max(ev["qB"][i], EPS)) - math.log(
                                max(ev["qA"][i], EPS)
                            )
                            diff = log_t - log_c
                            if abs(diff) < gate_D:
                                continue
                            chosen = ev["A"][i] if diff > 0 else ev["B"][i]
                        if dir_thr > 0.0 and CYCLE0_DIRECTION_ACCURACY.get(
                            (ev["A"][i], ev["B"][i]), 0.0
                        ) < dir_thr:
                            continue
                        decisions += 1
                        correct += chosen == ev["gt"][i]
                wrong = decisions - correct
                curve.append(
                    {
                        "gate_D": gate_D,
                        "direction_accuracy": dir_thr,
                        "last_active_cycle": last_cycle,
                        "decisions": decisions,
                        "correct": correct,
                        "wrong": wrong,
                        "precision_pct": round(100 * correct / decisions, 1),
                        "net_correct": correct - wrong,
                    }
                )

    write_csv(out_dir / "decision_curve.csv", curve)
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
