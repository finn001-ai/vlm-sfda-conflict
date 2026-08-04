#!/usr/bin/env python3
"""Reproduce the Top-K / swap selection analysis archived 2026-08-04.

Input: per-cycle conflict_samples.csv from the TV (VISDA-C train->validation,
seed 2020) Top-K conflict probe run.
Outputs (written under --out):
  per_cycle_swap_stats.csv   per-cycle swap counts / accuracies / top pairs
  selection_curve.csv        label-free D-gated selection scheme (cycle0=CLIP,
                             cycle>=1 preference-ratio + decision-strength gate)
  baselines.csv              weighted vote / always CLIP / always task
  pair_orientation.csv       per orientation-pair side accuracy
  prior_agree_disagree.csv   oracle-informed previous-cycle pair prior
                             (agree/disagree zones; GT of previous cycle used
                              ONLY to estimate the prior upper bound)

The scheme (for bidirectional_cross_support / pure-swap conflicts only):
  A = task top1 (also clip top2), B = clip top1 (also task top2)
  eA = pA * qA, eB = pB * qB
  pick A if log(eA) - log(eB) >= D; pick B if log(eB) - log(eA) >= D;
  otherwise abstain (no label). Cycle 0 special case: always pick B (CLIP top1).
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
from pathlib import Path


CLASS_NAMES = {
    0: "aeroplane", 1: "bicycle", 2: "bus", 3: "car", 4: "horse", 5: "knife",
    6: "motorcycle", 7: "person", 8: "plant", 9: "skateboard", 10: "train",
    11: "truck",
}
EPS = 1e-9
TOTAL_SAMPLES = 55_388
THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def read_swap_rows(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    return [r for r in rows if r["bidirectional_cross_support"] == "True"]


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


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

    per_cycle: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    curve: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    agree_rows: list[dict[str, object]] = []
    prev = None  # previous cycle swaps (for oracle-informed prior)
    total_swap = 0

    for cycle in range(8):
        path = data_dir / f"cycle_{cycle:03d}" / "conflict_samples.csv"
        rows_all = list(csv.DictReader(path.open()))
        bi = [r for r in rows_all if r["bidirectional_cross_support"] == "True"]
        n = len(bi)
        total_swap += n
        g = [r["gt_label_probe"] for r in bi]
        A = [r["task_top1_id"] for r in bi]
        B = [r["clip_top1_id"] for r in bi]
        pA = [float(r["task_top1_prob"]) for r in bi]
        pB = [float(r["task_top2_prob"]) for r in bi]
        qB = [float(r["clip_top1_score"]) for r in bi]
        qA = [float(r["clip_top2_score"]) for r in bi]
        lrT = [math.log(max(p, EPS)) - math.log(max(b, EPS)) for p, b in zip(pA, pB)]
        lrC = [math.log(max(q, EPS)) - math.log(max(a, EPS)) for q, a in zip(qB, qA)]
        D = [abs(x - y) for x, y in zip(lrT, lrC)]

        task_acc = sum(1 for a, gg in zip(A, g) if a == gg) / n
        clip_acc = sum(1 for b, gg in zip(B, g) if b == gg) / n

        pairs = collections.Counter(
            (CLASS_NAMES[int(a)], CLASS_NAMES[int(b)]) for a, b in zip(A, B)
        )
        top_pairs = "; ".join(f"{a}/{b}:{v}" for (a, b), v in pairs.most_common(3))
        per_cycle.append(
            {
                "cycle": cycle,
                "conflicts": len(rows_all),
                "n_swap": n,
                "swap_pct_of_conflicts": round(pct(n, len(rows_all)), 1),
                "swap_pct_of_all": round(pct(n, TOTAL_SAMPLES), 2),
                "task_acc": round(100 * task_acc, 1),
                "clip_acc": round(100 * clip_acc, 1),
                "top_pairs": top_pairs,
            }
        )

        agg = collections.defaultdict(lambda: [0, 0, 0])  # (A,B) -> n, task_c, clip_c
        for a, b, gg in zip(A, B, g):
            agg[(a, b)][0] += 1
            agg[(a, b)][1] += a == gg
            agg[(a, b)][2] += b == gg
        for (a, b), (cnt, tc, cc) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]:
            pair_rows.append(
                {
                    "cycle": cycle,
                    "task_top1": CLASS_NAMES[int(a)],
                    "clip_top1": CLASS_NAMES[int(b)],
                    "n": cnt,
                    "task_acc": round(100 * tc / cnt, 1),
                    "clip_acc": round(100 * cc / cnt, 1),
                }
            )

        # ---- label-free scheme: cycle 0 always CLIP; later preference-ratio + gate
        for thr in THRESHOLDS:
            if cycle == 0:
                dec = n
                cor = sum(1 for b, gg in zip(B, g) if b == gg)
            else:
                dec = cor = 0
                for dd, t, a, b, gg in zip(D, [x > y for x, y in zip(lrT, lrC)], A, B, g):
                    if dd >= thr:
                        dec += 1
                        cor += (a == gg) if t else (b == gg)
            row = next((r for r in curve if r["threshold"] == thr), None)
            if row is None:
                row = {"threshold": thr, "decisions": 0, "correct": 0, "scheme": "label_free"}
                curve.append(row)
            row["decisions"] += dec
            row["correct"] += cor

        # ---- baselines
        va = [p + q for p, q in zip(pA, qA)]
        vb = [p + q for p, q in zip(pB, qB)]
        wv = sum((a == gg) if vA >= vB else (b == gg)
                 for vA, vB, a, b, gg in zip(va, vb, A, B, g))
        cl = sum(1 for b, gg in zip(B, g) if b == gg)
        ta = sum(1 for a, gg in zip(A, g) if a == gg)
        for key, cor in (("weighted_vote", wv), ("always_clip", cl), ("always_task", ta)):
            row = next((r for r in baseline_rows if r["scheme"] == key), None)
            if row is None:
                row = {"scheme": key, "decisions": 0, "correct": 0}
                baseline_rows.append(row)
            row["decisions"] += n
            row["correct"] += cor

        # ---- oracle-informed previous-cycle pair prior (upper bound only)
        if prev is not None:
            prior: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
            for a, b, gg in zip(prev["A"], prev["B"], prev["g"]):
                prior[(a, b)][0] += a == gg
                prior[(a, b)][1] += b == gg
            agree = agree_cor = disagree = disagree_cor = no_prior = 0
            for dd, t, a, b, gg in zip(D, [x > y for x, y in zip(lrT, lrC)], A, B, g):
                k = (a, b)
                if k in prior:
                    prior_task = prior[k][0] >= prior[k][1]
                    if prior_task == t:
                        agree += 1
                        agree_cor += (a == gg) if t else (b == gg)
                    else:
                        disagree += 1
                        disagree_cor += (a == gg) if t else (b == gg)
                else:
                    no_prior += 1
            agree_rows.append(
                {
                    "cycle": cycle,
                    "agree_n": agree,
                    "agree_acc": round(100 * agree_cor / agree, 1) if agree else None,
                    "disagree_n": disagree,
                    "disagree_acc": round(100 * disagree_cor / disagree, 1) if disagree else None,
                    "no_prior_n": no_prior,
                    "note": "oracle-informed: prior uses previous-cycle ground truth",
                }
            )

        prev = {"A": A, "B": B, "g": g}

    write_csv(out_dir / "per_cycle_swap_stats.csv", per_cycle)
    write_csv(out_dir / "pair_orientation.csv", pair_rows)
    write_csv(out_dir / "baselines.csv",
              [{"scheme": r["scheme"], "decisions": r["decisions"], "correct": r["correct"],
                "precision": round(100 * r["correct"] / r["decisions"], 1)}
               for r in baseline_rows])
    write_csv(out_dir / "selection_curve.csv",
              [{"threshold": r["threshold"], "decisions": r["decisions"],
                "coverage_pct": round(100 * r["decisions"] / total_swap, 1),
                "correct": r["correct"], "wrong": r["decisions"] - r["correct"],
                "precision": round(100 * r["correct"] / r["decisions"], 1),
                "scheme": r["scheme"]}
               for r in curve])
    write_csv(out_dir / "prior_agree_disagree.csv", agree_rows)
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
