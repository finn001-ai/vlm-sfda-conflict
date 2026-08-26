#!/usr/bin/env python3
"""Summarize fixed-final and trajectory-peak DCR-SFDA ablations."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ACCURACY_PATTERN = re.compile(r"Accuracy\s*=\s*([0-9]+(?:\.[0-9]+)?)%")
DEFAULT_TASKS = ("AC", "CP", "PR", "RA")
DEFAULT_VARIANTS = ("full", "dcm_uniform", "clm_writable", "arg_none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report available runs instead of failing on a missing task/variant.",
    )
    return parser.parse_args()


def candidate_run_dirs(task: str, variant: str, seed: int) -> list[Path]:
    ablation = Path(
        "output/uda/office-home"
    ) / task / f"plmatch_dac_handoff_dcr_sfda_ablation_{variant}_office_home_seed{seed}"
    if variant != "full":
        return [ablation]
    official = Path(
        "output/uda/office-home"
    ) / task / f"plmatch_dac_handoff_credit_residual_office_home_full_seed{seed}"
    return [ablation, official]


def locate_log(task: str, variant: str, seed: int) -> Path | None:
    for run_dir in candidate_run_dirs(task, variant, seed):
        logs = sorted(run_dir.glob("*.txt"))
        if len(logs) == 1:
            return logs[0]
        if len(logs) > 1:
            raise RuntimeError(f"Expected one log in {run_dir}, found {len(logs)}")
    return None


def parse_log(path: Path) -> tuple[float, float, int]:
    values = [float(value) for value in ACCURACY_PATTERN.findall(path.read_text())]
    if len(values) != 16:
        raise RuntimeError(f"Expected 16 accuracy records in {path}, found {len(values)}")
    return max(values), values[-1], len(values)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for variant in args.variants:
        for task in args.tasks:
            log_path = locate_log(task, variant, args.seed)
            if log_path is None:
                missing.append(f"{variant}/{task}")
                continue
            peak, final, records = parse_log(log_path)
            rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "peak": peak,
                    "final": final,
                    "records": records,
                    "log": str(log_path),
                }
            )

    if missing and not args.allow_missing:
        raise SystemExit("Missing ablation runs: " + ", ".join(missing))
    if not rows:
        raise SystemExit("No completed ablation runs found")

    output_path = Path(f"output/dcr_sfda_ablation_summary_seed{args.seed}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    full_final = None
    if "full" in grouped and len(grouped["full"]) == len(args.tasks):
        full_final = sum(float(row["final"]) for row in grouped["full"]) / len(args.tasks)

    print("variant\ttasks\tpeak_mean\tfinal_mean\tdelta_final_vs_full")
    for variant in args.variants:
        variant_rows = grouped.get(variant, [])
        if not variant_rows:
            continue
        peak_mean = sum(float(row["peak"]) for row in variant_rows) / len(variant_rows)
        final_mean = sum(float(row["final"]) for row in variant_rows) / len(variant_rows)
        delta = "NA" if full_final is None else f"{final_mean - full_final:+.2f}"
        print(
            f"{variant}\t{len(variant_rows)}\t{peak_mean:.2f}\t"
            f"{final_mean:.2f}\t{delta}"
        )
    if missing:
        print("missing=" + ",".join(missing))
    print(f"csv={output_path}")


if __name__ == "__main__":
    main()
