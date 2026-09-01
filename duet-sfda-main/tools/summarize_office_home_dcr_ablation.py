#!/usr/bin/env python3
"""Summarize fixed-final and trajectory-peak DCR-SFDA ablations."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ACCURACY_PATTERN = re.compile(r"Accuracy\s*=\s*([0-9]+(?:\.[0-9]+)?)%")
CYCLE_PATTERN = re.compile(r"Cycle:\s*(\d+)\s*/\s*(\d+)")
TASK_PATTERN = re.compile(r"Task:\s*([A-Z]{2})\b")
DEFAULT_TASKS = ("AC", "CP", "PR", "RA")
DEFAULT_VARIANTS = ("full", "dcm_uniform", "clm_writable", "arg_none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--records-per-cycle", type=int, default=4)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report available runs instead of failing on a missing task/variant.",
    )
    return parser.parse_args()


def candidate_run_dirs(task: str, variant: str, seed: int) -> list[Path]:
    ablation = Path(
        "output/uda/office-home"
    ) / task / f"dcr_ablation_{variant}_office_home_rankadaptive_seed{seed}"
    if variant != "full":
        return [ablation]
    official = Path(
        "output/uda/office-home"
    ) / task / f"dcr_office_home_rankadaptive_seed{seed}"
    return [ablation, official]


def locate_log(task: str, variant: str, seed: int) -> Path | None:
    for run_dir in candidate_run_dirs(task, variant, seed):
        logs = sorted(run_dir.glob("*.txt"))
        if len(logs) == 1:
            return logs[0]
        if len(logs) > 1:
            raise RuntimeError(f"Expected one log in {run_dir}, found {len(logs)}")
    return None


def parse_log(
    path: Path,
    task: str,
    cycles: int = 4,
    records_per_cycle: int = 4,
) -> tuple[float, float, int, int]:
    """Read a fixed-budget prefix from either 4- or extended-cycle logs."""
    records: list[tuple[int | None, float]] = []
    for line in path.read_text().splitlines():
        task_match = TASK_PATTERN.search(line)
        accuracy_match = ACCURACY_PATTERN.search(line)
        if (
            task_match is None
            or task_match.group(1) != task
            or accuracy_match is None
        ):
            continue
        cycle_match = CYCLE_PATTERN.search(line)
        cycle = int(cycle_match.group(1)) if cycle_match else None
        records.append((cycle, float(accuracy_match.group(1))))

    expected = cycles * records_per_cycle
    has_cycle = [cycle is not None for cycle, _ in records]
    if records and all(has_cycle):
        selected = [
            accuracy
            for cycle, accuracy in records
            if cycle is not None and cycle <= cycles
        ]
    elif records and not any(has_cycle):
        # Compatibility with old logs that predate the explicit Cycle field.
        selected = [accuracy for _, accuracy in records[:expected]]
    else:
        raise RuntimeError(f"Mixed cycle metadata in {path}")

    if len(selected) != expected:
        raise RuntimeError(
            f"Expected {expected} accuracy records through cycle {cycles} "
            f"in {path}, found {len(selected)} (total task records={len(records)})"
        )
    return max(selected), selected[-1], len(selected), len(records)


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
            peak, final, records, records_total = parse_log(
                log_path,
                task,
                cycles=args.cycles,
                records_per_cycle=args.records_per_cycle,
            )
            rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "peak": peak,
                    "final": final,
                    "records": records,
                    "records_total": records_total,
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
    extended = [
        f"{row['variant']}/{row['task']}:{row['records_total']}->{row['records']}"
        for row in rows
        if int(row["records_total"]) != int(row["records"])
    ]
    if extended:
        print(
            f"fixed_budget=cycles_1_to_{args.cycles}; "
            "extended_logs_truncated=" + ",".join(extended)
        )
    print(f"csv={output_path}")


if __name__ == "__main__":
    main()
