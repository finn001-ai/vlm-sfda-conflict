#!/usr/bin/env python
"""Build unified VisDA-C DUET-FCP and Office-Home pure-DUET tables."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

try:
    from tools.extract_final_accuracy import select_final_and_peak
    from tools.summarize_visda_run import DEFAULT_CLASSES, parse_records
except ModuleNotFoundError:
    from extract_final_accuracy import select_final_and_peak
    from summarize_visda_run import DEFAULT_CLASSES, parse_records


OFFICE_TASKS = (
    "AC",
    "AP",
    "AR",
    "CA",
    "CP",
    "CR",
    "PA",
    "PC",
    "PR",
    "RA",
    "RC",
    "RP",
)


def mean(values):
    values = tuple(values)
    return sum(values) / len(values)


def load_one(pattern):
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one VisDA-C log for {pattern!r}, found {len(paths)}"
        )
    return paths[0], paths[0].read_text(errors="ignore")


def parse_office_logs(pattern):
    rows = []
    for path_string in sorted(glob.glob(pattern)):
        path = Path(path_string)
        final, peak = select_final_and_peak(path.read_text(errors="ignore"))
        if final is None:
            continue
        task, final_iter, _, final_cycle, max_cycle, final_accuracy = final
        _, peak_iter, _, peak_cycle, _, peak_accuracy = peak
        rows.append(
            {
                "task": task,
                "final_accuracy": float(final_accuracy),
                "final_cycle": int(final_cycle),
                "final_iter": int(final_iter),
                "oracle_peak_accuracy": float(peak_accuracy),
                "oracle_peak_cycle": int(peak_cycle),
                "oracle_peak_iter": int(peak_iter),
                "max_cycle": int(max_cycle),
                "log": str(path),
            }
        )
    by_task = {row["task"]: row for row in rows}
    missing = sorted(set(OFFICE_TASKS) - set(by_task))
    duplicates = len(rows) != len(by_task)
    if missing or duplicates or len(rows) != len(OFFICE_TASKS):
        raise ValueError(
            "Office-Home requires one log for every task; "
            f"missing={missing}, rows={len(rows)}, unique={len(by_task)}"
        )
    incomplete = [
        row["task"]
        for row in rows
        if row["final_cycle"] != 4 or row["max_cycle"] != 4
    ]
    if incomplete:
        raise ValueError(
            f"Office-Home tasks did not finish 4 cycles: {incomplete}"
        )
    return [by_task[task] for task in OFFICE_TASKS]


def build_report(visda_records, office_rows, class_names=DEFAULT_CLASSES):
    if max(row["cycle"] for row in visda_records) != 8:
        raise ValueError("VisDA-C run did not finish 8 cycles")
    if any(row["max_cycle"] != 8 for row in visda_records):
        raise ValueError("VisDA-C log is not an 8-cycle run")

    visda_final = visda_records[-1]
    visda_peak = max(visda_records, key=lambda row: row["accuracy"])
    cycle_rows = []
    for cycle in range(1, 9):
        records = [row for row in visda_records if row["cycle"] == cycle]
        if not records:
            raise ValueError(f"VisDA-C is missing cycle {cycle}")
        cycle_final = records[-1]
        cycle_rows.append(
            {
                "cycle": cycle,
                "final_accuracy": cycle_final["accuracy"],
                "iteration": cycle_final["iteration"],
            }
        )

    class_rows = [
        {
            "class": name,
            "final_accuracy": visda_final["class_accuracy"][index],
        }
        for index, name in enumerate(class_names)
    ]
    office_final_mean = mean(row["final_accuracy"] for row in office_rows)
    office_peak_mean = mean(
        row["oracle_peak_accuracy"] for row in office_rows
    )
    summary_rows = [
        {
            "dataset": "VisDA-C",
            "method": "DUET-FCP",
            "seed": 2020,
            "cycles": 8,
            "tasks": 1,
            "final_metric": round(visda_final["accuracy"], 4),
            "oracle_peak_metric": round(visda_peak["accuracy"], 4),
        },
        {
            "dataset": "Office-Home",
            "method": "Pure DUET",
            "seed": 2020,
            "cycles": 4,
            "tasks": 12,
            "final_metric": round(office_final_mean, 4),
            "oracle_peak_metric": round(office_peak_mean, 4),
        },
    ]
    return {
        "summary": summary_rows,
        "visda_cycles": cycle_rows,
        "visda_classes": class_rows,
        "office_home_tasks": office_rows,
        "selection_note": (
            "final checkpoint is primary; oracle peak reads target labels "
            "and is diagnostic only"
        ),
    }


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row[header]) for header in headers)
            + " |"
        )
    return "\n".join(lines)


def render_markdown(report):
    summary = [
        {
            "Dataset": row["dataset"],
            "Method": row["method"],
            "Seed": row["seed"],
            "Cycles": row["cycles"],
            "Tasks": row["tasks"],
            "Final": f'{row["final_metric"]:.4f}',
            "Oracle peak": f'{row["oracle_peak_metric"]:.4f}',
        }
        for row in report["summary"]
    ]
    cycles = [
        {
            "Cycle": row["cycle"],
            "Final accuracy": f'{row["final_accuracy"]:.2f}',
            "Iteration": row["iteration"],
        }
        for row in report["visda_cycles"]
    ]
    classes = [
        {
            "Class": row["class"],
            "Final accuracy": f'{row["final_accuracy"]:.2f}',
        }
        for row in report["visda_classes"]
    ]
    office = [
        {
            "Task": row["task"],
            "Final": f'{row["final_accuracy"]:.2f}',
            "Oracle peak": f'{row["oracle_peak_accuracy"]:.2f}',
            "Peak cycle": row["oracle_peak_cycle"],
        }
        for row in report["office_home_tasks"]
    ]
    return "\n\n".join(
        (
            "# DUET-FCP VisDA-C / Pure DUET Office-Home",
            report["selection_note"],
            "## Overall\n\n"
            + markdown_table(
                (
                    "Dataset",
                    "Method",
                    "Seed",
                    "Cycles",
                    "Tasks",
                    "Final",
                    "Oracle peak",
                ),
                summary,
            ),
            "## VisDA-C: final checkpoint of each cycle\n\n"
            + markdown_table(
                ("Cycle", "Final accuracy", "Iteration"),
                cycles,
            ),
            "## VisDA-C: final per-class accuracy\n\n"
            + markdown_table(("Class", "Final accuracy"), classes),
            "## Office-Home: pure DUET, 12 tasks\n\n"
            + markdown_table(
                ("Task", "Final", "Oracle peak", "Peak cycle"),
                office,
            ),
        )
    ) + "\n"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visda-glob",
        default=(
            "output/uda/VISDA-C/TV/"
            "duet_first_cycle_prior_visda_full_seed2020/*.txt"
        ),
    )
    parser.add_argument(
        "--office-glob",
        default=(
            "output/uda/office-home/*/"
            "plmatch_office_home_full_seed2020/*.txt"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="output/uda/benchmark_tables",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, visda_text = load_one(args.visda_glob)
    report = build_report(
        parse_records(visda_text),
        parse_office_logs(args.office_glob),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "duet_fcp_visda8_office_home_duet"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (out_dir / f"{stem}.md").write_text(render_markdown(report))
    write_csv(out_dir / f"{stem}_summary.csv", report["summary"])
    write_csv(out_dir / f"{stem}_visda_cycles.csv", report["visda_cycles"])
    write_csv(out_dir / f"{stem}_visda_classes.csv", report["visda_classes"])
    write_csv(
        out_dir / f"{stem}_office_home_tasks.csv",
        report["office_home_tasks"],
    )
    print(out_dir / f"{stem}.md")


if __name__ == "__main__":
    main()
