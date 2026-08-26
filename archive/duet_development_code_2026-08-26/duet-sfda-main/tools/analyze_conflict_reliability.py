#!/usr/bin/env python
"""Evaluate oracle-free conflict selectors from exported diagnostic CSV files.

This script reads `*_conflicts.csv` files produced by
`export_conflict_diagnostics.py`. Ground-truth correctness columns are used only
for evaluation. Selector scores are computed from source/CLIP predictions,
confidences, and agreement-derived class reliability.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/plmatch/diagnostics/*_conflicts.csv",
        help="Glob for per-sample diagnostic CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="output/uda/office-home/conflict_reliability",
        help="Directory for reliability analysis outputs.",
    )
    parser.add_argument(
        "--score-gap",
        type=float,
        default=0.0,
        help="Reject conflict samples when class-wise source/CLIP scores differ by less than this value.",
    )
    parser.add_argument(
        "--conf-gap",
        type=float,
        default=0.0,
        help="Reject conflict samples when source/CLIP confidences differ by less than this value.",
    )
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"true", "1", "yes"}


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def task_name(path: Path) -> str:
    return path.name.split("_conflicts.csv")[0]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pct(num: float, denom: float) -> float:
    return round(100.0 * num / denom, 4) if denom else 0.0


def correctness(row: dict[str, str], side: str) -> bool:
    return parse_bool(row[f"{side}_correct"])


def source_or_clip_acc(rows: Iterable[dict[str, str]], side: str) -> float:
    rows = list(rows)
    return pct(sum(correctness(row, side) for row in rows), len(rows))


def selector_eval(
    rows: list[dict[str, str]],
    choices: list[str | None],
) -> dict[str, float | int]:
    assert len(rows) == len(choices)
    selected = [(row, choice) for row, choice in zip(rows, choices) if choice is not None]
    rejected = [(row, choice) for row, choice in zip(rows, choices) if choice is None]
    correct = sum(correctness(row, choice) for row, choice in selected)
    useful_total = sum(
        parse_bool(row["source_correct"]) != parse_bool(row["clip_correct"])
        for row in rows
    )
    selected_useful = sum(
        correctness(row, choice)
        for row, choice in selected
    )
    rejected_harmful = sum(
        (not parse_bool(row["source_correct"])) and (not parse_bool(row["clip_correct"]))
        for row, _ in rejected
    )
    return {
        "selected": len(selected),
        "rejected": len(rejected),
        "coverage": pct(len(selected), len(rows)),
        "accuracy": pct(correct, len(selected)),
        "useful_capture": pct(selected_useful, useful_total),
        "rejected_harmful_rate": pct(rejected_harmful, len(rejected)),
    }


def class_reliability(rows: list[dict[str, str]]) -> tuple[dict[int, float], dict[int, float]]:
    agree_rows = [row for row in rows if parse_bool(row["agree"])]
    global_source = mean(parse_float(row, "source_conf") for row in agree_rows) if agree_rows else 1.0
    global_clip = mean(parse_float(row, "clip_conf") for row in agree_rows) if agree_rows else 1.0

    source_values: dict[int, list[float]] = defaultdict(list)
    clip_values: dict[int, list[float]] = defaultdict(list)
    for row in agree_rows:
        cls = int(row["source_pred"])
        source_values[cls].append(parse_float(row, "source_conf"))
        clip_values[cls].append(parse_float(row, "clip_conf"))

    source_rel = {cls: mean(values) for cls, values in source_values.items()}
    clip_rel = {cls: mean(values) for cls, values in clip_values.items()}
    source_rel[-1] = global_source
    clip_rel[-1] = global_clip
    return source_rel, clip_rel


def rel_lookup(rel: dict[int, float], cls: int) -> float:
    return rel.get(cls, rel[-1])


def analyze_file(path: Path, score_gap: float, conf_gap: float) -> dict[str, float | int | str]:
    rows = load_rows(path)
    conflicts = [row for row in rows if not parse_bool(row["agree"])]
    useful_conflicts = [
        row for row in conflicts
        if parse_bool(row["source_correct"]) != parse_bool(row["clip_correct"])
    ]
    source_rel, clip_rel = class_reliability(rows)

    always_source = ["source"] * len(conflicts)
    always_clip = ["clip"] * len(conflicts)
    higher_conf = [
        "source" if parse_float(row, "source_conf") >= parse_float(row, "clip_conf") else "clip"
        for row in conflicts
    ]
    higher_conf_reject = []
    classwise = []
    classwise_reject = []

    for row in conflicts:
        source_conf = parse_float(row, "source_conf")
        clip_conf = parse_float(row, "clip_conf")
        source_cls = int(row["source_pred"])
        clip_cls = int(row["clip_pred"])
        source_score = source_conf * rel_lookup(source_rel, source_cls)
        clip_score = clip_conf * rel_lookup(clip_rel, clip_cls)

        if abs(source_conf - clip_conf) < conf_gap:
            higher_conf_reject.append(None)
        else:
            higher_conf_reject.append("source" if source_conf >= clip_conf else "clip")

        classwise.append("source" if source_score >= clip_score else "clip")
        if abs(source_score - clip_score) < score_gap:
            classwise_reject.append(None)
        else:
            classwise_reject.append("source" if source_score >= clip_score else "clip")

    always_source_eval = selector_eval(conflicts, always_source)
    always_clip_eval = selector_eval(conflicts, always_clip)
    higher_conf_eval = selector_eval(conflicts, higher_conf)
    higher_conf_reject_eval = selector_eval(conflicts, higher_conf_reject)
    classwise_eval = selector_eval(conflicts, classwise)
    classwise_reject_eval = selector_eval(conflicts, classwise_reject)

    return {
        "task": task_name(path),
        "total_samples": len(rows),
        "conflict_samples": len(conflicts),
        "conflict_rate": pct(len(conflicts), len(rows)),
        "useful_conflict_samples": len(useful_conflicts),
        "useful_conflict_rate": pct(len(useful_conflicts), len(conflicts)),
        "always_source_acc": always_source_eval["accuracy"],
        "always_clip_acc": always_clip_eval["accuracy"],
        "higher_conf_acc": higher_conf_eval["accuracy"],
        "higher_conf_reject_acc": higher_conf_reject_eval["accuracy"],
        "higher_conf_reject_coverage": higher_conf_reject_eval["coverage"],
        "classwise_acc": classwise_eval["accuracy"],
        "classwise_reject_acc": classwise_reject_eval["accuracy"],
        "classwise_reject_coverage": classwise_reject_eval["coverage"],
        "classwise_reject_useful_capture": classwise_reject_eval["useful_capture"],
        "classwise_reject_harmful_in_rejected": classwise_reject_eval["rejected_harmful_rate"],
    }


def write_markdown(rows: list[dict[str, float | int | str]], path: Path) -> None:
    columns = [
        "task",
        "conflict_rate",
        "useful_conflict_rate",
        "always_source_acc",
        "always_clip_acc",
        "higher_conf_acc",
        "classwise_acc",
        "classwise_reject_acc",
        "classwise_reject_coverage",
    ]
    lines = [
        "# Conflict Reliability Analysis",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row[col]
            vals.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No diagnostic CSV files matched: {args.glob}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [analyze_file(path, args.score_gap, args.conf_gap) for path in paths]
    csv_path = out_dir / "conflict_reliability_analysis.csv"
    json_path = out_dir / "conflict_reliability_analysis.json"
    md_path = out_dir / "conflict_reliability_analysis.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    macro = {}
    for key in rows[0]:
        if key == "task":
            continue
        values = [row[key] for row in rows if isinstance(row[key], (int, float))]
        macro[key] = mean(values)

    json_path.write_text(json.dumps({"tasks": rows, "macro": macro}, indent=2))
    write_markdown(rows, md_path)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")
    print(json.dumps({"macro": macro}, indent=2))


if __name__ == "__main__":
    main()
