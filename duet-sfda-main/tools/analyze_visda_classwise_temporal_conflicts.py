#!/usr/bin/env python
"""Diagnose whether VisDA temporal corrections support classwise routing."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

try:
    from tools.analyze_temporal_conflict_dynamics import load_cycles, pct, teacher_label
except ModuleNotFoundError:
    from analyze_temporal_conflict_dynamics import load_cycles, pct, teacher_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--class-names", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--min-selected", type=int, default=100)
    parser.add_argument("--min-gain-pp", type=float, default=1.0)
    parser.add_argument("--min-win-rate", type=float, default=60.0)
    return parser.parse_args()


def masked_accuracy(pred: np.ndarray, label: np.ndarray, mask: np.ndarray) -> float:
    count = int(np.sum(mask))
    return pct(np.sum(pred[mask] == label[mask]), count)


def class_row(
    view: str,
    class_index: int,
    class_name: str,
    class_mask: np.ndarray,
    label: np.ndarray,
    initial_conflict: np.ndarray,
    selected: np.ndarray,
    source: np.ndarray,
    clip: np.ndarray,
    teacher: np.ndarray,
    min_selected: int,
    min_gain_pp: float,
    min_win_rate: float,
) -> dict[str, object]:
    selected_mask = class_mask & selected
    conflict_mask = class_mask & initial_conflict
    teacher_correct = teacher == label
    clip_correct = clip == label
    corrections = selected_mask & teacher_correct & (~clip_correct)
    degradations = selected_mask & (~teacher_correct) & clip_correct
    correction_count = int(np.sum(corrections))
    degradation_count = int(np.sum(degradations))
    discordant = correction_count + degradation_count
    win_rate = pct(correction_count, discordant)
    teacher_selected_accuracy = masked_accuracy(teacher, label, selected_mask)
    clip_selected_accuracy = masked_accuracy(clip, label, selected_mask)
    gain = teacher_selected_accuracy - clip_selected_accuracy
    selected_count = int(np.sum(selected_mask))
    routable = (
        selected_count >= min_selected
        and gain >= min_gain_pp
        and correction_count > degradation_count
        and win_rate >= min_win_rate
    )
    return {
        "view": view,
        "class_index": class_index,
        "class": class_name,
        "samples": int(np.sum(class_mask)),
        "initial_conflicts": int(np.sum(conflict_mask)),
        "stable_selected": selected_count,
        "stable_conflict_coverage": pct(selected_count, int(np.sum(conflict_mask))),
        "final_source_accuracy": masked_accuracy(source, label, class_mask),
        "final_clip_accuracy": masked_accuracy(clip, label, class_mask),
        "final_teacher_accuracy": masked_accuracy(teacher, label, class_mask),
        "stable_source_accuracy": masked_accuracy(source, label, selected_mask),
        "stable_clip_accuracy": clip_selected_accuracy,
        "stable_teacher_accuracy": teacher_selected_accuracy,
        "teacher_minus_clip_pp": round(gain, 4),
        "corrections_over_clip": correction_count,
        "degradations_from_clip": degradation_count,
        "net_correct_gain": correction_count - degradation_count,
        "discordant_win_rate": win_rate,
        "supports_class_route": bool(routable),
    }


def analyze_cycles(
    cycles: list[dict[str, np.ndarray]],
    class_names: list[str],
    min_selected: int = 100,
    min_gain_pp: float = 1.0,
    min_win_rate: float = 60.0,
) -> dict[str, object]:
    if len(cycles) < 2:
        raise ValueError("Classwise temporal analysis needs at least two cycles")
    first, previous, final = cycles[0], cycles[-2], cycles[-1]
    label = final["target_label"].astype(np.int64)
    if len(class_names) != int(label.max()) + 1:
        raise ValueError("Class names do not cover all target labels")

    source = final["source_label"].astype(np.int64)
    clip = final["clip_label"].astype(np.int64)
    teacher = teacher_label(final).astype(np.int64)
    initial_conflict = first["source_label"] != first["clip_label"]
    stable = teacher_label(previous) == teacher
    selected = initial_conflict & stable

    true_rows = []
    predicted_rows = []
    for index, name in enumerate(class_names):
        true_rows.append(
            class_row(
                "true_class",
                index,
                name,
                label == index,
                label,
                initial_conflict,
                selected,
                source,
                clip,
                teacher,
                min_selected,
                min_gain_pp,
                min_win_rate,
            )
        )
        predicted_rows.append(
            class_row(
                "predicted_teacher_class",
                index,
                name,
                teacher == index,
                label,
                initial_conflict,
                selected,
                source,
                clip,
                teacher,
                min_selected,
                min_gain_pp,
                min_win_rate,
            )
        )

    eligible_routes = [row for row in predicted_rows if row["stable_selected"] >= min_selected]
    supported_routes = [row for row in eligible_routes if row["supports_class_route"]]
    non_supported_routes = [row for row in eligible_routes if not row["supports_class_route"]]
    supports_conditional = bool(supported_routes and non_supported_routes)

    global_mask = np.ones(label.shape[0], dtype=bool)
    global_row = class_row(
        "global",
        -1,
        "all",
        global_mask,
        label,
        initial_conflict,
        selected,
        source,
        clip,
        teacher,
        min_selected,
        min_gain_pp,
        min_win_rate,
    )
    return {
        "decision": (
            "supports_class_conditional_routing"
            if supports_conditional
            else "does_not_support_class_conditional_routing"
        ),
        "selection_warning": (
            "uses VisDA validation labels and is mechanism-diagnostic only; "
            "do not use these rows directly as an unsupervised routing oracle"
        ),
        "cycles": len(cycles),
        "criteria": {
            "min_selected": min_selected,
            "min_gain_pp": min_gain_pp,
            "min_discordant_win_rate": min_win_rate,
            "requires_both_supported_and_unsupported_predicted_classes": True,
        },
        "global": global_row,
        "supported_predicted_classes": [row["class"] for row in supported_routes],
        "unsupported_predicted_classes": [row["class"] for row in non_supported_routes],
        "true_class_diagnostics": true_rows,
        "predicted_class_routes": predicted_rows,
        "next": (
            "develop an unlabeled reliability-estimated class-conditional temporal residual"
            if supports_conditional
            else "do not build a class router; test PL/GTR stable cycles 3"
        ),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least two cycle files: {args.glob}")
    class_names = [
        line.strip()
        for line in Path(args.class_names).read_text().splitlines()
        if line.strip()
    ]
    result = analyze_cycles(
        load_cycles(paths),
        class_names,
        min_selected=args.min_selected,
        min_gain_pp=args.min_gain_pp,
        min_win_rate=args.min_win_rate,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    rows = result["true_class_diagnostics"] + result["predicted_class_routes"]
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
