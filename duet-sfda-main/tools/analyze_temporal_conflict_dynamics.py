#!/usr/bin/env python
"""Analyze temporal prediction dynamics on source/CLIP conflict samples."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover - scipy is available in the cloud env.
    binomtest = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/temporal_probe/temporal_diagnostics/*_cycle*.npz",
    )
    parser.add_argument(
        "--out",
        default="output/uda/office-home/temporal_conflict_dynamics_probe.json",
    )
    parser.add_argument("--min-coverage", type=float, default=5.0)
    parser.add_argument("--min-pass-tasks", type=int, default=2)
    return parser.parse_args()


def pct(value: int | float, denom: int) -> float:
    return round(100.0 * float(value) / denom, 4) if denom else 0.0


def accuracy(pred: np.ndarray, label: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(label.shape[0], dtype=bool)
    return pct(np.sum(pred[mask] == label[mask]), int(np.sum(mask)))


def task_from_path(path: Path) -> str:
    return path.stem.split("_cycle", 1)[0]


def load_cycles(paths: list[Path]) -> list[dict[str, np.ndarray]]:
    cycles = []
    for path in paths:
        data = np.load(path)
        cycles.append({key: data[key] for key in data.files})
    cycles.sort(key=lambda item: int(item["cycle"]))
    return cycles


def teacher_label(item: dict[str, np.ndarray]) -> np.ndarray:
    return item["teacher_label"] if "teacher_label" in item else item["mix_label"]


def analyze_task(task: str, paths: list[Path], min_coverage: float) -> dict[str, object]:
    cycles = load_cycles(paths)
    if len(cycles) < 2:
        raise ValueError(f"{task} needs at least two diagnostic cycles")

    first = cycles[0]
    final = cycles[-1]
    prev = cycles[-2]
    label = final["target_label"].astype(np.int64)
    initial_conflict = first["source_label"] != first["clip_label"]
    final_conflict = final["source_label"] != final["clip_label"]
    prev_teacher = teacher_label(prev)
    final_teacher = teacher_label(final)
    stable_mix = prev_teacher == final_teacher
    selected = initial_conflict & stable_mix

    final_mix = final_teacher
    final_clip = final["clip_label"]
    final_source = final["source_label"]
    first_clip = first["clip_label"]
    first_source = first["source_label"]
    selected_count = int(np.sum(selected))
    initial_conflict_count = int(np.sum(initial_conflict))

    mix_correct = final_mix[selected] == label[selected]
    clip_correct = final_clip[selected] == label[selected]
    corrections = mix_correct & (~clip_correct)
    degradations = (~mix_correct) & clip_correct
    net_gain = int(np.sum(corrections) - np.sum(degradations))
    discordant = int(np.sum(corrections) + np.sum(degradations))
    p_value = None
    if binomtest is not None and discordant:
        p_value = float(
            binomtest(
                int(np.sum(corrections)),
                discordant,
                p=0.5,
                alternative="greater",
            ).pvalue
        )

    outside_final_candidates = selected & (final_mix != final_source) & (final_mix != final_clip)
    cycle_metrics = []
    for item in cycles:
        cycle_label = item["target_label"].astype(np.int64)
        cycle_metrics.append(
            {
                "cycle": int(item["cycle"]),
                "source_accuracy": accuracy(item["source_label"], cycle_label),
                "clip_accuracy": accuracy(item["clip_label"], cycle_label),
                "mix_accuracy": accuracy(item["mix_label"], cycle_label),
                "teacher_accuracy": accuracy(teacher_label(item), cycle_label),
                "agreement_rate": pct(
                    np.sum(item["source_label"] == item["clip_label"]),
                    cycle_label.shape[0],
                ),
                "valid_pseudo_labels": int(np.sum(item["label_mask"])),
                "valid_pseudo_label_accuracy": accuracy(
                    item["mix_label"], cycle_label, item["label_mask"].astype(bool)
                ),
            }
        )

    selected_coverage = pct(selected_count, initial_conflict_count)
    pass_task_gate = (
        selected_coverage >= min_coverage
        and net_gain > 0
        and accuracy(final_mix, label, selected) > accuracy(final_clip, label, selected)
        and (p_value is None or p_value < 0.05)
    )

    return {
        "task": task,
        "cycles": len(cycles),
        "samples": int(label.shape[0]),
        "initial_conflicts": initial_conflict_count,
        "initial_conflict_rate": pct(initial_conflict_count, label.shape[0]),
        "final_conflicts": int(np.sum(final_conflict)),
        "final_conflict_rate": pct(np.sum(final_conflict), label.shape[0]),
        "stable_mix_on_initial_conflicts": selected_count,
        "stable_mix_conflict_coverage": selected_coverage,
        "first_source_accuracy_on_initial_conflicts": accuracy(first_source, label, initial_conflict),
        "first_clip_accuracy_on_initial_conflicts": accuracy(first_clip, label, initial_conflict),
        "final_source_accuracy_on_initial_conflicts": accuracy(final_source, label, initial_conflict),
        "final_clip_accuracy_on_initial_conflicts": accuracy(final_clip, label, initial_conflict),
        "final_mix_accuracy_on_initial_conflicts": accuracy(final_mix, label, initial_conflict),
        "stable_mix_accuracy": accuracy(final_mix, label, selected),
        "stable_clip_accuracy": accuracy(final_clip, label, selected),
        "stable_source_accuracy": accuracy(final_source, label, selected),
        "corrections_over_final_clip": int(np.sum(corrections)),
        "degradations_from_final_clip": int(np.sum(degradations)),
        "net_correct_gain_over_final_clip": net_gain,
        "discordant_decisions": discordant,
        "one_sided_correction_p_value": round(p_value, 6) if p_value is not None else None,
        "outside_final_candidate_selected": int(np.sum(outside_final_candidates)),
        "outside_final_candidate_accuracy": accuracy(final_mix, label, outside_final_candidates),
        "cycle_metrics": cycle_metrics,
        "pass_task_gate": bool(pass_task_gate),
    }


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[Path]] = defaultdict(list)
    for raw_path in glob.glob(args.glob):
        path = Path(raw_path)
        grouped[task_from_path(path)].append(path)
    if not grouped:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    tasks = [
        analyze_task(task, sorted(paths), args.min_coverage)
        for task, paths in sorted(grouped.items())
    ]
    pass_tasks = sum(1 for task in tasks if task["pass_task_gate"])
    output = {
        "config": vars(args),
        "decision": "pass_training_gate"
        if pass_tasks >= args.min_pass_tasks
        else "fail_training_gate",
        "reason": (
            "Temporal stable predictions beat final CLIP on enough conflict tasks."
            if pass_tasks >= args.min_pass_tasks
            else "Temporal stable predictions do not beat final CLIP reliably enough."
        ),
        "pass_tasks": int(pass_tasks),
        "tasks": tasks,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
