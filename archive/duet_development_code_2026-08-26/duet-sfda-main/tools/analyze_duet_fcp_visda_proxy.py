#!/usr/bin/env python
"""Gate DUET-FCP against the matched original-DUET VisDA proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HARD_CLASS_INDICES = (3, 7, 11)  # car, person, truck


def mean(values):
    values = tuple(values)
    return sum(values) / len(values)


def analyze(
    control,
    candidate,
    *,
    min_final_improvement=0.15,
    max_hard_mean_regression=0.0,
    max_other9_regression=0.1,
    max_hard_class_regression=0.5,
):
    control_final = control["final"]
    candidate_final = candidate["final"]
    control_classes = control_final["class_accuracy"]
    candidate_classes = candidate_final["class_accuracy"]
    if len(control_classes) != 12 or len(candidate_classes) != 12:
        raise ValueError("VisDA summaries must contain 12 class accuracies")

    hard_deltas = {
        name: round(candidate_classes[index] - control_classes[index], 4)
        for name, index in zip(("car", "person", "truck"), HARD_CLASS_INDICES)
    }
    other_indices = [
        index for index in range(12) if index not in HARD_CLASS_INDICES
    ]
    final_delta = candidate_final["accuracy"] - control_final["accuracy"]
    hard_mean_delta = mean(
        candidate_classes[index] for index in HARD_CLASS_INDICES
    ) - mean(control_classes[index] for index in HARD_CLASS_INDICES)
    other9_delta = mean(
        candidate_classes[index] for index in other_indices
    ) - mean(control_classes[index] for index in other_indices)

    checks = {
        "matched_contract": (
            control.get("num_checkpoints") == 16
            and candidate.get("num_checkpoints") == 16
            and control_final.get("cycle") == 4
            and candidate_final.get("cycle") == 4
        ),
        "final_improvement": final_delta >= min_final_improvement,
        "hard_mean_noninferior": hard_mean_delta >= -max_hard_mean_regression,
        "other9_noninferior": other9_delta >= -max_other9_regression,
        "no_hard_class_compensation": all(
            delta >= -max_hard_class_regression
            for delta in hard_deltas.values()
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "pass_duet_fcp_proxy_gate"
            if passed
            else "fail_duet_fcp_proxy_gate"
        ),
        "metric": "VisDA mean per-class accuracy at final checkpoint",
        "control_final": control_final["accuracy"],
        "candidate_final": candidate_final["accuracy"],
        "final_delta": round(final_delta, 4),
        "hard_mean_delta": round(hard_mean_delta, 4),
        "other9_delta": round(other9_delta, 4),
        "hard_class_deltas": hard_deltas,
        "thresholds": {
            "min_final_improvement": min_final_improvement,
            "max_hard_mean_regression": max_hard_mean_regression,
            "max_other9_regression": max_other9_regression,
            "max_hard_class_regression": max_hard_class_regression,
        },
        "checks": checks,
        "next": (
            "prepare one full VisDA DUET-FCP seed"
            if passed
            else "stop DUET-FCP; do not run a full VisDA job"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    control = json.loads(Path(args.control_summary).read_text())
    candidate = json.loads(Path(args.candidate_summary).read_text())
    report = analyze(control, candidate)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
