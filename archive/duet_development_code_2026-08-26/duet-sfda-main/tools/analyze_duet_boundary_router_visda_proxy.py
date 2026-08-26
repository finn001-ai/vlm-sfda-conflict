#!/usr/bin/env python
"""Gate the DUET boundary router against a matched original-DUET proxy."""

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
    min_final_improvement=0.20,
    max_hard_mean_regression=0.0,
    control_provenance="matched_current_output_hashes",
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
    final_delta = candidate_final["accuracy"] - control_final["accuracy"]
    hard_mean_delta = mean(
        candidate_classes[index] for index in HARD_CLASS_INDICES
    ) - mean(control_classes[index] for index in HARD_CLASS_INDICES)
    car_truck_mean_delta = mean(
        candidate_classes[index] for index in (3, 11)
    ) - mean(control_classes[index] for index in (3, 11))
    car_truck_exchange = hard_deltas["car"] * hard_deltas["truck"] < 0.0

    checks = {
        "matched_contract": (
            control.get("num_checkpoints") == 16
            and candidate.get("num_checkpoints") == 16
            and control_final.get("cycle") == 4
            and candidate_final.get("cycle") == 4
        ),
        "final_improvement": final_delta >= min_final_improvement,
        "hard_mean_noninferior": hard_mean_delta >= -max_hard_mean_regression,
    }
    passed = all(checks.values())
    provenance_limitations = []
    if control_provenance == "archived_2026-07-23_source_and_list_hashes_unavailable":
        provenance_limitations.append(
            "The archived control preserves checksummed log/summary and its run "
            "contract, but not the historical source-checkpoint or proxy-list hashes."
        )
    return {
        "decision": (
            "pass_boundary_router_proxy_gate"
            if passed
            else "fail_boundary_router_proxy_gate"
        ),
        "metric": "VisDA mean per-class accuracy at final checkpoint",
        "control_provenance": control_provenance,
        "control_provenance_limitations": provenance_limitations,
        "control_final": control_final["accuracy"],
        "candidate_final": candidate_final["accuracy"],
        "final_delta": round(final_delta, 4),
        "hard_mean_delta": round(hard_mean_delta, 4),
        "car_truck_mean_delta": round(car_truck_mean_delta, 4),
        "hard_class_deltas": hard_deltas,
        "car_truck_exchange_observed": car_truck_exchange,
        "thresholds": {
            "min_final_improvement": min_final_improvement,
            "max_hard_mean_regression": max_hard_mean_regression,
        },
        "checks": checks,
        "next": (
            "prepare one matched full VisDA boundary-router/control pair"
            if passed
            else "stop boundary router; do not run a full VisDA job"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument(
        "--control-provenance",
        choices=(
            "matched_current_output_hashes",
            "archived_2026-07-23_source_and_list_hashes_unavailable",
        ),
        default="matched_current_output_hashes",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    control = json.loads(Path(args.control_summary).read_text())
    candidate = json.loads(Path(args.candidate_summary).read_text())
    report = analyze(
        control,
        candidate,
        control_provenance=args.control_provenance,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
