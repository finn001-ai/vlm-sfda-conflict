#!/usr/bin/env python
"""Gate the attribute-reliability candidate against arithmetic DUET proxy25."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HARD_CLASS_INDICES = (3, 7, 11)  # car, person, truck
OTHER_CLASS_INDICES = (0, 1, 2, 4, 5, 6, 8, 9, 10)


def _mean(values):
    values = tuple(values)
    return sum(values) / len(values)


def analyze(
    control,
    candidate,
    *,
    min_final_improvement=0.20,
    max_subgroup_regression=0.0,
    control_provenance="matched_current_source_and_proxy_hashes",
):
    control_final = control["final"]
    candidate_final = candidate["final"]
    control_classes = control_final["class_accuracy"]
    candidate_classes = candidate_final["class_accuracy"]
    if len(control_classes) != 12 or len(candidate_classes) != 12:
        raise ValueError("VisDA summaries must contain 12 class accuracies")

    class_deltas = [
        candidate_value - control_value
        for candidate_value, control_value in zip(
            candidate_classes,
            control_classes,
        )
    ]
    final_delta = candidate_final["accuracy"] - control_final["accuracy"]
    hard_mean_delta = _mean(class_deltas[index] for index in HARD_CLASS_INDICES)
    other9_mean_delta = _mean(class_deltas[index] for index in OTHER_CLASS_INDICES)
    car_truck_mean_delta = _mean(class_deltas[index] for index in (3, 11))
    hard_deltas = {
        name: round(class_deltas[index], 4)
        for name, index in zip(("car", "person", "truck"), HARD_CLASS_INDICES)
    }
    checks = {
        "matched_four_cycle_contract": (
            control.get("num_checkpoints") == 16
            and candidate.get("num_checkpoints") == 16
            and control_final.get("cycle") == 4
            and candidate_final.get("cycle") == 4
        ),
        "final_macro_improvement_at_least_0.20pp": (
            final_delta >= min_final_improvement
        ),
        "car_person_truck_mean_noninferior": (
            hard_mean_delta >= -max_subgroup_regression
        ),
        "other_nine_mean_noninferior": (other9_mean_delta >= -max_subgroup_regression),
    }
    passed = all(checks.values())
    limitations = []
    if control_provenance == "archived_control_without_source_or_proxy_hashes":
        limitations.append(
            "The archived control has checksummed logs and a fixed run contract, "
            "but its source-checkpoint and proxy-list byte hashes were not archived."
        )
    return {
        "decision": (
            "PASS_ATTRIBUTE_RELIABILITY_PROXY_GATE"
            if passed
            else "REJECT_ATTRIBUTE_RELIABILITY_PROXY"
        ),
        "metric": "VisDA mean per-class accuracy at the final checkpoint",
        "control_provenance": control_provenance,
        "control_provenance_limitations": limitations,
        "control_final": control_final["accuracy"],
        "candidate_final": candidate_final["accuracy"],
        "final_delta": round(final_delta, 4),
        "hard_mean_delta": round(hard_mean_delta, 4),
        "other9_mean_delta": round(other9_mean_delta, 4),
        "car_truck_mean_delta": round(car_truck_mean_delta, 4),
        "hard_class_deltas": hard_deltas,
        "car_truck_exchange_observed": class_deltas[3] * class_deltas[11] < 0.0,
        "thresholds": {
            "min_final_macro_improvement_pp": min_final_improvement,
            "max_hard_or_other9_mean_regression_pp": max_subgroup_regression,
            "selection_uses_oracle_peak": False,
        },
        "checks": checks,
        "next": (
            "eligible for one separately approved matched full VisDA run"
            if passed
            else "stop; do not run full VisDA"
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument(
        "--control-provenance",
        choices=(
            "matched_current_source_and_proxy_hashes",
            "archived_control_without_source_or_proxy_hashes",
        ),
        default="matched_current_source_and_proxy_hashes",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
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
