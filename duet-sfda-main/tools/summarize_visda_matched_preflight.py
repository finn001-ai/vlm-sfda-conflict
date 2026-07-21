#!/usr/bin/env python
"""Gate a short VisDA-C run against the matched prefix of the full baseline."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

try:
    from tools.summarize_visda_temporal_precision_head import parse_records
except ModuleNotFoundError:
    from summarize_visda_temporal_precision_head import parse_records


MIX_PATTERN = re.compile(r"TARGET_HEAD_MIX:\s*([0-9.]+)")
INT_CONFIG_PATTERNS = {
    "pl_stable_cycles": re.compile(r"PL_STABLE_CYCLES:\s*(\d+)"),
    "gtr_stable_cycles": re.compile(r"GTR_STABLE_CYCLES:\s*(\d+)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-glob", required=True)
    parser.add_argument("--candidate-glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--matched-start-cycle", type=int, default=1)
    parser.add_argument("--matched-cycles", type=int, default=4)
    parser.add_argument("--min-improvement", type=float, default=0.25)
    parser.add_argument("--required-full-peak", type=float, default=91.4)
    parser.add_argument("--expected-baseline-mix", type=float, default=0.3)
    parser.add_argument("--expected-candidate-mix", type=float, default=0.4)
    parser.add_argument("--expected-baseline-pl-stable-cycles", type=int)
    parser.add_argument("--expected-candidate-pl-stable-cycles", type=int)
    parser.add_argument("--expected-baseline-gtr-stable-cycles", type=int)
    parser.add_argument("--expected-candidate-gtr-stable-cycles", type=int)
    parser.add_argument("--dynamics-json")
    parser.add_argument(
        "--pass-command",
        default="bash tools/run_visda_temporal_precision_head_mix040_seed2020.sh",
    )
    parser.add_argument(
        "--fail-next",
        default=(
            "do not run the full mix-0.4 job; next test temporal stability "
            "with PL/GTR stable cycles 3"
        ),
    )
    return parser.parse_args()


def read_one(pattern: str) -> str:
    paths = sorted(glob.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one log for {pattern!r}, found {len(paths)}")
    return Path(paths[0]).read_text(errors="ignore")


def read_mix(text: str) -> float:
    match = MIX_PATTERN.search(text)
    if match is None:
        raise ValueError("Training log does not contain TARGET_HEAD_MIX")
    return float(match.group(1))


def read_int_config(text: str, name: str) -> int:
    match = INT_CONFIG_PATTERNS[name].search(text)
    if match is None:
        raise ValueError(f"Training log does not contain {name}")
    return int(match.group(1))


def summarize_preflight(
    baseline_records: list[dict],
    candidate_records: list[dict],
    baseline_mix: float,
    candidate_mix: float,
    matched_cycles: int = 4,
    min_improvement: float = 0.25,
    required_full_peak: float = 91.4,
    expected_baseline_mix: float = 0.3,
    expected_candidate_mix: float = 0.4,
    matched_start_cycle: int = 1,
    baseline_pl_stable_cycles: int | None = None,
    candidate_pl_stable_cycles: int | None = None,
    baseline_gtr_stable_cycles: int | None = None,
    candidate_gtr_stable_cycles: int | None = None,
    expected_baseline_pl_stable_cycles: int | None = None,
    expected_candidate_pl_stable_cycles: int | None = None,
    expected_baseline_gtr_stable_cycles: int | None = None,
    expected_candidate_gtr_stable_cycles: int | None = None,
    mechanism_valid: bool = True,
    pass_command: str = "bash tools/run_visda_temporal_precision_head_mix040_seed2020.sh",
    fail_next: str = (
        "do not run the full mix-0.4 job; next test temporal stability "
        "with PL/GTR stable cycles 3"
    ),
) -> dict:
    if not 1 <= matched_start_cycle <= matched_cycles:
        raise ValueError("Matched cycle window must be positive and ordered")
    baseline_window = [
        row
        for row in baseline_records
        if matched_start_cycle <= row["cycle"] <= matched_cycles
    ]
    candidate_window = [
        row
        for row in candidate_records
        if matched_start_cycle <= row["cycle"] <= matched_cycles
    ]
    if not baseline_window:
        raise ValueError("Baseline has no records in the matched cycle range")
    if not candidate_window:
        raise ValueError("Candidate has no records in the matched cycle range")
    if max(row["cycle"] for row in candidate_records) != matched_cycles:
        raise ValueError("Candidate preflight did not finish the matched cycle budget")
    if any(row["max_cycle"] != matched_cycles for row in candidate_records):
        raise ValueError("Candidate log is not a clean matched-cycle preflight")

    baseline_matched_peak = max(row["accuracy"] for row in baseline_window)
    baseline_full_peak = max(row["accuracy"] for row in baseline_records)
    candidate_matched_peak = max(row["accuracy"] for row in candidate_window)
    matched_improvement = candidate_matched_peak - baseline_matched_peak
    baseline_late_gain = baseline_full_peak - baseline_matched_peak
    projected_full_peak = candidate_matched_peak + baseline_late_gain

    config_checks = [
        abs(baseline_mix - expected_baseline_mix) < 1e-9,
        abs(candidate_mix - expected_candidate_mix) < 1e-9,
    ]
    stable_cycle_pairs = (
        (baseline_pl_stable_cycles, expected_baseline_pl_stable_cycles),
        (candidate_pl_stable_cycles, expected_candidate_pl_stable_cycles),
        (baseline_gtr_stable_cycles, expected_baseline_gtr_stable_cycles),
        (candidate_gtr_stable_cycles, expected_candidate_gtr_stable_cycles),
    )
    config_checks.extend(
        actual == expected
        for actual, expected in stable_cycle_pairs
        if expected is not None
    )
    checks = {
        "config_valid": all(config_checks),
        "mechanism_valid": mechanism_valid,
        "matched_improvement": matched_improvement >= min_improvement,
        "projected_to_beat_reference": projected_full_peak >= required_full_peak,
    }
    passed = all(checks.values())
    return {
        "decision": "pass_full_training_gate" if passed else "fail_full_training_gate",
        "metric": "mean per-class accuracy",
        "selection_warning": "all peaks use VisDA validation labels and are oracle diagnostics",
        "matched_start_cycle": matched_start_cycle,
        "matched_cycles": matched_cycles,
        "baseline_target_head_mix": baseline_mix,
        "candidate_target_head_mix": candidate_mix,
        "baseline_pl_stable_cycles": baseline_pl_stable_cycles,
        "candidate_pl_stable_cycles": candidate_pl_stable_cycles,
        "baseline_gtr_stable_cycles": baseline_gtr_stable_cycles,
        "candidate_gtr_stable_cycles": candidate_gtr_stable_cycles,
        "baseline_matched_peak": round(baseline_matched_peak, 4),
        "candidate_matched_peak": round(candidate_matched_peak, 4),
        "matched_improvement": round(matched_improvement, 4),
        "minimum_improvement": min_improvement,
        "baseline_full_peak": round(baseline_full_peak, 4),
        "baseline_late_gain": round(baseline_late_gain, 4),
        "projected_candidate_full_peak": round(projected_full_peak, 4),
        "required_full_peak": required_full_peak,
        "checks": checks,
        "next": pass_command if passed else fail_next,
    }


def main() -> None:
    args = parse_args()
    baseline_text = read_one(args.baseline_glob)
    candidate_text = read_one(args.candidate_glob)
    mechanism_valid = True
    if args.dynamics_json:
        dynamics = json.loads(Path(args.dynamics_json).read_text())
        mechanism_valid = dynamics.get("decision") == "pass_training_gate"
    summary = summarize_preflight(
        parse_records(baseline_text),
        parse_records(candidate_text),
        read_mix(baseline_text),
        read_mix(candidate_text),
        matched_cycles=args.matched_cycles,
        min_improvement=args.min_improvement,
        required_full_peak=args.required_full_peak,
        expected_baseline_mix=args.expected_baseline_mix,
        expected_candidate_mix=args.expected_candidate_mix,
        matched_start_cycle=args.matched_start_cycle,
        baseline_pl_stable_cycles=read_int_config(
            baseline_text, "pl_stable_cycles"
        ),
        candidate_pl_stable_cycles=read_int_config(
            candidate_text, "pl_stable_cycles"
        ),
        baseline_gtr_stable_cycles=read_int_config(
            baseline_text, "gtr_stable_cycles"
        ),
        candidate_gtr_stable_cycles=read_int_config(
            candidate_text, "gtr_stable_cycles"
        ),
        expected_baseline_pl_stable_cycles=(
            args.expected_baseline_pl_stable_cycles
        ),
        expected_candidate_pl_stable_cycles=(
            args.expected_candidate_pl_stable_cycles
        ),
        expected_baseline_gtr_stable_cycles=(
            args.expected_baseline_gtr_stable_cycles
        ),
        expected_candidate_gtr_stable_cycles=(
            args.expected_candidate_gtr_stable_cycles
        ),
        mechanism_valid=mechanism_valid,
        pass_command=args.pass_command,
        fail_next=args.fail_next,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
