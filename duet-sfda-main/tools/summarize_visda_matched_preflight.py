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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-glob", required=True)
    parser.add_argument("--candidate-glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--matched-cycles", type=int, default=4)
    parser.add_argument("--min-improvement", type=float, default=0.25)
    parser.add_argument("--required-full-peak", type=float, default=91.4)
    parser.add_argument("--expected-baseline-mix", type=float, default=0.3)
    parser.add_argument("--expected-candidate-mix", type=float, default=0.4)
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
) -> dict:
    baseline_prefix = [row for row in baseline_records if row["cycle"] <= matched_cycles]
    if not baseline_prefix:
        raise ValueError("Baseline has no records in the matched cycle range")
    if max(row["cycle"] for row in candidate_records) != matched_cycles:
        raise ValueError("Candidate preflight did not finish the matched cycle budget")
    if any(row["max_cycle"] != matched_cycles for row in candidate_records):
        raise ValueError("Candidate log is not a clean matched-cycle preflight")

    baseline_matched_peak = max(row["accuracy"] for row in baseline_prefix)
    baseline_full_peak = max(row["accuracy"] for row in baseline_records)
    candidate_matched_peak = max(row["accuracy"] for row in candidate_records)
    matched_improvement = candidate_matched_peak - baseline_matched_peak
    baseline_late_gain = baseline_full_peak - baseline_matched_peak
    projected_full_peak = candidate_matched_peak + baseline_late_gain

    checks = {
        "config_valid": (
            abs(baseline_mix - expected_baseline_mix) < 1e-9
            and abs(candidate_mix - expected_candidate_mix) < 1e-9
        ),
        "matched_improvement": matched_improvement >= min_improvement,
        "projected_to_beat_reference": projected_full_peak >= required_full_peak,
    }
    passed = all(checks.values())
    return {
        "decision": "pass_full_training_gate" if passed else "fail_full_training_gate",
        "metric": "mean per-class accuracy",
        "selection_warning": "all peaks use VisDA validation labels and are oracle diagnostics",
        "matched_cycles": matched_cycles,
        "baseline_target_head_mix": baseline_mix,
        "candidate_target_head_mix": candidate_mix,
        "baseline_matched_peak": round(baseline_matched_peak, 4),
        "candidate_matched_peak": round(candidate_matched_peak, 4),
        "matched_improvement": round(matched_improvement, 4),
        "minimum_improvement": min_improvement,
        "baseline_full_peak": round(baseline_full_peak, 4),
        "baseline_late_gain": round(baseline_late_gain, 4),
        "projected_candidate_full_peak": round(projected_full_peak, 4),
        "required_full_peak": required_full_peak,
        "checks": checks,
        "next": (
            "run tools/run_visda_temporal_precision_head_mix040_seed2020.sh"
            if passed
            else "do not run the full mix-0.4 job; next test temporal stability with PL/GTR stable cycles 3"
        ),
    }


def main() -> None:
    args = parse_args()
    baseline_text = read_one(args.baseline_glob)
    candidate_text = read_one(args.candidate_glob)
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
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
