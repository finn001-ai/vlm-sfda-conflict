#!/usr/bin/env python
"""Summarize the full-data VisDA Stage14 dual-tier memory experiment."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

try:
    from tools.summarize_visda_stage14_prior_memory import (
        EXPECTED_FULL_VISDA_SAMPLES,
        parse_refresh_metrics,
    )
    from tools.summarize_visda_temporal_precision_head import (
        load_class_names,
        parse_records,
    )
except ModuleNotFoundError:
    from summarize_visda_stage14_prior_memory import (
        EXPECTED_FULL_VISDA_SAMPLES,
        parse_refresh_metrics,
    )
    from summarize_visda_temporal_precision_head import (
        load_class_names,
        parse_records,
    )


CONFIG_PATTERNS = {
    "calib_mode": re.compile(r"^\s+CALIB_MODE:\s*(\S+)\s*$", re.MULTILINE),
    "calib_power": re.compile(r"^\s+CALIB_POWER:\s*([0-9.eE+-]+)\s*$", re.MULTILINE),
    "pl_memory": re.compile(r"^\s+PL_MEMORY:\s*(\S+)\s*$", re.MULTILINE),
    "stable_cycles": re.compile(r"^\s+PL_STABLE_CYCLES:\s*(\d+)\s*$", re.MULTILINE),
    "stable_memory": re.compile(r"^\s+PL_STABLE_MEMORY:\s*(\S+)\s*$", re.MULTILINE),
    "warmup_cycles": re.compile(
        r"^\s+PL_MEMORY_WARMUP_CYCLES:\s*(\d+)\s*$", re.MULTILINE
    ),
    "min_conf": re.compile(
        r"^\s+PL_MEMORY_MIN_CONF:\s*([0-9.eE+-]+)\s*$",
        re.MULTILINE,
    ),
    "pending_weight": re.compile(
        r"^\s+PL_PENDING_WEIGHT:\s*([0-9.eE+-]+)\s*$",
        re.MULTILINE,
    ),
    "pl_expand": re.compile(r"^\s+PL_EXPAND:\s*(\S+)\s*$", re.MULTILINE),
    "pl_topk_per_class": re.compile(
        r"^\s+PL_TOPK_PER_CLASS:\s*(\d+)\s*$", re.MULTILINE
    ),
    "pl_class_balance": re.compile(
        r"^\s+PL_CLASS_BALANCE:\s*(True|False)\s*$", re.MULTILINE
    ),
    "target_head": re.compile(
        r"^\s+TARGET_HEAD_ADAPT:\s*(True|False)\s*$", re.MULTILINE
    ),
    "gtr_par": re.compile(r"^\s+GTR_PAR:\s*([0-9.eE+-]+)\s*$", re.MULTILINE),
}
MEMORY_PATTERN = re.compile(
    r"DCCL pseudo-label memory:\s*mode=dual_tier;\s*"
    r"stable_memory=(\S+);\s*warmup=(\d+);\s*"
    r"current=(\d+);\s*stable=(\d+);\s*pending=(\d+);\s*"
    r"conflict=(\d+);\s*low_confidence=(\d+);\s*"
    r"selected=(\d+);\s*effective_weight=([0-9.]+);\s*"
    r"pending_mean_weight=([0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duet-glob", required=True)
    parser.add_argument("--both-prior-stable-glob", required=True)
    parser.add_argument("--none-stable-glob", required=True)
    parser.add_argument("--both-prior-monotonic-glob", required=True)
    parser.add_argument("--none-monotonic-glob", required=True)
    parser.add_argument("--both-prior-dual-tier-glob", required=True)
    parser.add_argument("--none-dual-tier-glob", required=True)
    parser.add_argument("--class-names", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--min-memory-gain", type=float, default=0.10)
    parser.add_argument("--min-duet-gain", type=float, default=0.10)
    return parser.parse_args()


def round4(value: float) -> float:
    return round(float(value), 4)


def read_one_log(pattern: str, name: str) -> tuple[Path, str]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"{name}: expected exactly one clean log, found {len(paths)}")
    return paths[0], paths[0].read_text(errors="ignore")


def parse_config(text: str, name: str) -> dict:
    values = {}
    for key, pattern in CONFIG_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"{name}: log does not contain {key}")
        values[key] = match.group(1)
    return {
        "calib_mode": values["calib_mode"],
        "calib_power": float(values["calib_power"]),
        "pl_memory": values["pl_memory"],
        "stable_cycles": int(values["stable_cycles"]),
        "stable_memory": values["stable_memory"],
        "warmup_cycles": int(values["warmup_cycles"]),
        "min_conf": float(values["min_conf"]),
        "pending_weight": float(values["pending_weight"]),
        "pl_expand": values["pl_expand"],
        "pl_topk_per_class": int(values["pl_topk_per_class"]),
        "pl_class_balance": values["pl_class_balance"] == "True",
        "target_head": values["target_head"] == "True",
        "gtr_par": float(values["gtr_par"]),
    }


def validate_dual_config(config: dict, name: str, calib_mode: str) -> None:
    checks = {
        "calib_mode": config["calib_mode"] == calib_mode,
        "calib_power": abs(config["calib_power"] - 0.5) < 1e-12,
        "pl_memory": config["pl_memory"] == "dual_tier",
        "stable_cycles": config["stable_cycles"] == 2,
        "stable_memory": config["stable_memory"] == "reversible",
        "warmup_cycles": config["warmup_cycles"] == 1,
        "min_conf": abs(config["min_conf"]) < 1e-12,
        "pending_weight": abs(config["pending_weight"] - 0.5) < 1e-12,
        "pl_expand": config["pl_expand"] == "none",
        "pl_topk_per_class": config["pl_topk_per_class"] == 0,
        "pl_class_balance": config["pl_class_balance"] is False,
        "target_head": config["target_head"] is True,
        "gtr_par": abs(config["gtr_par"] - 0.05) < 1e-12,
    }
    failures = [key for key, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            f"{name}: fixed dual-tier config check failed for " + ", ".join(failures)
        )


def parse_memory_dynamics(text: str, name: str) -> list[dict]:
    matches = MEMORY_PATTERN.findall(text)
    if len(matches) != 4:
        raise ValueError(
            f"{name}: expected four dual-tier memory records, " f"found {len(matches)}"
        )
    rows = []
    for cycle, values in enumerate(matches, start=1):
        (
            stable_memory,
            warmup,
            current,
            stable,
            pending,
            conflict,
            low_confidence,
            selected,
            effective_weight,
            pending_mean_weight,
        ) = values
        row = {
            "cycle": cycle,
            "stable_memory": stable_memory,
            "warmup": bool(int(warmup)),
            "current": int(current),
            "stable": int(stable),
            "pending": int(pending),
            "conflict": int(conflict),
            "low_confidence": int(low_confidence),
            "selected": int(selected),
            "effective_weight": float(effective_weight),
            "pending_mean_weight": float(pending_mean_weight),
        }
        if row["current"] != row["stable"] + row["pending"]:
            raise ValueError(f"{name}: cycle {cycle} current != stable + pending")
        if row["selected"] != row["current"]:
            raise ValueError(f"{name}: cycle {cycle} selected != current")
        if row["conflict"] + row["current"] + row["low_confidence"] != (
            EXPECTED_FULL_VISDA_SAMPLES
        ):
            raise ValueError(f"{name}: cycle {cycle} state partition is incomplete")
        if cycle == 1 and not row["warmup"]:
            raise ValueError(f"{name}: cycle 1 must be the fixed warmup")
        if cycle > 1 and row["warmup"]:
            raise ValueError(f"{name}: warmup leaked beyond cycle 1")
        rows.append(row)
    return rows


def load_run(
    pattern: str,
    name: str,
    *,
    dual_calib_mode: str | None = None,
) -> dict:
    path, text = read_one_log(pattern, name)
    records = parse_records(text)
    if len(records) != 16:
        raise ValueError(f"{name}: expected 16 checkpoints, found {len(records)}")
    final = records[-1]
    if (
        final["cycle"] != 4
        or final["max_cycle"] != 4
        or final["iteration"] != final["max_iteration"]
    ):
        raise ValueError(f"{name}: log is not a complete four-cycle run")
    refresh = parse_refresh_metrics(text, name)
    peak = max(records, key=lambda row: row["accuracy"])
    cycle4 = [row for row in records if row["cycle"] == 4]
    cycle4_peak = max(cycle4, key=lambda row: row["accuracy"])
    result = {
        "name": name,
        "log": str(path),
        "final": float(final["accuracy"]),
        "oracle_peak": float(peak["accuracy"]),
        "oracle_peak_cycle": int(peak["cycle"]),
        "oracle_peak_iteration": int(peak["iteration"]),
        "cycle4_peak": float(cycle4_peak["accuracy"]),
        "cycle4_peak_iteration": int(cycle4_peak["iteration"]),
        "cycle4_peak_to_final": (
            float(final["accuracy"]) - float(cycle4_peak["accuracy"])
        ),
        "class_accuracy": [float(value) for value in final["class_accuracy"]],
        "refresh": refresh,
    }
    if dual_calib_mode is not None:
        config = parse_config(text, name)
        validate_dual_config(config, name, dual_calib_mode)
        result["config"] = config
        result["memory_dynamics"] = parse_memory_dynamics(text, name)
    return result


def compact_run(run: dict, duet: dict) -> dict:
    result = {
        "final": round4(run["final"]),
        "delta_final_vs_duet": round4(run["final"] - duet["final"]),
        "oracle_peak": round4(run["oracle_peak"]),
        "oracle_peak_cycle": run["oracle_peak_cycle"],
        "oracle_peak_iteration": run["oracle_peak_iteration"],
        "cycle4_peak": round4(run["cycle4_peak"]),
        "cycle4_peak_iteration": run["cycle4_peak_iteration"],
        "cycle4_peak_to_final": round4(run["cycle4_peak_to_final"]),
        "coverage": round4(run["refresh"]["coverage"]),
        "pseudo_label_precision": round4(run["refresh"]["pseudo_label_precision"]),
        "mix_accuracy": round4(run["refresh"]["mix_accuracy"]),
        "log": run["log"],
    }
    if "config" in run:
        result["config"] = run["config"]
        result["memory_dynamics"] = [
            {
                **row,
                "effective_weight": round4(row["effective_weight"]),
                "effective_coverage": round4(
                    100.0 * row["effective_weight"] / EXPECTED_FULL_VISDA_SAMPLES
                ),
                "pending_mean_weight": round4(row["pending_mean_weight"]),
            }
            for row in run["memory_dynamics"]
        ]
    return result


def dual_tier_summary(
    runs: dict[str, dict],
    *,
    min_memory_gain: float,
    min_duet_gain: float,
) -> dict:
    duet = runs["duet"]
    effects = {
        "both_prior_dual_minus_stable": (
            runs["both_prior_dual_tier"]["final"] - runs["both_prior_stable"]["final"]
        ),
        "both_prior_dual_minus_monotonic": (
            runs["both_prior_dual_tier"]["final"]
            - runs["both_prior_monotonic"]["final"]
        ),
        "none_dual_minus_stable": (
            runs["none_dual_tier"]["final"] - runs["none_stable"]["final"]
        ),
        "none_dual_minus_monotonic": (
            runs["none_dual_tier"]["final"] - runs["none_monotonic"]["final"]
        ),
    }
    effects = {key: round4(value) for key, value in effects.items()}
    dual_names = ["both_prior_dual_tier", "none_dual_tier"]
    best_name = max(dual_names, key=lambda name: runs[name]["final"])
    best = runs[best_name]
    stable_gains = [
        effects["both_prior_dual_minus_stable"],
        effects["none_dual_minus_stable"],
    ]
    monotonic_gains = [
        effects["both_prior_dual_minus_monotonic"],
        effects["none_dual_minus_monotonic"],
    ]

    beats_duet = best["final"] - duet["final"] >= min_duet_gain - 1e-9
    rescues_pending_consistently = all(
        gain >= min_memory_gain - 1e-9 for gain in stable_gains
    )
    beats_binary_extremes_consistently = rescues_pending_consistently and all(
        gain >= min_memory_gain - 1e-9 for gain in monotonic_gains
    )
    if beats_duet:
        decision = "dual_tier_beats_duet_advance"
        next_step = (
            "freeze the winning prior setting; run the 8-cycle seed-2020 "
            "confirmation, then repeat the frozen Office-Home positive control"
        )
    elif beats_binary_extremes_consistently:
        decision = "dual_tier_supported_but_duet_not_beaten"
        next_step = (
            "the three-state memory is supported, but not sufficient; keep "
            "it fixed and test only a label-free late-cycle regularizer"
        )
    elif rescues_pending_consistently:
        decision = "pending_rescue_supported_conflict_rule_not_yet_supported"
        next_step = (
            "Pending weak CE is useful, but the comparison with monotonic "
            "does not support the full conflict-exclusion claim"
        )
    elif max(stable_gains) >= min_memory_gain - 1e-9:
        decision = "dual_tier_interacts_with_prior"
        next_step = (
            "the memory gain is prior-dependent; retain only the successful "
            "calibration stratum before any further run"
        )
    else:
        decision = "dual_tier_hypothesis_rejected"
        next_step = (
            "do not tune pending weight; return to the target-head/loss "
            "timing diagnosis because restoring Pending did not recover Stage14"
        )

    return {
        "decision": decision,
        "hypothesis": (
            "temporal stability should weight supervision strength instead "
            "of acting as a binary admission gate"
        ),
        "metric": "VisDA mean per-class accuracy at cycle-4 final checkpoint",
        "selection_warning": (
            "target labels are diagnostic only; final checkpoint and fixed "
            "thresholds are used, with no oracle checkpoint selection"
        ),
        "seed": 2020,
        "adaptation_samples": EXPECTED_FULL_VISDA_SAMPLES,
        "fixed_pending_rule": "Stable=1.0, Pending=0.5*confidence, Conflict=0",
        "minimum_material_memory_gain": min_memory_gain,
        "duet_gain_required_to_advance": min_duet_gain,
        "effects_accuracy_points": effects,
        "best_dual_tier": best_name,
        "best_dual_tier_delta_vs_duet": round4(best["final"] - duet["final"]),
        "runs": {name: compact_run(run, duet) for name, run in runs.items()},
        "next": next_step,
    }


def write_per_class_csv(
    path: str,
    runs: dict[str, dict],
    class_names: list[str],
) -> None:
    rows = []
    for dual_name, stable_name, monotonic_name in (
        (
            "both_prior_dual_tier",
            "both_prior_stable",
            "both_prior_monotonic",
        ),
        ("none_dual_tier", "none_stable", "none_monotonic"),
    ):
        dual = runs[dual_name]
        for index, class_name in enumerate(class_names):
            accuracy = dual["class_accuracy"][index]
            rows.append(
                {
                    "variant": dual_name,
                    "class": class_name,
                    "accuracy": accuracy,
                    "delta_vs_duet": round4(
                        accuracy - runs["duet"]["class_accuracy"][index]
                    ),
                    "delta_vs_matched_stable": round4(
                        accuracy - runs[stable_name]["class_accuracy"][index]
                    ),
                    "delta_vs_matched_monotonic": round4(
                        accuracy - runs[monotonic_name]["class_accuracy"][index]
                    ),
                }
            )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    specs = {
        "duet": (args.duet_glob, None),
        "both_prior_stable": (args.both_prior_stable_glob, None),
        "none_stable": (args.none_stable_glob, None),
        "both_prior_monotonic": (
            args.both_prior_monotonic_glob,
            None,
        ),
        "none_monotonic": (args.none_monotonic_glob, None),
        "both_prior_dual_tier": (
            args.both_prior_dual_tier_glob,
            "both_prior",
        ),
        "none_dual_tier": (args.none_dual_tier_glob, "none"),
    }
    runs = {
        name: load_run(pattern, name, dual_calib_mode=calib_mode)
        for name, (pattern, calib_mode) in specs.items()
    }
    class_names = load_class_names(args.class_names)
    if any(len(run["class_accuracy"]) != len(class_names) for run in runs.values()):
        raise ValueError("class-name count does not match per-class accuracy")

    payload = dual_tier_summary(
        runs,
        min_memory_gain=args.min_memory_gain,
        min_duet_gain=args.min_duet_gain,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_per_class_csv(args.csv_out, runs, class_names)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
