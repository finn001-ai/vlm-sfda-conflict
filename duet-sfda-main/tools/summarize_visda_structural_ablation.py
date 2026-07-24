#!/usr/bin/env python
"""Summarize the VisDA proxy memory/head structural ablation."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

try:
    from tools.summarize_visda_temporal_precision_head import (
        DEFAULT_CLASSES,
        load_class_names,
        parse_records,
    )
except ModuleNotFoundError:
    from summarize_visda_temporal_precision_head import (
        DEFAULT_CLASSES,
        load_class_names,
        parse_records,
    )


HARD_CLASS_INDICES = (3, 7, 11)
PSEUDO_PATTERN = re.compile(
    r"Number of valid pseudo-labeled samples:\s*(\d+)/(\d+);\s*"
    r"Accuracy\s*=\s*([0-9.]+)%"
)
MIX_PATTERN = re.compile(r"all_mix_output Accuracy\s*=\s*([0-9.]+)%")
CONFIG_PATTERNS = {
    "calib_mode": re.compile(r"^\s+CALIB_MODE:\s*(\S+)\s*$", re.MULTILINE),
    "calib_power": re.compile(
        r"^\s+CALIB_POWER:\s*([0-9.eE+-]+)\s*$", re.MULTILINE
    ),
    "pl_memory": re.compile(r"^\s+PL_MEMORY:\s*(\S+)\s*$", re.MULTILINE),
    "target_head": re.compile(
        r"^\s+TARGET_HEAD_ADAPT:\s*(True|False)\s*$", re.MULTILINE
    ),
    "gtr_par": re.compile(r"^\s+GTR_PAR:\s*([0-9.eE+-]+)\s*$", re.MULTILINE),
}
EXPECTED_VARIANTS = {
    "v1_monotonic_head": {"pl_memory": "monotonic", "target_head": True},
    "v2_stable_nohead": {"pl_memory": "stable", "target_head": False},
    "v3_monotonic_nohead": {"pl_memory": "monotonic", "target_head": False},
}
ARCHIVED_V0 = {
    "variant": "v0_stable_head_archived",
    "pl_memory": "stable",
    "target_head": True,
    "final": 87.83,
    "oracle_peak": 87.83,
    "class_accuracy": [
        97.56,
        86.01,
        85.61,
        75.36,
        96.27,
        95.81,
        93.50,
        80.68,
        91.91,
        94.56,
        91.34,
        65.34,
    ],
    "selected_count": 11309,
    "total_count": 13847,
    "coverage": 81.67,
    "pseudo_label_precision": 94.16,
    "mix_accuracy": 87.60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-glob", required=True)
    parser.add_argument("--v1-glob", required=True)
    parser.add_argument("--v2-glob", required=True)
    parser.add_argument("--v3-glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--class-names")
    parser.add_argument("--min-final-improvement", type=float, default=0.15)
    parser.add_argument("--min-hard-mean-improvement", type=float, default=0.0)
    parser.add_argument("--max-other9-regression", type=float, default=0.10)
    parser.add_argument("--max-hard-class-regression", type=float, default=0.50)
    return parser.parse_args()


def mean_at(values: list[float], indices: tuple[int, ...]) -> float:
    return sum(values[index] for index in indices) / len(indices)


def validate_records(records: list[dict], name: str) -> None:
    if len(records) != 16:
        raise ValueError(f"{name} must contain 16 checkpoints, found {len(records)}")
    final = records[-1]
    if final["cycle"] != 4 or final["max_cycle"] != 4:
        raise ValueError(f"{name} did not finish a clean four-cycle budget")
    if final["iteration"] != final["max_iteration"]:
        raise ValueError(f"{name} final record is not the end of cycle 4")


def parse_refresh_metrics(text: str) -> dict:
    pseudo = PSEUDO_PATTERN.findall(text)
    mix = MIX_PATTERN.findall(text)
    if len(pseudo) != 4 or len(mix) != 4:
        raise ValueError(
            "Expected four pseudo-label and mixed-output refresh records, "
            f"found {len(pseudo)} and {len(mix)}"
        )
    selected, total, precision = pseudo[-1]
    return {
        "num_refreshes": 4,
        "selected_count": int(selected),
        "total_count": int(total),
        "coverage": 100.0 * int(selected) / int(total),
        "pseudo_label_precision": float(precision),
        "mix_accuracy": float(mix[-1]),
    }


def read_config(text: str) -> dict:
    values = {}
    for name, pattern in CONFIG_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"Training log does not contain DCCL.{name}")
        values[name] = match.group(1)
    return {
        "calib_mode": values["calib_mode"],
        "calib_power": float(values["calib_power"]),
        "pl_memory": values["pl_memory"],
        "target_head": values["target_head"] == "True",
        "gtr_par": float(values["gtr_par"]),
    }


def validate_variant_config(config: dict, variant: str) -> bool:
    expected = EXPECTED_VARIANTS[variant]
    return (
        config["calib_mode"] == "both_prior"
        and abs(config["calib_power"] - 0.5) < 1e-9
        and config["pl_memory"] == expected["pl_memory"]
        and config["target_head"] is expected["target_head"]
        and abs(config["gtr_par"]) < 1e-12
    )


def summarize_run(
    name: str,
    records: list[dict],
    refresh: dict,
    control_final: dict,
    config_valid: bool,
    *,
    min_final_improvement: float,
    min_hard_mean_improvement: float,
    max_other9_regression: float,
    max_hard_class_regression: float,
) -> dict:
    validate_records(records, name)
    final = records[-1]
    peak = max(records, key=lambda row: row["accuracy"])
    classes = final["class_accuracy"]
    control_classes = control_final["class_accuracy"]
    other_indices = tuple(
        index for index in range(len(classes)) if index not in HARD_CLASS_INDICES
    )
    hard_mean = mean_at(classes, HARD_CLASS_INDICES)
    control_hard_mean = mean_at(control_classes, HARD_CLASS_INDICES)
    other9_mean = mean_at(classes, other_indices)
    control_other9_mean = mean_at(control_classes, other_indices)
    hard_class_deltas = [
        classes[index] - control_classes[index] for index in HARD_CLASS_INDICES
    ]
    checks = {
        "config_valid": config_valid,
        "final_improvement": (
            final["accuracy"] - control_final["accuracy"]
            >= min_final_improvement - 1e-9
        ),
        "hard_mean_noninferior": (
            hard_mean - control_hard_mean
            >= min_hard_mean_improvement - 1e-9
        ),
        "other9_noninferior": (
            other9_mean - control_other9_mean
            >= -max_other9_regression - 1e-9
        ),
        "no_hard_class_compensation": (
            min(hard_class_deltas) >= -max_hard_class_regression - 1e-9
        ),
    }
    return {
        "variant": name,
        "final": round(final["accuracy"], 4),
        "oracle_peak": round(peak["accuracy"], 4),
        "oracle_peak_cycle": peak["cycle"],
        "oracle_peak_iteration": peak["iteration"],
        "delta_final_vs_duet": round(
            final["accuracy"] - control_final["accuracy"], 4
        ),
        "hard_mean": round(hard_mean, 4),
        "delta_hard_mean_vs_duet": round(hard_mean - control_hard_mean, 4),
        "other9_mean": round(other9_mean, 4),
        "delta_other9_mean_vs_duet": round(other9_mean - control_other9_mean, 4),
        "hard_class_deltas_vs_duet": {
            "car": round(hard_class_deltas[0], 4),
            "person": round(hard_class_deltas[1], 4),
            "truck": round(hard_class_deltas[2], 4),
        },
        "selected_count": refresh["selected_count"],
        "total_count": refresh["total_count"],
        "coverage": round(refresh["coverage"], 4),
        "pseudo_label_precision": refresh["pseudo_label_precision"],
        "mix_accuracy": refresh["mix_accuracy"],
        "checks": checks,
        "pass_proxy_gate": all(checks.values()),
        "class_accuracy": classes,
    }


def summarize_ablation(
    control_records: list[dict],
    candidate_records: dict[str, list[dict]],
    control_refresh: dict,
    candidate_refresh: dict[str, dict],
    config_valid: dict[str, bool],
    *,
    min_final_improvement: float = 0.15,
    min_hard_mean_improvement: float = 0.0,
    max_other9_regression: float = 0.10,
    max_hard_class_regression: float = 0.50,
) -> dict:
    validate_records(control_records, "official_duet_control")
    control_final = control_records[-1]
    control_peak = max(control_records, key=lambda row: row["accuracy"])
    control_classes = control_final["class_accuracy"]
    other_indices = tuple(
        index
        for index in range(len(control_classes))
        if index not in HARD_CLASS_INDICES
    )
    control = {
        "final": round(control_final["accuracy"], 4),
        "oracle_peak": round(control_peak["accuracy"], 4),
        "hard_mean": round(mean_at(control_classes, HARD_CLASS_INDICES), 4),
        "other9_mean": round(mean_at(control_classes, other_indices), 4),
        "selected_count": control_refresh["selected_count"],
        "total_count": control_refresh["total_count"],
        "coverage": round(control_refresh["coverage"], 4),
        "pseudo_label_precision": control_refresh["pseudo_label_precision"],
        "mix_accuracy": control_refresh["mix_accuracy"],
        "class_accuracy": control_classes,
    }
    variants = {
        name: summarize_run(
            name,
            candidate_records[name],
            candidate_refresh[name],
            control_final,
            config_valid[name],
            min_final_improvement=min_final_improvement,
            min_hard_mean_improvement=min_hard_mean_improvement,
            max_other9_regression=max_other9_regression,
            max_hard_class_regression=max_hard_class_regression,
        )
        for name in EXPECTED_VARIANTS
    }
    passing = [
        result for result in variants.values() if result["pass_proxy_gate"]
    ]
    best_overall = max(variants.values(), key=lambda result: result["final"])
    winner = (
        max(passing, key=lambda result: result["final"])["variant"]
        if passing
        else None
    )
    ranking_hints = {
        "v1_monotonic_head": (
            "monotonic memory with the target head ranks first; stable-memory "
            "coverage loss is the leading suspect"
        ),
        "v2_stable_nohead": (
            "stable memory without the target head ranks first; target-head "
            "drift is the leading suspect"
        ),
        "v3_monotonic_nohead": (
            "removing both interventions ranks first; both stable memory and "
            "the target head are suspect"
        ),
    }
    v0 = dict(ARCHIVED_V0)
    v0["hard_mean"] = round(
        mean_at(v0["class_accuracy"], HARD_CLASS_INDICES), 4
    )
    v0["other9_mean"] = round(
        mean_at(
            v0["class_accuracy"],
            tuple(
                index
                for index in range(len(v0["class_accuracy"]))
                if index not in HARD_CLASS_INDICES
            ),
        ),
        4,
    )
    effects = {
        "memory_effect_with_head_v1_minus_archived_v0": round(
            variants["v1_monotonic_head"]["final"] - v0["final"], 4
        ),
        "memory_effect_without_head_v3_minus_v2": round(
            variants["v3_monotonic_nohead"]["final"]
            - variants["v2_stable_nohead"]["final"],
            4,
        ),
        "head_effect_with_monotonic_v1_minus_v3": round(
            variants["v1_monotonic_head"]["final"]
            - variants["v3_monotonic_nohead"]["final"],
            4,
        ),
        "head_effect_with_stable_archived_v0_minus_v2": round(
            v0["final"] - variants["v2_stable_nohead"]["final"], 4
        ),
    }
    decision = "pass_proxy_gate" if winner is not None else "fail_proxy_gate"
    if winner is not None:
        next_step = (
            f"run one matched full-data four-cycle preflight for {winner}; "
            "do not launch eight cycles yet"
        )
    else:
        next_step = (
            "do not run a full-data structural variant; retain the official "
            "DUET VisDA path and redesign DCCL as a baseline-preserving "
            "conflict-only intervention"
        )
    return {
        "decision": decision,
        "metric": "VisDA mean per-class accuracy",
        "selection_warning": (
            "final checkpoint is primary; oracle peaks and validation-label "
            "class gates are development diagnostics only"
        ),
        "proxy_contract": {
            "adaptation_samples": 13847,
            "evaluation_samples": 55388,
            "cycles": 4,
            "seed": 2020,
            "calib_mode": "both_prior",
            "calib_power": 0.5,
            "gtr_par": 0.0,
        },
        "thresholds": {
            "min_final_improvement": min_final_improvement,
            "min_hard_mean_improvement": min_hard_mean_improvement,
            "max_other9_regression": max_other9_regression,
            "max_hard_class_regression": max_hard_class_regression,
        },
        "official_duet_control": control,
        "archived_v0_reference": v0,
        "archived_v0_warning": (
            "V0 is the archived matched GTR=0 result and is not rerun by this job"
        ),
        "variants": variants,
        "factorial_effects": effects,
        "best_overall_variant": best_overall["variant"],
        "ranking_hint": ranking_hints[best_overall["variant"]],
        "passing_variant": winner,
        "next": next_step,
    }


def read_one(pattern: str) -> tuple[str, list[dict]]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one clean log for {pattern!r}")
    text = paths[0].read_text(errors="ignore")
    return text, parse_records(text)


def write_csv(path: Path, result: dict, class_names: list[str]) -> None:
    rows = []
    control = result["official_duet_control"]
    rows.append(
        {
            "variant": "official_duet_control",
            "pl_memory": "monotonic",
            "target_head": False,
            "final": control["final"],
            "oracle_peak": control["oracle_peak"],
            "delta_final_vs_duet": 0.0,
            "hard_mean": control["hard_mean"],
            "other9_mean": control["other9_mean"],
            "coverage": control["coverage"],
            "pseudo_label_precision": control["pseudo_label_precision"],
            "mix_accuracy": control["mix_accuracy"],
            "pass_proxy_gate": True,
        }
    )
    v0 = result["archived_v0_reference"]
    rows.append(
        {
            "variant": v0["variant"],
            "pl_memory": v0["pl_memory"],
            "target_head": v0["target_head"],
            "final": v0["final"],
            "oracle_peak": v0["oracle_peak"],
            "delta_final_vs_duet": round(v0["final"] - control["final"], 4),
            "hard_mean": v0["hard_mean"],
            "other9_mean": v0["other9_mean"],
            "coverage": v0["coverage"],
            "pseudo_label_precision": v0["pseudo_label_precision"],
            "mix_accuracy": v0["mix_accuracy"],
            "pass_proxy_gate": False,
        }
    )
    for name, variant in result["variants"].items():
        rows.append(
            {
                "variant": name,
                "pl_memory": EXPECTED_VARIANTS[name]["pl_memory"],
                "target_head": EXPECTED_VARIANTS[name]["target_head"],
                "final": variant["final"],
                "oracle_peak": variant["oracle_peak"],
                "delta_final_vs_duet": variant["delta_final_vs_duet"],
                "hard_mean": variant["hard_mean"],
                "other9_mean": variant["other9_mean"],
                "coverage": variant["coverage"],
                "pseudo_label_precision": variant["pseudo_label_precision"],
                "mix_accuracy": variant["mix_accuracy"],
                "pass_proxy_gate": variant["pass_proxy_gate"],
            }
        )
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    class_path = path.with_name(path.stem + "_per_class.csv")
    with class_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "class", "final_accuracy"])
        writer.writerows(
            (
                variant,
                class_name,
                values[index],
            )
            for variant, values in [
                ("official_duet_control", control["class_accuracy"]),
                (v0["variant"], v0["class_accuracy"]),
                *[
                    (name, item["class_accuracy"])
                    for name, item in result["variants"].items()
                ],
            ]
            for index, class_name in enumerate(class_names)
        )


def main() -> None:
    args = parse_args()
    texts = {}
    records = {}
    patterns = {
        "control": args.control_glob,
        "v1_monotonic_head": args.v1_glob,
        "v2_stable_nohead": args.v2_glob,
        "v3_monotonic_nohead": args.v3_glob,
    }
    for name, pattern in patterns.items():
        texts[name], records[name] = read_one(pattern)
    config_valid = {
        name: validate_variant_config(read_config(texts[name]), name)
        for name in EXPECTED_VARIANTS
    }
    result = summarize_ablation(
        records["control"],
        {name: records[name] for name in EXPECTED_VARIANTS},
        parse_refresh_metrics(texts["control"]),
        {
            name: parse_refresh_metrics(texts[name])
            for name in EXPECTED_VARIANTS
        },
        config_valid,
        min_final_improvement=args.min_final_improvement,
        min_hard_mean_improvement=args.min_hard_mean_improvement,
        max_other9_regression=args.max_other9_regression,
        max_hard_class_regression=args.max_hard_class_regression,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(
        Path(args.csv_out),
        result,
        load_class_names(args.class_names)
        if args.class_names
        else DEFAULT_CLASSES,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
