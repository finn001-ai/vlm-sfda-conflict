#!/usr/bin/env python
"""Summarize the full-data VisDA Stage14 prior/memory factorial audit."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

try:
    from tools.summarize_visda_temporal_precision_head import (
        load_class_names,
        parse_records,
    )
except ModuleNotFoundError:
    from summarize_visda_temporal_precision_head import (
        load_class_names,
        parse_records,
    )


PSEUDO_PATTERN = re.compile(
    r"Number of valid pseudo-labeled samples:\s*(\d+)/(\d+);\s*"
    r"Accuracy\s*=\s*([0-9.]+)%"
)
SELECTED_MIX_PATTERN = re.compile(
    r"Mixed output with valid mask:\s*([0-9.]+)%"
)
MIX_PATTERN = re.compile(r"all_mix_output Accuracy\s*=\s*([0-9.]+)%")
CONFIG_PATTERNS = {
    "calib_mode": re.compile(
        r"^\s+CALIB_MODE:\s*(\S+)\s*$", re.MULTILINE
    ),
    "calib_power": re.compile(
        r"^\s+CALIB_POWER:\s*([0-9.eE+-]+)\s*$", re.MULTILINE
    ),
    "pl_memory": re.compile(
        r"^\s+PL_MEMORY:\s*(\S+)\s*$", re.MULTILINE
    ),
    "target_head": re.compile(
        r"^\s+TARGET_HEAD_ADAPT:\s*(True|False)\s*$", re.MULTILINE
    ),
    "gtr_par": re.compile(
        r"^\s+GTR_PAR:\s*([0-9.eE+-]+)\s*$", re.MULTILINE
    ),
}
VARIANT_SPECS = {
    "both_prior_stable": {
        "calib_mode": "both_prior",
        "pl_memory": "stable",
    },
    "none_stable": {
        "calib_mode": "none",
        "pl_memory": "stable",
    },
    "both_prior_monotonic": {
        "calib_mode": "both_prior",
        "pl_memory": "monotonic",
    },
    "none_monotonic": {
        "calib_mode": "none",
        "pl_memory": "monotonic",
    },
}
EXPECTED_FULL_VISDA_SAMPLES = 55388


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reproduction", "factorial"))
    parser.add_argument("--control-glob", required=True)
    parser.add_argument("--current-glob", required=True)
    parser.add_argument("--none-stable-glob")
    parser.add_argument("--both-prior-monotonic-glob")
    parser.add_argument("--none-monotonic-glob")
    parser.add_argument("--class-names")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out")
    parser.add_argument("--max-reproduction-delta", type=float, default=-0.15)
    parser.add_argument("--min-material-effect", type=float, default=0.10)
    parser.add_argument("--min-duet-gain", type=float, default=0.10)
    return parser.parse_args()


def round4(value: float) -> float:
    return round(float(value), 4)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def read_one_log(pattern: str, name: str) -> tuple[Path, str]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if len(paths) != 1:
        raise ValueError(
            f"{name}: expected exactly one clean log, found {len(paths)}"
        )
    return paths[0], paths[0].read_text(errors="ignore")


def validate_records(records: list[dict], name: str) -> None:
    if len(records) != 16:
        raise ValueError(
            f"{name}: expected 16 checkpoints for four cycles, "
            f"found {len(records)}"
        )
    final = records[-1]
    if final["cycle"] != 4 or final["max_cycle"] != 4:
        raise ValueError(f"{name}: run is not a clean four-cycle experiment")
    if final["iteration"] != final["max_iteration"]:
        raise ValueError(f"{name}: final record is not the end of cycle four")


def parse_refresh_metrics(text: str, name: str) -> dict:
    pseudo = PSEUDO_PATTERN.findall(text)
    selected_mix = SELECTED_MIX_PATTERN.findall(text)
    mix = MIX_PATTERN.findall(text)
    if len(pseudo) != 4 or len(selected_mix) != 4 or len(mix) != 4:
        raise ValueError(
            f"{name}: expected four pseudo-label refresh records, found "
            f"{len(pseudo)}, {len(selected_mix)}, and {len(mix)}"
        )
    selected, total, source_precision = pseudo[-1]
    selected_count = int(selected)
    total_count = int(total)
    if total_count != EXPECTED_FULL_VISDA_SAMPLES:
        raise ValueError(
            f"{name}: expected full VisDA adaptation with "
            f"{EXPECTED_FULL_VISDA_SAMPLES} samples, found {total_count}"
        )
    return {
        "selected_count": selected_count,
        "total_count": total_count,
        "coverage": 100.0 * selected_count / total_count,
        "selected_source_label_precision": float(source_precision),
        "pseudo_label_precision": float(selected_mix[-1]),
        "mix_accuracy": float(mix[-1]),
    }


def read_candidate_config(text: str, name: str) -> dict:
    values = {}
    for key, pattern in CONFIG_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"{name}: training log does not contain {key}")
        values[key] = match.group(1)
    return {
        "calib_mode": values["calib_mode"],
        "calib_power": float(values["calib_power"]),
        "pl_memory": values["pl_memory"],
        "target_head": values["target_head"] == "True",
        "gtr_par": float(values["gtr_par"]),
    }


def validate_candidate_config(config: dict, name: str) -> None:
    expected = VARIANT_SPECS[name]
    checks = {
        "calib_mode": config["calib_mode"] == expected["calib_mode"],
        "calib_power": abs(config["calib_power"] - 0.5) < 1e-9,
        "pl_memory": config["pl_memory"] == expected["pl_memory"],
        "target_head": config["target_head"] is True,
        "gtr_par": abs(config["gtr_par"] - 0.05) < 1e-12,
    }
    failures = [key for key, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            f"{name}: fixed Stage14 configuration check failed for "
            + ", ".join(failures)
        )


def load_run(pattern: str, name: str, *, candidate: bool) -> dict:
    path, text = read_one_log(pattern, name)
    records = parse_records(text)
    validate_records(records, name)
    refresh = parse_refresh_metrics(text, name)
    config = None
    if candidate:
        config = read_candidate_config(text, name)
        validate_candidate_config(config, name)
    final = records[-1]
    peak = max(records, key=lambda row: row["accuracy"])
    return {
        "name": name,
        "log": str(path),
        "final": float(final["accuracy"]),
        "oracle_peak": float(peak["accuracy"]),
        "oracle_peak_cycle": int(peak["cycle"]),
        "oracle_peak_iteration": int(peak["iteration"]),
        "class_accuracy": [float(value) for value in final["class_accuracy"]],
        "refresh": refresh,
        "config": config,
    }


def compact_run(
    run: dict,
    control: dict | None = None,
    current: dict | None = None,
) -> dict:
    result = {
        "final": round4(run["final"]),
        "oracle_peak": round4(run["oracle_peak"]),
        "oracle_peak_cycle": run["oracle_peak_cycle"],
        "oracle_peak_iteration": run["oracle_peak_iteration"],
        "selected_count": run["refresh"]["selected_count"],
        "total_count": run["refresh"]["total_count"],
        "coverage": round4(run["refresh"]["coverage"]),
        "selected_source_label_precision": round4(
            run["refresh"]["selected_source_label_precision"]
        ),
        "pseudo_label_precision": round4(
            run["refresh"]["pseudo_label_precision"]
        ),
        "mix_accuracy": round4(run["refresh"]["mix_accuracy"]),
        "class_accuracy": run["class_accuracy"],
        "log": run["log"],
    }
    if run["config"] is not None:
        result["config"] = run["config"]
    if control is not None:
        result["delta_final_vs_duet"] = round4(
            run["final"] - control["final"]
        )
        result["delta_coverage_vs_duet"] = round4(
            run["refresh"]["coverage"] - control["refresh"]["coverage"]
        )
    if current is not None:
        result["delta_final_vs_current_stage14"] = round4(
            run["final"] - current["final"]
        )
        result["delta_coverage_vs_current_stage14"] = round4(
            run["refresh"]["coverage"] - current["refresh"]["coverage"]
        )
    return result


def reproduction_summary(
    control: dict,
    current: dict,
    max_reproduction_delta: float,
) -> dict:
    delta = current["final"] - control["final"]
    gap_reproduced = delta <= max_reproduction_delta + 1e-9
    return {
        "decision": (
            "gap_reproduced_run_factorial"
            if gap_reproduced
            else "gap_not_reproduced_stop"
        ),
        "metric": "VisDA mean per-class accuracy at cycle-4 final checkpoint",
        "seed": 2020,
        "adaptation_samples": EXPECTED_FULL_VISDA_SAMPLES,
        "reproduction_threshold": {
            "stage14_minus_duet_must_be_at_most": max_reproduction_delta
        },
        "gap_reproduced": gap_reproduced,
        "observed_stage14_minus_duet": round4(delta),
        "duet": compact_run(control),
        "current_stage14": compact_run(current, control),
        "next": (
            "run the remaining three prior/memory factorial arms"
            if gap_reproduced
            else (
                "stop: the failure mechanism was not reproduced under the "
                "matched full-data four-cycle setup"
            )
        ),
    }


def causal_conclusion(effects: dict, minimum: float) -> str:
    prior_consistent = (
        effects["remove_prior_with_stable"] >= minimum
        and effects["remove_prior_with_monotonic"] >= minimum
    )
    memory_consistent = (
        effects["monotonic_with_both_prior"] >= minimum
        and effects["monotonic_with_no_prior"] >= minimum
    )
    if prior_consistent and memory_consistent:
        return "both_prior_and_stable_memory_both_harm_consistently"
    if prior_consistent:
        return "both_prior_is_primary_consistent_cause"
    if memory_consistent:
        return "stable_memory_is_primary_consistent_cause"
    if (
        effects["remove_both_jointly"] >= minimum
        and effects["interaction_remove_prior_x_monotonic"] >= minimum
    ):
        return "prior_memory_interaction_is_primary_cause"
    if max(effects.values()) >= minimum:
        return "one_stratum_improves_but_cause_is_not_consistent"
    return "neither_factor_materially_recovers_stage14"


def factorial_summary(
    control: dict,
    runs: dict[str, dict],
    *,
    max_reproduction_delta: float,
    min_material_effect: float,
    min_duet_gain: float,
) -> dict:
    current = runs["both_prior_stable"]
    reproduction = reproduction_summary(
        control, current, max_reproduction_delta
    )
    if not reproduction["gap_reproduced"]:
        raise ValueError(
            "factorial summary refused because the matched Stage14 gap "
            "was not reproduced"
        )

    finals = {name: run["final"] for name, run in runs.items()}
    effects = {
        "remove_prior_with_stable": (
            finals["none_stable"] - finals["both_prior_stable"]
        ),
        "remove_prior_with_monotonic": (
            finals["none_monotonic"] - finals["both_prior_monotonic"]
        ),
        "remove_prior_average": (
            mean([finals["none_stable"], finals["none_monotonic"]])
            - mean(
                [
                    finals["both_prior_stable"],
                    finals["both_prior_monotonic"],
                ]
            )
        ),
        "monotonic_with_both_prior": (
            finals["both_prior_monotonic"] - finals["both_prior_stable"]
        ),
        "monotonic_with_no_prior": (
            finals["none_monotonic"] - finals["none_stable"]
        ),
        "monotonic_average": (
            mean(
                [
                    finals["both_prior_monotonic"],
                    finals["none_monotonic"],
                ]
            )
            - mean([finals["both_prior_stable"], finals["none_stable"]])
        ),
        "interaction_remove_prior_x_monotonic": (
            (
                finals["none_monotonic"]
                - finals["both_prior_monotonic"]
            )
            - (finals["none_stable"] - finals["both_prior_stable"])
        ),
        "remove_both_jointly": (
            finals["none_monotonic"] - finals["both_prior_stable"]
        ),
    }
    effects = {key: round4(value) for key, value in effects.items()}
    conclusion = causal_conclusion(effects, min_material_effect)

    ablation_names = [
        "none_stable",
        "both_prior_monotonic",
        "none_monotonic",
    ]
    best_ablation_name = max(
        ablation_names, key=lambda name: runs[name]["final"]
    )
    best_ablation = runs[best_ablation_name]
    best_delta_duet = best_ablation["final"] - control["final"]
    best_delta_current = best_ablation["final"] - current["final"]

    if best_delta_duet >= min_duet_gain - 1e-9:
        decision = "candidate_beats_duet_at_cycle4"
        next_step = (
            "freeze this structural choice; run its full eight-cycle seed-2020 "
            "confirmation, then recheck the frozen Office-Home positive control"
        )
    elif best_delta_current >= min_material_effect - 1e-9:
        decision = "cause_supported_but_duet_not_recovered"
        next_step = (
            "do not add a boundary module; refine only the supported factor "
            "with label-free coverage and class-marginal constraints"
        )
    else:
        decision = "factorial_does_not_recover_stage14"
        next_step = (
            "neither prior calibration nor stable memory explains enough of "
            "the gap; return to loss/head diagnostics without hard-coded classes"
        )

    compact_runs = {
        name: compact_run(run, control, current)
        for name, run in runs.items()
    }
    return {
        "decision": decision,
        "metric": "VisDA mean per-class accuracy at cycle-4 final checkpoint",
        "selection_warning": (
            "target labels are used only for this fixed causal audit; do not "
            "use per-class target accuracy for unsupervised hyperparameter tuning"
        ),
        "seed": 2020,
        "adaptation_samples": EXPECTED_FULL_VISDA_SAMPLES,
        "reproduction": reproduction,
        "causal_conclusion": conclusion,
        "minimum_material_effect": min_material_effect,
        "factor_effects_accuracy_points": effects,
        "duet": compact_run(control),
        "variants": compact_runs,
        "best_ablation": best_ablation_name,
        "best_ablation_delta_vs_duet": round4(best_delta_duet),
        "best_ablation_delta_vs_current_stage14": round4(best_delta_current),
        "cycle4_duet_gain_required_to_advance": min_duet_gain,
        "next": next_step,
    }


def write_json(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def write_factorial_csv(
    path: str,
    control: dict,
    runs: dict[str, dict],
    class_names: list[str],
) -> None:
    rows = []
    current = runs["both_prior_stable"]
    for name, run in runs.items():
        for index, class_name in enumerate(class_names):
            rows.append(
                {
                    "variant": name,
                    "calib_mode": run["config"]["calib_mode"],
                    "pl_memory": run["config"]["pl_memory"],
                    "class": class_name,
                    "accuracy": run["class_accuracy"][index],
                    "delta_vs_duet": round4(
                        run["class_accuracy"][index]
                        - control["class_accuracy"][index]
                    ),
                    "delta_vs_current_stage14": round4(
                        run["class_accuracy"][index]
                        - current["class_accuracy"][index]
                    ),
                }
            )
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    class_names = load_class_names(args.class_names)
    control = load_run(args.control_glob, "official_duet", candidate=False)
    current = load_run(
        args.current_glob, "both_prior_stable", candidate=True
    )

    if args.phase == "reproduction":
        payload = reproduction_summary(
            control, current, args.max_reproduction_delta
        )
        write_json(args.out, payload)
        print(json.dumps(payload, indent=2))
        return 0 if payload["gap_reproduced"] else 2

    required = {
        "none_stable": args.none_stable_glob,
        "both_prior_monotonic": args.both_prior_monotonic_glob,
        "none_monotonic": args.none_monotonic_glob,
    }
    missing = [name for name, pattern in required.items() if not pattern]
    if missing:
        raise ValueError(
            "factorial phase requires log globs for " + ", ".join(missing)
        )
    runs = {"both_prior_stable": current}
    runs.update(
        {
            name: load_run(pattern, name, candidate=True)
            for name, pattern in required.items()
        }
    )
    payload = factorial_summary(
        control,
        runs,
        max_reproduction_delta=args.max_reproduction_delta,
        min_material_effect=args.min_material_effect,
        min_duet_gain=args.min_duet_gain,
    )
    write_json(args.out, payload)
    if args.csv_out:
        write_factorial_csv(args.csv_out, control, runs, class_names)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
