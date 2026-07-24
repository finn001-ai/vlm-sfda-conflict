#!/usr/bin/env python
"""Summarize and gate reciprocal-boundary preflight experiments."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

try:
    from tools.summarize_visda_temporal_precision_head import parse_records
except ModuleNotFoundError:
    from summarize_visda_temporal_precision_head import parse_records


HARD_CLASS_INDICES = (3, 7, 11)
OFFICE_TASKS = ("AC", "PC", "RC")
ALL_OFFICE_TASKS = (
    "AC",
    "AP",
    "AR",
    "CA",
    "CP",
    "CR",
    "PA",
    "PC",
    "PR",
    "RA",
    "RC",
    "RP",
)
OFFICE_RECORD_PATTERN = re.compile(
    r"(?<!Trajectory Ensemble )Task:\s*([A-Z]{2}),\s*"
    r"Iter:(\d+)/(\d+);\s*Cycle:\s*(\d+)/(\d+);\s*"
    r"Accuracy\s*=\s*([0-9.]+)%"
)
STATE_PATTERN = re.compile(
    r"DCCL reciprocal boundary state:\s*cycle=(\d+);\s*"
    r"conflicts=(\d+);\s*stable_anchors=(\d+);\s*"
    r"eligible_pairs=(\d+);\s*active_pairs=(\d+);\s*"
    r"active_conflicts=(\d+);\s*frozen=(True|False)"
)
CHECKPOINT_PATTERN = re.compile(
    r"boundary_active_pairs=(\d+);\s*"
    r"boundary_active_conflicts=(\d+);\s*"
    r"boundary_coefficient_norm=([0-9.eE+-]+);\s*"
    r"boundary_frozen=(True|False)"
)
ACTION_PATTERN = re.compile(
    r"DCCL reciprocal boundary action:\s*cycle=(\d+);\s*"
    r"changed_top1=(\d+);\s*mean_probability_l1=([0-9.eE+-]+);\s*"
    r"max_probability_l1=([0-9.eE+-]+)"
)
CONFIG_PATTERN = {
    "enabled": re.compile(
        r"^\s+RECIPROCAL_BOUNDARY:\s*(True|False)\s*$", re.MULTILINE
    ),
    "target_head": re.compile(
        r"^\s+TARGET_HEAD_ADAPT:\s*(True|False)\s*$", re.MULTILINE
    ),
    "pair_feature": re.compile(
        r"^\s+PAIR_FEATURE_ADAPT:\s*(True|False)\s*$", re.MULTILINE
    ),
    "cov_transport": re.compile(
        r"^\s+COV_TRANSPORT_ADAPT:\s*(True|False)\s*$", re.MULTILINE
    ),
    "graph_teacher": re.compile(
        r"^\s+GRAPH_TEACHER_FUSION:\s*(True|False)\s*$", re.MULTILINE
    ),
    "calib_mode": re.compile(
        r"^\s+CALIB_MODE:\s*(\S+)\s*$", re.MULTILINE
    ),
    "pl_memory": re.compile(
        r"^\s+PL_MEMORY:\s*(\S+)\s*$", re.MULTILINE
    ),
    "cand_par": re.compile(
        r"^\s+CAND_PAR:\s*([0-9.eE+-]+)\s*$", re.MULTILINE
    ),
    "gtr_par": re.compile(
        r"^\s+GTR_PAR:\s*([0-9.eE+-]+)\s*$", re.MULTILINE
    ),
    "margin_par": re.compile(
        r"^\s+BOUNDARY_MARGIN_PAR:\s*([0-9.eE+-]+)\s*$", re.MULTILINE
    ),
    "consistency_par": re.compile(
        r"^\s+BOUNDARY_CONSISTENCY_PAR:\s*([0-9.eE+-]+)\s*$",
        re.MULTILINE,
    ),
    "keep_par": re.compile(
        r"^\s+BOUNDARY_KEEP_PAR:\s*([0-9.eE+-]+)\s*$", re.MULTILINE
    ),
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def one_log(pattern: str) -> Path:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one log for {pattern!r}, found {len(paths)}")
    return paths[0]


def validate_visda_records(records: list[dict], name: str) -> None:
    if len(records) != 16:
        raise ValueError(f"{name} must contain 16 checkpoints, found {len(records)}")
    final = records[-1]
    if (
        final["cycle"] != 4
        or final["max_cycle"] != 4
        or final["iteration"] != final["max_iteration"]
    ):
        raise ValueError(f"{name} is not a complete four-cycle proxy run")


def parse_office_records(text: str) -> list[dict]:
    records = []
    for task, iteration, max_iteration, cycle, max_cycle, accuracy in (
        OFFICE_RECORD_PATTERN.findall(text)
    ):
        records.append(
            {
                "task": task,
                "iteration": int(iteration),
                "max_iteration": int(max_iteration),
                "cycle": int(cycle),
                "max_cycle": int(max_cycle),
                "accuracy": float(accuracy),
            }
        )
    if not records:
        raise ValueError("No Office-Home accuracy records found")
    return records


def validate_office_records(records: list[dict], name: str) -> None:
    if len(records) != 16:
        raise ValueError(f"{name} must contain 16 checkpoints, found {len(records)}")
    if len({record["task"] for record in records}) != 1:
        raise ValueError(f"{name} contains more than one task")
    final = records[-1]
    if (
        final["cycle"] != 4
        or final["max_cycle"] != 4
        or final["iteration"] != final["max_iteration"]
    ):
        raise ValueError(f"{name} is not a complete four-cycle run")


def _config_value(text: str, name: str) -> str:
    match = CONFIG_PATTERN[name].search(text)
    if match is None:
        raise ValueError(f"Training log does not contain {name}")
    return match.group(1)


def validate_candidate_config(
    text: str,
    *,
    consistency_par: float,
    keep_par: float,
) -> bool:
    expected_false = ("target_head", "pair_feature", "cov_transport", "graph_teacher")
    return (
        _config_value(text, "enabled") == "True"
        and all(_config_value(text, name) == "False" for name in expected_false)
        and _config_value(text, "calib_mode") == "none"
        and _config_value(text, "pl_memory") == "monotonic"
        and abs(float(_config_value(text, "cand_par"))) < 1e-12
        and abs(float(_config_value(text, "gtr_par"))) < 1e-12
        and abs(float(_config_value(text, "margin_par")) - 0.1) < 1e-12
        and abs(
            float(_config_value(text, "consistency_par")) - consistency_par
        )
        < 1e-12
        and abs(float(_config_value(text, "keep_par")) - keep_par) < 1e-12
    )


def validate_host_config(text: str) -> bool:
    expected_false = (
        "enabled",
        "target_head",
        "pair_feature",
        "cov_transport",
        "graph_teacher",
    )
    return (
        all(_config_value(text, name) == "False" for name in expected_false)
        and _config_value(text, "calib_mode") == "none"
        and _config_value(text, "pl_memory") == "monotonic"
        and abs(float(_config_value(text, "cand_par"))) < 1e-12
        and abs(float(_config_value(text, "gtr_par"))) < 1e-12
    )


def _loss_raw_values(text: str, name: str) -> list[float]:
    return [
        float(value)
        for value in re.findall(rf"{re.escape(name)}_raw=([0-9.eE+-]+)", text)
    ]


def mechanism_summary(
    text: str,
    *,
    require_consistency: bool,
    require_keep: bool,
    expected_refreshes: int = 4,
) -> dict:
    states = [
        {
            "cycle": int(cycle),
            "conflicts": int(conflicts),
            "stable_anchors": int(stable_anchors),
            "eligible_pairs": int(eligible_pairs),
            "active_pairs": int(active_pairs),
            "active_conflicts": int(active_conflicts),
            "frozen": frozen == "True",
        }
        for (
            cycle,
            conflicts,
            stable_anchors,
            eligible_pairs,
            active_pairs,
            active_conflicts,
            frozen,
        ) in STATE_PATTERN.findall(text)
    ]
    checkpoints = [
        {
            "active_pairs": int(active_pairs),
            "active_conflicts": int(active_conflicts),
            "coefficient_norm": float(coefficient_norm),
            "frozen": frozen == "True",
        }
        for active_pairs, active_conflicts, coefficient_norm, frozen in (
            CHECKPOINT_PATTERN.findall(text)
        )
    ]
    actions = [
        {
            "cycle": int(cycle),
            "changed_top1": int(changed_top1),
            "mean_probability_l1": float(mean_probability_l1),
            "max_probability_l1": float(max_probability_l1),
        }
        for cycle, changed_top1, mean_probability_l1, max_probability_l1 in (
            ACTION_PATTERN.findall(text)
        )
    ]
    margin_values = _loss_raw_values(text, "boundary_margin")
    consistency_values = _loss_raw_values(text, "boundary_consistency")
    keep_values = _loss_raw_values(text, "boundary_keep")
    checks = {
        "expected_state_refreshes": len(states) == expected_refreshes,
        "pairs_frozen": any(state["frozen"] for state in states),
        "active_pairs": max(
            (state["active_pairs"] for state in states), default=0
        )
        > 0,
        "active_conflicts": max(
            (state["active_conflicts"] for state in states), default=0
        )
        > 0,
        "head_updated": max(
            (record["coefficient_norm"] for record in checkpoints), default=0.0
        )
        > 0.0,
        "boundary_effect_nonzero": max(
            (action["mean_probability_l1"] for action in actions), default=0.0
        )
        > 0.0,
        "margin_loss_active": max(margin_values, default=0.0) > 0.0,
        "consistency_loss_active": (
            max(consistency_values, default=0.0) > 0.0
            if require_consistency
            else True
        ),
        "keep_loss_active": (
            max(keep_values, default=0.0) > 0.0 if require_keep else True
        ),
    }
    return {
        "checks": checks,
        "valid": all(checks.values()),
        "max_active_pairs": max(
            (state["active_pairs"] for state in states), default=0
        ),
        "max_active_conflicts": max(
            (state["active_conflicts"] for state in states), default=0
        ),
        "final_coefficient_norm": (
            checkpoints[-1]["coefficient_norm"] if checkpoints else 0.0
        ),
        "max_changed_top1": max(
            (action["changed_top1"] for action in actions), default=0
        ),
        "max_mean_probability_l1": max(
            (action["mean_probability_l1"] for action in actions), default=0.0
        ),
        "states": states,
        "actions": actions,
    }


def summarize_visda(
    control_text: str,
    host_text: str,
    variants: dict[str, str],
    *,
    max_host_gap: float,
    min_final_improvement: float,
    min_hard_improvement: float,
    max_other_regression: float,
    min_hard_class_delta: float,
) -> dict:
    control_records = parse_records(control_text)
    host_records = parse_records(host_text)
    validate_visda_records(control_records, "official_duet_control")
    validate_visda_records(host_records, "boundary_disabled_host_control")
    control_final = control_records[-1]
    host_final = host_records[-1]
    hard_indices = HARD_CLASS_INDICES
    other_indices = tuple(
        index
        for index in range(len(control_final["class_accuracy"]))
        if index not in hard_indices
    )
    expected = {
        "margin_only": (0.0, 0.0),
        "margin_consistency": (0.05, 0.0),
        "full": (0.05, 0.05),
    }
    variant_summaries = {}
    for name, text in variants.items():
        records = parse_records(text)
        validate_visda_records(records, name)
        final = records[-1]
        consistency_par, keep_par = expected[name]
        mechanism = mechanism_summary(
            text,
            require_consistency=consistency_par > 0,
            require_keep=keep_par > 0,
        )
        classes = final["class_accuracy"]
        hard_deltas = [
            classes[index] - control_final["class_accuracy"][index]
            for index in hard_indices
        ]
        variant_summaries[name] = {
            "final": final["accuracy"],
            "oracle_peak": max(record["accuracy"] for record in records),
            "delta_final_vs_duet": round(
                final["accuracy"] - control_final["accuracy"], 4
            ),
            "hard_mean": round(mean([classes[index] for index in hard_indices]), 4),
            "delta_hard_mean_vs_duet": round(
                mean([classes[index] for index in hard_indices])
                - mean(
                    [
                        control_final["class_accuracy"][index]
                        for index in hard_indices
                    ]
                ),
                4,
            ),
            "other9_mean": round(
                mean([classes[index] for index in other_indices]), 4
            ),
            "delta_other9_mean_vs_duet": round(
                mean([classes[index] for index in other_indices])
                - mean(
                    [
                        control_final["class_accuracy"][index]
                        for index in other_indices
                    ]
                ),
                4,
            ),
            "hard_class_deltas_vs_duet": {
                "car": round(hard_deltas[0], 4),
                "person": round(hard_deltas[1], 4),
                "truck": round(hard_deltas[2], 4),
            },
            "class_accuracy": classes,
            "config_valid": validate_candidate_config(
                text,
                consistency_par=consistency_par,
                keep_par=keep_par,
            ),
            "mechanism": mechanism,
        }

    full = variant_summaries["full"]
    full_checks = {
        "host_config_valid": validate_host_config(host_text),
        "host_matches_official_duet": (
            abs(host_final["accuracy"] - control_final["accuracy"])
            <= max_host_gap + 1e-9
        ),
        "config_valid": full["config_valid"],
        "mechanism_valid": full["mechanism"]["valid"],
        "final_improvement": (
            full["delta_final_vs_duet"] >= min_final_improvement - 1e-9
        ),
        "hard_mean_improvement": (
            full["delta_hard_mean_vs_duet"] >= min_hard_improvement - 1e-9
        ),
        "other9_noninferior": (
            full["delta_other9_mean_vs_duet"] >= -max_other_regression - 1e-9
        ),
        "no_hard_class_regression": (
            min(full["hard_class_deltas_vs_duet"].values())
            >= min_hard_class_delta - 1e-9
        ),
    }
    passed = all(full_checks.values())
    return {
        "decision": "pass_visda_proxy_gate" if passed else "fail_visda_proxy_gate",
        "selection": "final checkpoint only; oracle peak is diagnostic",
        "control": {
            "final": control_final["accuracy"],
            "hard_mean": round(
                mean(
                    [
                        control_final["class_accuracy"][index]
                        for index in hard_indices
                    ]
                ),
                4,
            ),
            "other9_mean": round(
                mean(
                    [
                        control_final["class_accuracy"][index]
                        for index in other_indices
                    ]
                ),
                4,
            ),
            "class_accuracy": control_final["class_accuracy"],
        },
        "boundary_disabled_host": {
            "final": host_final["accuracy"],
            "delta_final_vs_duet": round(
                host_final["accuracy"] - control_final["accuracy"], 4
            ),
            "hard_mean": round(
                mean(
                    [
                        host_final["class_accuracy"][index]
                        for index in hard_indices
                    ]
                ),
                4,
            ),
            "delta_hard_mean_vs_duet": round(
                mean(
                    [
                        host_final["class_accuracy"][index]
                        for index in hard_indices
                    ]
                )
                - mean(
                    [
                        control_final["class_accuracy"][index]
                        for index in hard_indices
                    ]
                ),
                4,
            ),
            "other9_mean": round(
                mean(
                    [
                        host_final["class_accuracy"][index]
                        for index in other_indices
                    ]
                ),
                4,
            ),
            "delta_other9_mean_vs_duet": round(
                mean(
                    [
                        host_final["class_accuracy"][index]
                        for index in other_indices
                    ]
                )
                - mean(
                    [
                        control_final["class_accuracy"][index]
                        for index in other_indices
                    ]
                ),
                4,
            ),
            "class_accuracy": host_final["class_accuracy"],
        },
        "gate_thresholds": {
            "max_host_gap": max_host_gap,
            "min_final_improvement": min_final_improvement,
            "min_hard_improvement": min_hard_improvement,
            "max_other_regression": max_other_regression,
            "min_hard_class_delta": min_hard_class_delta,
        },
        "checks": full_checks,
        "variants": variant_summaries,
    }


def load_office_logs(
    pattern: str,
    candidate: bool,
    expected_tasks: tuple[str, ...] = OFFICE_TASKS,
) -> dict[str, tuple[Path, str]]:
    result = {}
    for path_string in sorted(glob.glob(pattern)):
        path = Path(path_string)
        text = path.read_text(errors="ignore")
        records = parse_office_records(text)
        validate_office_records(records, path.name)
        task = records[-1]["task"]
        if task in result:
            raise ValueError(f"Duplicate Office-Home task {task} for {pattern!r}")
        if candidate and task not in expected_tasks:
            raise ValueError(f"Unexpected preflight task {task}")
        result[task] = (path, text)
    if set(result) != set(expected_tasks):
        raise ValueError(
            f"Expected Office-Home tasks {expected_tasks}, found {sorted(result)}"
        )
    return result


def summarize_office(
    controls: dict[str, tuple[Path, str]],
    hosts: dict[str, tuple[Path, str]] | None,
    candidates: dict[str, tuple[Path, str]],
    *,
    min_mean_improvement: float,
    min_task_delta: float,
    min_task_wins: int,
    max_host_gap: float = 0.15,
    expected_tasks: tuple[str, ...] = OFFICE_TASKS,
    decision_prefix: str = "office_home_preflight",
) -> dict:
    tasks = []
    for task in expected_tasks:
        control_text = controls[task][1]
        host_text = hosts[task][1] if hosts is not None else control_text
        candidate_text = candidates[task][1]
        control_records = parse_office_records(control_text)
        host_records = parse_office_records(host_text)
        candidate_records = parse_office_records(candidate_text)
        validate_office_records(control_records, f"{task} control")
        validate_office_records(host_records, f"{task} host")
        validate_office_records(candidate_records, f"{task} candidate")
        control_final = control_records[-1]["accuracy"]
        host_final = host_records[-1]["accuracy"]
        candidate_final = candidate_records[-1]["accuracy"]
        mechanism = mechanism_summary(
            candidate_text,
            require_consistency=True,
            require_keep=True,
        )
        tasks.append(
            {
                "task": task,
                "control_final": control_final,
                "host_final": host_final,
                "host_delta_vs_duet": round(host_final - control_final, 4),
                "host_config_valid": (
                    validate_host_config(host_text)
                    if hosts is not None
                    else True
                ),
                "candidate_final": candidate_final,
                "delta_vs_duet": round(candidate_final - control_final, 4),
                "config_valid": validate_candidate_config(
                    candidate_text,
                    consistency_par=0.05,
                    keep_par=0.05,
                ),
                "mechanism": mechanism,
            }
        )
    control_mean = mean([task["control_final"] for task in tasks])
    candidate_mean = mean([task["candidate_final"] for task in tasks])
    checks = {
        "host_matches_official_duet": all(
            abs(task["host_delta_vs_duet"]) <= max_host_gap + 1e-9
            for task in tasks
        ),
        "all_host_configs_valid": all(
            task["host_config_valid"] for task in tasks
        ),
        "all_configs_valid": all(task["config_valid"] for task in tasks),
        "all_mechanisms_valid": all(task["mechanism"]["valid"] for task in tasks),
        "mean_improvement": (
            candidate_mean - control_mean >= min_mean_improvement - 1e-9
        ),
        "no_task_collapse": (
            min(task["delta_vs_duet"] for task in tasks)
            >= min_task_delta - 1e-9
        ),
        "task_wins": (
            sum(task["delta_vs_duet"] >= 0.0 for task in tasks) >= min_task_wins
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            f"pass_{decision_prefix}_gate"
            if passed
            else f"fail_{decision_prefix}_gate"
        ),
        "selection": "final checkpoint only",
        "control_mean": round(control_mean, 4),
        "candidate_mean": round(candidate_mean, 4),
        "delta_mean_vs_duet": round(candidate_mean - control_mean, 4),
        "gate_thresholds": {
            "min_mean_improvement": min_mean_improvement,
            "min_task_delta": min_task_delta,
            "min_task_wins": min_task_wins,
            "max_host_gap": max_host_gap,
        },
        "checks": checks,
        "tasks": tasks,
    }


def summarize_visda_full(
    control_text: str,
    candidate_text: str,
    *,
    min_final_improvement: float,
    min_hard_improvement: float,
    max_other_regression: float,
    min_hard_class_delta: float,
) -> dict:
    control_records = parse_records(control_text)
    candidate_records = parse_records(candidate_text)
    for name, records in (
        ("official_duet_full_control", control_records),
        ("reciprocal_boundary_full", candidate_records),
    ):
        if len(records) != 32:
            raise ValueError(f"{name} must contain 32 checkpoints")
        final = records[-1]
        if (
            final["cycle"] != 8
            or final["max_cycle"] != 8
            or final["iteration"] != final["max_iteration"]
        ):
            raise ValueError(f"{name} is not a complete eight-cycle run")
    control = control_records[-1]
    candidate = candidate_records[-1]
    hard_indices = HARD_CLASS_INDICES
    other_indices = tuple(
        index
        for index in range(len(control["class_accuracy"]))
        if index not in hard_indices
    )
    hard_deltas = [
        candidate["class_accuracy"][index] - control["class_accuracy"][index]
        for index in hard_indices
    ]
    delta_final = candidate["accuracy"] - control["accuracy"]
    delta_hard = mean(
        [candidate["class_accuracy"][index] for index in hard_indices]
    ) - mean([control["class_accuracy"][index] for index in hard_indices])
    delta_other = mean(
        [candidate["class_accuracy"][index] for index in other_indices]
    ) - mean([control["class_accuracy"][index] for index in other_indices])
    mechanism = mechanism_summary(
        candidate_text,
        require_consistency=True,
        require_keep=True,
        expected_refreshes=8,
    )
    checks = {
        "config_valid": validate_candidate_config(
            candidate_text,
            consistency_par=0.05,
            keep_par=0.05,
        ),
        "mechanism_valid": mechanism["valid"],
        "final_improvement": delta_final >= min_final_improvement - 1e-9,
        "hard_mean_improvement": delta_hard >= min_hard_improvement - 1e-9,
        "other9_noninferior": delta_other >= -max_other_regression - 1e-9,
        "no_hard_class_regression": (
            min(hard_deltas) >= min_hard_class_delta - 1e-9
        ),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "pass_visda_full_seed2020_gate"
            if passed
            else "fail_visda_full_seed2020_gate"
        ),
        "selection": "final checkpoint only",
        "control_final": control["accuracy"],
        "candidate_final": candidate["accuracy"],
        "delta_final_vs_duet": round(delta_final, 4),
        "delta_hard_mean_vs_duet": round(delta_hard, 4),
        "delta_other9_mean_vs_duet": round(delta_other, 4),
        "hard_class_deltas_vs_duet": {
            "car": round(hard_deltas[0], 4),
            "person": round(hard_deltas[1], 4),
            "truck": round(hard_deltas[2], 4),
        },
        "checks": checks,
        "mechanism": mechanism,
        "control_class_accuracy": control["class_accuracy"],
        "candidate_class_accuracy": candidate["class_accuracy"],
    }


def write_json(path: str, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


def write_visda_csv(path: str, summary: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = (
        "control",
        "boundary_disabled_host",
        "margin_only",
        "margin_consistency",
        "full",
    )
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "final",
                "delta_final_vs_duet",
                "hard_mean",
                "delta_hard_mean_vs_duet",
                "other9_mean",
                "delta_other9_mean_vs_duet",
            ]
        )
        for name in names:
            if name == "control":
                row = summary["control"]
            elif name == "boundary_disabled_host":
                row = summary["boundary_disabled_host"]
            else:
                row = summary["variants"][name]
            writer.writerow(
                [
                    name,
                    row["final"],
                    row.get("delta_final_vs_duet", 0.0),
                    row["hard_mean"],
                    row.get("delta_hard_mean_vs_duet", 0.0),
                    row["other9_mean"],
                    row.get("delta_other9_mean_vs_duet", 0.0),
                ]
            )


def write_office_csv(path: str, summary: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task",
                "control_final",
                "host_final",
                "host_delta_vs_duet",
                "candidate_final",
                "delta_vs_duet",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(summary["tasks"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    visda = subparsers.add_parser("visda")
    visda.add_argument("--control-glob", required=True)
    visda.add_argument("--host-glob", required=True)
    visda.add_argument("--margin-glob", required=True)
    visda.add_argument("--consistency-glob", required=True)
    visda.add_argument("--full-glob", required=True)
    visda.add_argument("--out", required=True)
    visda.add_argument("--csv-out", required=True)
    visda.add_argument("--max-host-gap", type=float, default=0.1)
    visda.add_argument("--min-final-improvement", type=float, default=0.2)
    visda.add_argument("--min-hard-improvement", type=float, default=0.2)
    visda.add_argument("--max-other-regression", type=float, default=0.1)
    visda.add_argument("--min-hard-class-delta", type=float, default=0.0)

    office = subparsers.add_parser("office-home")
    office.add_argument("--control-glob", required=True)
    office.add_argument("--host-glob", required=True)
    office.add_argument("--candidate-glob", required=True)
    office.add_argument("--out", required=True)
    office.add_argument("--csv-out", required=True)
    office.add_argument("--min-mean-improvement", type=float, default=0.2)
    office.add_argument("--min-task-delta", type=float, default=-0.3)
    office.add_argument("--min-task-wins", type=int, default=2)
    office.add_argument("--max-host-gap", type=float, default=0.15)

    visda_full = subparsers.add_parser("visda-full")
    visda_full.add_argument("--control-glob", required=True)
    visda_full.add_argument("--candidate-glob", required=True)
    visda_full.add_argument("--out", required=True)
    visda_full.add_argument("--min-final-improvement", type=float, default=0.2)
    visda_full.add_argument("--min-hard-improvement", type=float, default=0.2)
    visda_full.add_argument("--max-other-regression", type=float, default=0.1)
    visda_full.add_argument("--min-hard-class-delta", type=float, default=0.0)

    office_full = subparsers.add_parser("office-home-full")
    office_full.add_argument("--control-glob", required=True)
    office_full.add_argument("--candidate-glob", required=True)
    office_full.add_argument("--out", required=True)
    office_full.add_argument("--csv-out", required=True)
    office_full.add_argument("--min-mean-improvement", type=float, default=0.2)
    office_full.add_argument("--min-task-delta", type=float, default=-0.5)
    office_full.add_argument("--min-task-wins", type=int, default=7)

    joint = subparsers.add_parser("joint")
    joint.add_argument("--visda", required=True)
    joint.add_argument("--office-home", required=True)
    joint.add_argument("--out", required=True)
    joint_full = subparsers.add_parser("joint-full")
    joint_full.add_argument("--visda", required=True)
    joint_full.add_argument("--office-home", required=True)
    joint_full.add_argument("--out", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "visda":
        summary = summarize_visda(
            one_log(args.control_glob).read_text(errors="ignore"),
            one_log(args.host_glob).read_text(errors="ignore"),
            {
                "margin_only": one_log(args.margin_glob).read_text(errors="ignore"),
                "margin_consistency": one_log(
                    args.consistency_glob
                ).read_text(errors="ignore"),
                "full": one_log(args.full_glob).read_text(errors="ignore"),
            },
            max_host_gap=args.max_host_gap,
            min_final_improvement=args.min_final_improvement,
            min_hard_improvement=args.min_hard_improvement,
            max_other_regression=args.max_other_regression,
            min_hard_class_delta=args.min_hard_class_delta,
        )
        write_json(args.out, summary)
        write_visda_csv(args.csv_out, summary)
    elif args.command in {"office-home", "office-home-full"}:
        full = args.command == "office-home-full"
        expected_tasks = ALL_OFFICE_TASKS if full else OFFICE_TASKS
        summary = summarize_office(
            load_office_logs(
                args.control_glob,
                candidate=False,
                expected_tasks=expected_tasks,
            ),
            (
                load_office_logs(
                    args.host_glob,
                    candidate=False,
                    expected_tasks=expected_tasks,
                )
                if not full
                else None
            ),
            load_office_logs(
                args.candidate_glob,
                candidate=True,
                expected_tasks=expected_tasks,
            ),
            min_mean_improvement=args.min_mean_improvement,
            min_task_delta=args.min_task_delta,
            min_task_wins=args.min_task_wins,
            max_host_gap=(args.max_host_gap if not full else 0.15),
            expected_tasks=expected_tasks,
            decision_prefix=(
                "office_home_full_seed2020" if full else "office_home_preflight"
            ),
        )
        write_json(args.out, summary)
        write_office_csv(args.csv_out, summary)
    elif args.command == "visda-full":
        summary = summarize_visda_full(
            one_log(args.control_glob).read_text(errors="ignore"),
            one_log(args.candidate_glob).read_text(errors="ignore"),
            min_final_improvement=args.min_final_improvement,
            min_hard_improvement=args.min_hard_improvement,
            max_other_regression=args.max_other_regression,
            min_hard_class_delta=args.min_hard_class_delta,
        )
        write_json(args.out, summary)
    elif args.command == "joint":
        visda = json.loads(Path(args.visda).read_text())
        office = json.loads(Path(args.office_home).read_text())
        checks = {
            "visda_proxy": visda["decision"] == "pass_visda_proxy_gate",
            "office_home_preflight": (
                office["decision"] == "pass_office_home_preflight_gate"
            ),
        }
        passed = all(checks.values())
        write_json(
            args.out,
            {
                "decision": (
                    "pass_reciprocal_boundary_preflight"
                    if passed
                    else "fail_reciprocal_boundary_preflight"
                ),
                "checks": checks,
                "next": (
                    "run full VisDA and complete 12-task Office-Home validation"
                    if passed
                    else "inspect the failed metric; do not launch full runs"
                ),
            },
        )
    else:
        visda = json.loads(Path(args.visda).read_text())
        office = json.loads(Path(args.office_home).read_text())
        checks = {
            "visda_full_seed2020": (
                visda["decision"] == "pass_visda_full_seed2020_gate"
            ),
            "office_home_full_seed2020": (
                office["decision"] == "pass_office_home_full_seed2020_gate"
            ),
        }
        passed = all(checks.values())
        write_json(
            args.out,
            {
                "decision": (
                    "pass_reciprocal_boundary_seed2020_gate"
                    if passed
                    else "fail_reciprocal_boundary_seed2020_gate"
                ),
                "checks": checks,
                "next": (
                    "run fixed-hyperparameter seeds 2021 and 2022"
                    if passed
                    else "do not start a seed sweep"
                ),
            },
        )


if __name__ == "__main__":
    main()
