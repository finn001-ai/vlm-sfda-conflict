#!/usr/bin/env python
"""Audit the exact logit-space KL influence added by attribute reliability.

Phase 1 reads only locked label-free probabilities plus path fields needed to
identify the exact proxy25 rows.  It writes and locks the candidate-minus-
control KL logit direction.  Phase 2 parses target labels solely for an oracle
direction diagnostic and reads the already-completed matched-proxy gate.

No image, checkpoint, model forward, backward pass, optimizer, or training is
used.  The result is a first-order logit-space diagnostic, not a network-
parameter gradient claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.attribute_kl_influence_audit import (  # noqa: E402
    evaluate_attribute_kl_influence,
    kl_logit_descent_directions,
    oracle_logit_influence,
    paired_bootstrap_mean_ci,
)
from tools.audit_visda_conflict_attribute_mass import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_TARGET_SAMPLES,
    _load_class_names,
    _load_locked_inputs,
    _parse_labels_after_lock,
    _sha256,
    _write_csv,
)


EXPECTED_PROXY_SAMPLES = 13_847
KL_WEIGHT = 0.4
BOOTSTRAP_REPEATS = 2_000
DEFAULT_SOURCE_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_pairwise_attribute_audit_seed2020/pairwise_attribute_audit"
)
DEFAULT_RELIABILITY_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_attribute_reliability_audit_seed2020/attribute_reliability_audit"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_attribute_kl_influence_audit_seed2020/"
    "attribute_kl_influence_audit"
)
DEFAULT_CANDIDATE_DIR = Path(
    "output/uda/VISDA-C/TV/duet_attribute_reliability_kl_visda_proxy25_seed2020"
)
DEFAULT_PROXY_GATE = Path(
    "output/uda/VISDA-C/"
    "duet_attribute_reliability_kl_visda_proxy25_seed2020_gate.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--reliability-dir", type=Path, default=DEFAULT_RELIABILITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-list",
        type=Path,
        default=Path("data/VISDA-C/validation_list.txt"),
    )
    parser.add_argument(
        "--proxy-list",
        type=Path,
        default=Path("data/VISDA-C/validation_proxy25_seed2020_list.txt"),
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=Path("data/VISDA-C/classname.txt"),
    )
    parser.add_argument("--candidate-log", type=Path)
    parser.add_argument("--proxy-gate", type=Path, default=DEFAULT_PROXY_GATE)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--kl-weight", type=float, default=KL_WEIGHT)
    return parser.parse_args()


def _path_fields_without_labels(path: Path) -> list[str]:
    """Read only the path field; deliberately do not parse the label field."""
    result: list[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        fields = stripped.rsplit(maxsplit=1)
        if len(fields) != 2 or not fields[0]:
            raise ValueError(f"Malformed path-bearing row {line_number} in {path}")
        result.append(fields[0])
    if not result:
        raise ValueError(f"No rows found in {path}")
    return result


def _resolve_candidate_log(candidate_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(f"Missing candidate log: {requested}")
        return requested
    candidates = sorted(candidate_dir.glob("*.txt"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one candidate log in {candidate_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _parse_candidate_contract(path: Path) -> dict[str, Any]:
    pattern = re.compile(
        r"DUET attribute reliability KL applied: cycle=1; "
        r"active_conflicts=(\d+); changed_top1=(\d+); "
        r"mean_weight=([0-9.]+); target_labels=False; fitted_thresholds=False"
    )
    matches = pattern.findall(path.read_text(errors="replace"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one cycle-1 attribute contract in {path}, found {len(matches)}"
        )
    active, changed, mean_weight = matches[0]
    return {
        "active_conflicts": int(active),
        "changed_top1": int(changed),
        "mean_attribute_weight": float(mean_weight),
    }


def _load_reliability_target(
    directory: Path, *, source_signal_sha256: str
) -> dict[str, Any]:
    stem = "visda_conflict_attribute_reliability"
    target_path = directory / f"{stem}_target.npz"
    lock_path = directory / f"{stem}_target_lock.json"
    summary_path = directory / f"{stem}_summary.json"
    for path in (target_path, lock_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing reliability target input: {path}")
    lock = json.loads(lock_path.read_text())
    summary = json.loads(summary_path.read_text())
    target_hash = _sha256(target_path)
    lock_hash = _sha256(lock_path)
    checks = {
        "reliability_target_declared_label_free": not bool(
            lock.get("contains_target_labels", True)
        ),
        "reliability_target_declared_path_free": not bool(
            lock.get("contains_target_paths", True)
        ),
        "reliability_target_hash_matches_lock": (
            target_hash == lock.get("outputs", {}).get("target_npz", {}).get("sha256")
        ),
        "reliability_lock_hash_matches_summary": (
            lock_hash == summary.get("target_lock_sha256")
        ),
        "reliability_source_hash_matches": (
            source_signal_sha256
            == lock.get("inputs", {}).get("source_signal_npz", {}).get("sha256")
        ),
        "reliability_rule_has_no_label_threshold": not bool(
            lock.get("formula", {}).get("target_label_thresholds", True)
        ),
    }
    with np.load(target_path, allow_pickle=False) as archive:
        forbidden = {"label", "labels", "target_label", "path", "paths"}
        checks["reliability_npz_has_no_label_or_path_key"] = not bool(
            forbidden.intersection(archive.files)
        )
        required = {
            "index",
            "task_prediction",
            "clip_prediction",
            "fixed_clip_probability",
            "candidate_probability",
            "attribute_weight",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Reliability target is missing: {sorted(missing)}")
        arrays = {key: archive[key].copy() for key in required}
    return {
        **arrays,
        "lock": lock,
        "summary": summary,
        "checks": checks,
        "target_path": target_path,
        "target_sha256": target_hash,
        "lock_path": lock_path,
        "lock_sha256": lock_hash,
        "summary_path": summary_path,
        "summary_sha256": _sha256(summary_path),
    }


def _mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _rate(values: np.ndarray, predicate) -> float:
    return float(predicate(values).mean() * 100.0) if values.size else 0.0


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    oracle = summary["oracle_metrics"]
    gate = summary["gate"]
    lines = [
        "# VisDA Attribute-Reliability KL Influence Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "This is an oracle diagnostic of the exact candidate-minus-control KL",
        "descent direction in student-logit space. It is not a parameter-gradient",
        "or training-dynamics claim.",
        "",
        "## Exact proxy intervention",
        "",
        f"- Active proxy conflicts: `{summary['label_free_diagnostic']['active_conflicts']}`.",
        f"- Top-1 target changes: `{summary['label_free_diagnostic']['changed_top1']}`.",
        f"- Mean incremental-direction norm: "
        f"`{summary['label_free_diagnostic']['incremental_direction_norm_mean']:.10f}`.",
        f"- Increment norm / control KL direction norm: "
        f"`{summary['label_free_diagnostic']['increment_vs_control_norm_pct']:.6f}%`.",
        "",
        "## Oracle direction diagnostic",
        "",
        f"- Mean incremental projection: `{oracle['mean_incremental_projection']:.10f}`; "
        f"95% CI `{oracle['incremental_projection_bootstrap_95_ci']}`.",
        f"- Macro class-balanced projection: "
        f"`{oracle['macro_class_mean_incremental_projection']:.10f}`.",
        f"- Helpful / harmful / neutral coverage: "
        f"`{oracle['helpful_coverage_pct']:.4f}% / "
        f"{oracle['harmful_coverage_pct']:.4f}% / "
        f"{oracle['neutral_coverage_pct']:.4f}%`.",
        f"- True class inside task/CLIP pair: `{oracle['candidate_pair_coverage_pct']:.4f}%`.",
        "",
        "## Decisive checks",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`"
        for name, passed in {
            **gate["signal_checks"],
            **gate["translation_checks"],
        }.items()
    )
    lines.extend(
        [
            "",
            f"Diagnosis: `{gate['diagnosis']}`.",
            "",
            "This audit never authorizes another proxy or full VisDA run.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (
        args.target_list,
        args.proxy_list,
        args.class_names,
        args.proxy_gate,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing KL-influence input: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    candidate_log = _resolve_candidate_log(DEFAULT_CANDIDATE_DIR, args.candidate_log)
    logged = _parse_candidate_contract(candidate_log)
    class_names = _load_class_names(args.class_names)
    source = _load_locked_inputs(args.source_dir)
    reliability = _load_reliability_target(
        args.reliability_dir,
        source_signal_sha256=source["signal_npz_sha256"],
    )

    full_paths = _path_fields_without_labels(args.target_list)
    proxy_paths = _path_fields_without_labels(args.proxy_list)
    if len(full_paths) != EXPECTED_TARGET_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_TARGET_SAMPLES} full target paths")
    if len(proxy_paths) != EXPECTED_PROXY_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_PROXY_SAMPLES} proxy target paths")
    if len(set(full_paths)) != len(full_paths) or len(set(proxy_paths)) != len(
        proxy_paths
    ):
        raise RuntimeError("Target list paths must be unique")
    full_index = {path: index for index, path in enumerate(full_paths)}
    try:
        proxy_indices = np.asarray(
            [full_index[path] for path in proxy_paths], dtype=np.int64
        )
    except KeyError as error:
        raise RuntimeError(
            f"Proxy path is absent from full target list: {error}"
        ) from error

    source_checks = dict(source["checks"])
    target_checks = dict(reliability["checks"])
    target_checks.update(
        {
            "target_index_matches_source": np.array_equal(
                reliability["index"], source["index"]
            ),
            "target_task_prediction_matches_source": np.array_equal(
                reliability["task_prediction"], source["task_prediction"]
            ),
            "target_clip_prediction_matches_source": np.array_equal(
                reliability["clip_prediction"], source["clip_prediction"]
            ),
            "fixed_clip_probability_matches_source": np.allclose(
                reliability["fixed_clip_probability"],
                source["clip_probability"],
                atol=1e-12,
                rtol=1e-12,
            ),
            "target_list_hash_matches_source_lock": (
                _sha256(args.target_list) == source["expected_target_list_sha256"]
            ),
        }
    )
    if not all({**source_checks, **target_checks}.values()):
        failed = [
            name
            for name, passed in {**source_checks, **target_checks}.items()
            if not passed
        ]
        raise RuntimeError(f"Locked probability contract failed: {failed}")

    active_position = np.flatnonzero(np.isin(source["index"], proxy_indices))
    task_prediction = source["task_prediction"][active_position]
    clip_prediction = source["clip_prediction"][active_position]
    student = source["task_probability"][active_position]
    control_target = source["clip_probability"][active_position]
    candidate_target = np.asarray(
        reliability["candidate_probability"], dtype=np.float64
    )[active_position]
    attribute_weight = np.asarray(reliability["attribute_weight"], dtype=np.float64)[
        active_position
    ]

    directions = kl_logit_descent_directions(
        student,
        control_target,
        candidate_target,
        kl_weight=args.kl_weight,
    )
    increment = directions["incremental_direction"]
    increment_norm = np.linalg.norm(increment, axis=1)
    control_norm = np.linalg.norm(directions["control_direction"], axis=1)
    changed_top1 = int(
        np.sum(candidate_target.argmax(axis=1) != control_target.argmax(axis=1))
    )
    pair_mask = np.zeros_like(increment, dtype=bool)
    row = np.arange(active_position.size)
    pair_mask[row, task_prediction] = True
    pair_mask[row, clip_prediction] = True
    nonpair_max = float(np.abs(increment[~pair_mask]).max(initial=0.0))
    computed_checks = {
        "proxy_indices_are_unique": len(np.unique(proxy_indices))
        == EXPECTED_PROXY_SAMPLES,
        "active_conflicts_match_candidate_log": active_position.size
        == logged["active_conflicts"],
        "changed_top1_matches_candidate_log": changed_top1 == logged["changed_top1"],
        "mean_attribute_weight_matches_candidate_log": abs(
            float(attribute_weight.mean()) - logged["mean_attribute_weight"]
        )
        <= 5.1e-7,
        "increment_has_only_task_clip_pair_support": nonpair_max <= 1e-12,
        "kl_weight_matches_training_contract": abs(args.kl_weight - KL_WEIGHT) <= 1e-12,
    }
    input_contract_valid = all(
        {**source_checks, **target_checks, **computed_checks}.values()
    )
    if not input_contract_valid:
        failed = [name for name, passed in computed_checks.items() if not passed]
        raise RuntimeError(f"Exact proxy intervention contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_attribute_kl_influence"
    influence_path = args.output_dir / f"{stem}_direction.npz"
    mass_path = args.output_dir / f"{stem}_class_mass.csv"
    lock_path = args.output_dir / f"{stem}_direction_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    classwise_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    category_path = args.output_dir / f"{stem}_categorywise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"

    np.savez_compressed(
        influence_path,
        index=source["index"][active_position],
        task_prediction=task_prediction,
        clip_prediction=clip_prediction,
        student_probability=student,
        control_target=control_target,
        candidate_target=candidate_target,
        attribute_weight=attribute_weight,
        control_direction=directions["control_direction"],
        candidate_direction=directions["candidate_direction"],
        incremental_direction=increment,
    )
    class_mass_rows = []
    class_mass_delta = (candidate_target - control_target).mean(axis=0) * 100.0
    for class_index, class_name in enumerate(class_names):
        class_mass_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "control_expected_mass_pct": float(
                    control_target[:, class_index].mean() * 100.0
                ),
                "candidate_expected_mass_pct": float(
                    candidate_target[:, class_index].mean() * 100.0
                ),
                "candidate_minus_control_pp": float(class_mass_delta[class_index]),
            }
        )
    _write_csv(mass_path, class_mass_rows)

    direction_lock = {
        "phase": "LABEL_FREE_ATTRIBUTE_KL_DIRECTION_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "target_list_path_fields_read": True,
        "target_list_label_fields_parsed": False,
        "oracle_labels_parsed_after_this_manifest": True,
        "formula": {
            "control_logit_descent": "KL_PAR * (q_clip - p_task)",
            "candidate_logit_descent": "KL_PAR * (q_attribute - p_task)",
            "incremental_logit_descent": "KL_PAR * (q_attribute - q_clip)",
            "kl_par": args.kl_weight,
            "network_jacobian_included": False,
            "optimizer_dynamics_included": False,
            "target_label_thresholds": False,
            "class_specific_rules": False,
        },
        "input_contract_checks": {
            **source_checks,
            **target_checks,
            **computed_checks,
        },
        "inputs": {
            "source_signal_npz": {
                "path": str(source["signal_npz_path"]),
                "sha256": source["signal_npz_sha256"],
            },
            "reliability_target_npz": {
                "path": str(reliability["target_path"]),
                "sha256": reliability["target_sha256"],
            },
            "target_list_opaque_sha256": _sha256(args.target_list),
            "proxy_list_opaque_sha256": _sha256(args.proxy_list),
            "candidate_log": {
                "path": str(candidate_log),
                "sha256": _sha256(candidate_log),
                "parsed_fields": [
                    "cycle1_active_conflicts",
                    "cycle1_changed_top1",
                    "cycle1_mean_attribute_weight",
                ],
            },
            "matched_proxy_gate_opaque_sha256": _sha256(args.proxy_gate),
        },
        "outputs": {
            "direction_npz": {
                "path": str(influence_path),
                "sha256": _sha256(influence_path),
            },
            "class_mass_csv": {
                "path": str(mass_path),
                "sha256": _sha256(mass_path),
            },
        },
        "contract_sha256": {
            "src/utils/attribute_kl_influence_audit.py": _sha256(
                REPO_ROOT / "src/utils/attribute_kl_influence_audit.py"
            ),
            "tools/audit_visda_conflict_attribute_kl_influence.py": _sha256(
                Path(__file__).resolve()
            ),
            "src/methods/oh/plmatch.py": _sha256(
                REPO_ROOT / "src/methods/oh/plmatch.py"
            ),
        },
    }
    lock_path.write_text(json.dumps(direction_lock, indent=2) + "\n")

    # Oracle phase starts only after the label-free direction is locked.
    all_labels = _parse_labels_after_lock(args.target_list)
    labels = all_labels[source["index"][active_position]]
    influence = oracle_logit_influence(
        student,
        directions["control_direction"],
        directions["candidate_direction"],
        labels,
    )
    incremental_projection = influence["incremental_projection"]
    projection_ci = paired_bootstrap_mean_ci(
        incremental_projection,
        seed=args.seed,
        repeats=BOOTSTRAP_REPEATS,
    )
    true_mass_gain = candidate_target[row, labels] - control_target[row, labels]
    task_correct = labels == task_prediction
    clip_correct = labels == clip_prediction
    neither_correct = ~(task_correct | clip_correct)
    pair_covered = task_correct | clip_correct
    epsilon = 1e-12
    helpful = incremental_projection > epsilon
    harmful = incremental_projection < -epsilon
    neutral = ~(helpful | harmful)

    class_rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        if not mask.any():
            raise RuntimeError(f"Proxy active conflicts contain no {class_name} rows")
        values = incremental_projection[mask]
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "samples": int(mask.sum()),
                "task_clip_pair_coverage_pct": _rate(pair_covered[mask], lambda x: x),
                "mean_incremental_projection": float(values.mean()),
                "mean_control_projection": float(
                    influence["control_projection"][mask].mean()
                ),
                "mean_candidate_projection": float(
                    influence["candidate_projection"][mask].mean()
                ),
                "mean_true_probability_gain": float(true_mass_gain[mask].mean()),
                "helpful_coverage_pct": _rate(values, lambda x: x > epsilon),
                "harmful_coverage_pct": _rate(values, lambda x: x < -epsilon),
                "neutral_coverage_pct": _rate(values, lambda x: np.abs(x) <= epsilon),
            }
        )
    _write_csv(classwise_path, class_rows)

    categories = {
        "task_correct": task_correct,
        "clip_correct": clip_correct,
        "neither_candidate_correct": neither_correct,
    }
    category_rows = []
    for name, mask in categories.items():
        values = incremental_projection[mask]
        category_rows.append(
            {
                "category": name,
                "samples": int(mask.sum()),
                "coverage_pct": float(mask.mean() * 100.0),
                "mean_incremental_projection": _mean_or_zero(values),
                "mean_true_probability_gain": _mean_or_zero(true_mass_gain[mask]),
                "helpful_coverage_pct": _rate(values, lambda x: x > epsilon),
                "harmful_coverage_pct": _rate(values, lambda x: x < -epsilon),
                "neutral_coverage_pct": _rate(values, lambda x: np.abs(x) <= epsilon),
            }
        )
    _write_csv(category_path, category_rows)

    oracle_rows = []
    for local_index, target_index in enumerate(source["index"][active_position]):
        oracle_rows.append(
            {
                "index": int(target_index),
                "label": int(labels[local_index]),
                "label_name": class_names[int(labels[local_index])],
                "task_top1": int(task_prediction[local_index]),
                "clip_top1": int(clip_prediction[local_index]),
                "candidate_pair_covers_label": bool(pair_covered[local_index]),
                "attribute_weight": float(attribute_weight[local_index]),
                "incremental_direction_norm": float(increment_norm[local_index]),
                "control_projection": float(
                    influence["control_projection"][local_index]
                ),
                "candidate_projection": float(
                    influence["candidate_projection"][local_index]
                ),
                "incremental_projection": float(incremental_projection[local_index]),
                "control_cosine": float(influence["control_cosine"][local_index]),
                "candidate_cosine": float(influence["candidate_cosine"][local_index]),
                "true_probability_gain": float(true_mass_gain[local_index]),
                "influence": (
                    "helpful"
                    if helpful[local_index]
                    else "harmful" if harmful[local_index] else "neutral"
                ),
            }
        )
    _write_csv(oracle_path, oracle_rows)

    by_class = {row["class"]: row for row in class_rows}
    macro_class_projection = float(
        np.mean([row["mean_incremental_projection"] for row in class_rows])
    )
    proxy_gate = json.loads(args.proxy_gate.read_text())
    observed_contract = {
        "gate_is_matched_four_cycle": bool(
            proxy_gate.get("checks", {}).get("matched_four_cycle_contract")
        ),
        "gate_uses_final_not_oracle_peak": not bool(
            proxy_gate.get("thresholds", {}).get("selection_uses_oracle_peak", True)
        ),
        "gate_reports_numeric_final_delta": isinstance(
            proxy_gate.get("final_delta"), (int, float)
        ),
    }
    if not all(observed_contract.values()):
        raise RuntimeError(f"Matched proxy gate contract failed: {observed_contract}")
    gate = evaluate_attribute_kl_influence(
        input_contract_valid=input_contract_valid,
        active_conflict_count_matches=computed_checks[
            "active_conflicts_match_candidate_log"
        ],
        changed_top1_count_matches=computed_checks[
            "changed_top1_matches_candidate_log"
        ],
        mean_incremental_projection=float(incremental_projection.mean()),
        incremental_projection_ci=projection_ci,
        macro_class_mean_projection=macro_class_projection,
        car_mean_projection=by_class["car"]["mean_incremental_projection"],
        truck_mean_projection=by_class["truck"]["mean_incremental_projection"],
        observed_final_delta_pp=float(proxy_gate["final_delta"]),
        observed_hard_mean_delta_pp=float(proxy_gate["hard_mean_delta"]),
        min_observed_final_delta_pp=float(
            proxy_gate["thresholds"]["min_final_macro_improvement_pp"]
        ),
    )

    control_norm_sum = float(control_norm.sum())
    summary = {
        "dataset": "VISDA-C",
        "seed": args.seed,
        "oracle_diagnostic": True,
        "labels_used_only_after_direction_lock": True,
        "direction_lock": str(lock_path),
        "direction_lock_sha256": _sha256(lock_path),
        "scope": "exact proxy25 cycle-1 unresolved task/CLIP conflicts",
        "interpretation_limit": (
            "exact KL derivative in student-logit space at the locked weak-view "
            "probability; excludes network Jacobian, augmentation drift, optimizer, "
            "and multi-step training dynamics"
        ),
        "input_contract": {
            "passed": input_contract_valid and all(observed_contract.values()),
            "checks": {
                **source_checks,
                **target_checks,
                **computed_checks,
                **observed_contract,
            },
        },
        "label_free_diagnostic": {
            "proxy_samples": EXPECTED_PROXY_SAMPLES,
            "active_conflicts": int(active_position.size),
            "active_conflict_fraction_of_proxy_pct": float(
                active_position.size / EXPECTED_PROXY_SAMPLES * 100.0
            ),
            "changed_top1": changed_top1,
            "changed_top1_fraction_of_active_pct": float(
                changed_top1 / active_position.size * 100.0
            ),
            "mean_attribute_weight": float(attribute_weight.mean()),
            "logged_mean_attribute_weight": logged["mean_attribute_weight"],
            "incremental_direction_nonzero_fraction_pct": float(
                np.mean(increment_norm > epsilon) * 100.0
            ),
            "incremental_direction_norm_mean": float(increment_norm.mean()),
            "incremental_direction_norm_median": float(np.median(increment_norm)),
            "incremental_direction_norm_p95": float(
                np.percentile(increment_norm, 95.0)
            ),
            "control_direction_norm_mean": float(control_norm.mean()),
            "increment_vs_control_norm_pct": (
                float(increment_norm.sum() / control_norm_sum * 100.0)
                if control_norm_sum > 0.0
                else 0.0
            ),
            "max_abs_nonpair_increment": nonpair_max,
            "class_mass_rows": class_mass_rows,
        },
        "oracle_metrics": {
            "samples": int(active_position.size),
            "candidate_pair_coverage_pct": float(pair_covered.mean() * 100.0),
            "task_correct_coverage_pct": float(task_correct.mean() * 100.0),
            "clip_correct_coverage_pct": float(clip_correct.mean() * 100.0),
            "neither_correct_coverage_pct": float(neither_correct.mean() * 100.0),
            "mean_control_projection": float(influence["control_projection"].mean()),
            "mean_candidate_projection": float(
                influence["candidate_projection"].mean()
            ),
            "mean_incremental_projection": float(incremental_projection.mean()),
            "incremental_projection_bootstrap_95_ci": list(projection_ci),
            "macro_class_mean_incremental_projection": macro_class_projection,
            "mean_control_cosine": float(influence["control_cosine"].mean()),
            "mean_candidate_cosine": float(influence["candidate_cosine"].mean()),
            "mean_true_probability_gain": float(true_mass_gain.mean()),
            "helpful_coverage_pct": float(helpful.mean() * 100.0),
            "harmful_coverage_pct": float(harmful.mean() * 100.0),
            "neutral_coverage_pct": float(neutral.mean() * 100.0),
            "classwise": class_rows,
            "categorywise": category_rows,
        },
        "matched_proxy_observation": {
            "decision": proxy_gate.get("decision"),
            "control_final": proxy_gate.get("control_final"),
            "candidate_final": proxy_gate.get("candidate_final"),
            "final_delta_pp": proxy_gate.get("final_delta"),
            "hard_mean_delta_pp": proxy_gate.get("hard_mean_delta"),
            "hard_class_deltas": proxy_gate.get("hard_class_deltas"),
        },
        "gate": gate,
        "decision": gate["decision"],
        "next": (
            "stop the attribute-reliability branch; do not increase KL weight, "
            "extend cycles, rerun proxy, or run full VisDA"
            if gate["decision"] == "REJECT_ATTRIBUTE_BRANCH"
            else "diagnostic only; no training is authorized"
        ),
        "compute_contract": {
            "target_images_loaded": 0,
            "model_checkpoint_loads": 0,
            "model_forward_calls": 0,
            "network_jacobians": 0,
            "optimizer_constructed": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "gpu_required": False,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
        },
        "safety_contract": {
            "target_images_loaded": False,
            "model_checkpoint_loads": 0,
            "model_forward_calls": 0,
            "optimizer_constructed": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "model_parameters_updated": False,
            "training_code_modified": False,
            "training_authorized": False,
        },
        "runtime_seconds": float(time.monotonic() - started),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)

    for label, output in (
        ("label-free direction", influence_path),
        ("label-free class mass", mass_path),
        ("direction lock", lock_path),
        ("oracle diagnostic", oracle_path),
        ("classwise oracle diagnostic", classwise_path),
        ("categorywise oracle diagnostic", category_path),
        ("summary", summary_path),
    ):
        print(f"Wrote {label}: {output}")
    print(json.dumps({"decision": summary["decision"], "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
