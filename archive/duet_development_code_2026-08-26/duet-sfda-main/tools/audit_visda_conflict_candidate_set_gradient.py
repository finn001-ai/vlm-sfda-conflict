#!/usr/bin/env python
"""CPU-only oracle audit of candidate-set versus DUET KL logit directions.

Phase 1 reads only locked label-free probability and candidate-set tensors,
computes the exact logit descents for DUET's CLIP KL and top-1/top-2 set-mass
losses, and locks those directions. Phase 2 parses target labels solely for
oracle gradient diagnostics. No image, model, checkpoint, optimizer, backward
pass, parameter update, or training is involved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.candidate_set_audit import union_candidate_mask  # noqa: E402
from src.utils.candidate_set_gradient_audit import (  # noqa: E402
    evaluate_candidate_gradient_gate,
    kl_logit_descent,
    oracle_ce_logit_descent,
    paired_mean_bootstrap_ci,
    rowwise_oracle_alignment,
    set_mass_logit_descent,
)


EXPECTED_CONFLICT_SAMPLES = 28_255
EXPECTED_TARGET_SAMPLES = 55_388
EXPECTED_CLASSES = 12
BOOTSTRAP_REPEATS = 2_000
DEFAULT_PROBABILITY_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_pairwise_attribute_audit_seed2020/pairwise_attribute_audit"
)
DEFAULT_CANDIDATE_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_candidate_set_audit_seed2020/candidate_set_audit"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_candidate_set_gradient_audit_seed2020/"
    "candidate_set_gradient_audit"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-dir", type=Path, default=DEFAULT_PROBABILITY_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-list",
        type=Path,
        default=Path("data/VISDA-C/validation_list.txt"),
    )
    parser.add_argument(
        "--class-names", type=Path, default=Path("data/VISDA-C/classname.txt")
    )
    parser.add_argument("--seed", type=int, default=2020)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pct(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=np.float64).mean() * 100.0)


def _load_class_names(path: Path) -> list[str]:
    names = [
        line.strip().replace("_", " ")
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(names) != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} classes, found {len(names)}")
    return names


def _parse_labels_after_lock(path: Path) -> np.ndarray:
    labels = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            _image_path, label_text = stripped.rsplit(maxsplit=1)
            label = int(label_text)
        except ValueError as error:
            raise ValueError(
                f"Malformed VisDA row {line_number} in {path}: {stripped}"
            ) from error
        labels.append(label)
    result = np.asarray(labels, dtype=np.int64)
    if result.shape != (EXPECTED_TARGET_SAMPLES,):
        raise ValueError(
            f"Expected {EXPECTED_TARGET_SAMPLES} target labels, found {result.size}"
        )
    if np.any(result < 0) or np.any(result >= EXPECTED_CLASSES):
        raise ValueError("Target label is outside the class range")
    return result


def _load_label_free_inputs(
    probability_dir: Path,
    candidate_dir: Path,
) -> dict[str, Any]:
    probability_stem = "visda_conflict_pairwise_attribute"
    probability_path = probability_dir / f"{probability_stem}_signals.npz"
    probability_lock_path = probability_dir / f"{probability_stem}_signal_lock.json"
    candidate_stem = "visda_conflict_candidate_set"
    candidate_path = candidate_dir / f"{candidate_stem}_label_free.npz"
    candidate_lock_path = candidate_dir / f"{candidate_stem}_signal_lock.json"
    for path in (
        probability_path,
        probability_lock_path,
        candidate_path,
        candidate_lock_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing candidate-gradient input: {path}")

    probability_lock = json.loads(probability_lock_path.read_text())
    candidate_lock = json.loads(candidate_lock_path.read_text())
    probability_hash = _sha256(probability_path)
    probability_lock_hash = _sha256(probability_lock_path)
    candidate_hash = _sha256(candidate_path)
    candidate_lock_hash = _sha256(candidate_lock_path)
    expected_target_hash = probability_lock.get("target_list_sha256")
    candidate_target_hash = candidate_lock.get("inputs", {}).get(
        "target_list_expected_opaque_sha256"
    )
    checks: dict[str, bool] = {
        "probability_declared_label_free": not bool(
            probability_lock.get("contains_target_labels", True)
        ),
        "probability_loader_labels_ignored": bool(
            probability_lock.get("loader_labels_ignored")
        ),
        "probability_hash_matches_lock": probability_hash
        == probability_lock.get("signal_npz", {}).get("sha256"),
        "candidate_declared_label_free": not bool(
            candidate_lock.get("contains_target_labels", True)
        ),
        "candidate_declared_path_free": not bool(
            candidate_lock.get("contains_target_paths", True)
        ),
        "candidate_target_list_not_read_before_lock": not bool(
            candidate_lock.get("target_list_read_before_lock", True)
        ),
        "candidate_hash_matches_lock": candidate_hash
        == candidate_lock.get("outputs", {}).get("candidate_set_npz", {}).get("sha256"),
        "candidate_references_same_probability": probability_hash
        == candidate_lock.get("inputs", {}).get("source_signal_npz", {}).get("sha256"),
        "candidate_references_same_probability_lock": probability_lock_hash
        == candidate_lock.get("inputs", {}).get("source_signal_lock", {}).get("sha256"),
        "opaque_target_hashes_match": isinstance(expected_target_hash, str)
        and len(expected_target_hash) == 64
        and expected_target_hash == candidate_target_hash,
    }

    with np.load(probability_path, allow_pickle=False) as archive:
        forbidden = {"label", "labels", "target_label", "path", "paths"}
        checks["probability_npz_has_no_label_or_path_key"] = not bool(
            forbidden.intersection(archive.files)
        )
        required = {
            "index",
            "task_probability",
            "clip_probability",
            "task_prediction",
            "clip_prediction",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Probability signal is missing: {sorted(missing)}")
        probability_arrays = {key: archive[key].copy() for key in required}
    with np.load(candidate_path, allow_pickle=False) as archive:
        forbidden = {"label", "labels", "target_label", "path", "paths"}
        checks["candidate_npz_has_no_label_or_path_key"] = not bool(
            forbidden.intersection(archive.files)
        )
        required = {
            "index",
            "task_top1",
            "clip_top1",
            "task_top2",
            "clip_top2",
            "top1_union_mask",
            "top2_union_mask",
            "top2_union_set_size",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Candidate-set signal is missing: {sorted(missing)}")
        candidate_arrays = {key: archive[key].copy() for key in required}

    index = np.asarray(probability_arrays["index"], dtype=np.int64)
    candidate_index = np.asarray(candidate_arrays["index"], dtype=np.int64)
    task = np.asarray(probability_arrays["task_probability"], dtype=np.float64)
    clip = np.asarray(probability_arrays["clip_probability"], dtype=np.float64)
    task_prediction = np.asarray(probability_arrays["task_prediction"], dtype=np.int64)
    clip_prediction = np.asarray(probability_arrays["clip_prediction"], dtype=np.int64)
    top1_mask = np.asarray(candidate_arrays["top1_union_mask"], dtype=bool)
    top2_mask = np.asarray(candidate_arrays["top2_union_mask"], dtype=bool)
    recomputed_top1 = union_candidate_mask(task, clip, k=1)
    recomputed_top2 = union_candidate_mask(task, clip, k=2)
    checks.update(
        {
            "expected_conflict_count": index.shape == (EXPECTED_CONFLICT_SAMPLES,),
            "indices_strictly_increasing": bool(np.all(np.diff(index) > 0)),
            "candidate_indices_match_probabilities": np.array_equal(
                index, candidate_index
            ),
            "probability_shapes_valid": task.shape
            == clip.shape
            == (EXPECTED_CONFLICT_SAMPLES, EXPECTED_CLASSES),
            "probabilities_finite": bool(
                np.isfinite(task).all() and np.isfinite(clip).all()
            ),
            "probabilities_nonnegative": bool(
                np.all(task >= 0.0) and np.all(clip >= 0.0)
            ),
            "probabilities_normalized": bool(
                np.allclose(task.sum(axis=1), 1.0, atol=1e-5)
                and np.allclose(clip.sum(axis=1), 1.0, atol=1e-5)
            ),
            "saved_predictions_match_probabilities": np.array_equal(
                task_prediction, task.argmax(axis=1)
            )
            and np.array_equal(clip_prediction, clip.argmax(axis=1)),
            "all_rows_are_conflicts": bool(np.all(task_prediction != clip_prediction)),
            "top1_mask_shape_valid": top1_mask.shape == task.shape,
            "top2_mask_shape_valid": top2_mask.shape == task.shape,
            "top1_mask_exactly_reproduced": np.array_equal(
                top1_mask, recomputed_top1["union_mask"]
            ),
            "top2_mask_exactly_reproduced": np.array_equal(
                top2_mask, recomputed_top2["union_mask"]
            ),
            "task_top2_exactly_reproduced": np.array_equal(
                candidate_arrays["task_top2"], recomputed_top2["task_topk"]
            ),
            "clip_top2_exactly_reproduced": np.array_equal(
                candidate_arrays["clip_top2"], recomputed_top2["clip_topk"]
            ),
            "top2_sizes_exactly_reproduced": np.array_equal(
                candidate_arrays["top2_union_set_size"],
                recomputed_top2["set_size"],
            ),
            "top2_contains_top1": bool(np.all(~top1_mask | top2_mask)),
        }
    )
    return {
        "index": index,
        "task_probability": task,
        "clip_probability": clip,
        "top1_mask": top1_mask,
        "top2_mask": top2_mask,
        "checks": checks,
        "expected_target_hash": expected_target_hash,
        "paths": {
            "probability_npz": probability_path,
            "probability_lock": probability_lock_path,
            "candidate_npz": candidate_path,
            "candidate_lock": candidate_lock_path,
        },
        "hashes": {
            "probability_npz": probability_hash,
            "probability_lock": probability_lock_hash,
            "candidate_npz": candidate_hash,
            "candidate_lock": candidate_lock_hash,
        },
    }


def _method_metrics(alignment: dict[str, np.ndarray]) -> dict[str, Any]:
    first_order = alignment["first_order"]
    tolerance = 1e-15
    return {
        "mean_cosine": float(alignment["cosine"].mean()),
        "mean_oracle_unit_projection": float(
            alignment["oracle_unit_projection"].mean()
        ),
        "mean_first_order": float(first_order.mean()),
        "mean_direction_norm": float(alignment["candidate_norm"].mean()),
        "joint_nonzero_fraction_pct": _pct(alignment["joint_nonzero"]),
        "helpful_fraction_pct": _pct(first_order > tolerance),
        "harmful_fraction_pct": _pct(first_order < -tolerance),
        "neutral_fraction_pct": _pct(np.abs(first_order) <= tolerance),
    }


def _comparison(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    *,
    seed: int,
) -> dict[str, Any]:
    result = {}
    for offset, metric in enumerate(
        ("cosine", "oracle_unit_projection", "first_order")
    ):
        difference = candidate[metric] - baseline[metric]
        ci = paired_mean_bootstrap_ci(
            difference,
            repeats=BOOTSTRAP_REPEATS,
            seed=seed + offset,
        )
        result[metric] = {
            "mean_difference": float(difference.mean()),
            "paired_bootstrap_95_ci": [float(ci[0]), float(ci[1])],
        }
    return result


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    methods = summary["oracle_metrics"]["methods"]
    comparisons = summary["oracle_metrics"]["comparisons"]
    lines = [
        "# VisDA Conflict Candidate-Set Gradient Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "All candidate directions were computed and locked before target labels",
        "were parsed. Values below are oracle diagnostics in task-logit space,",
        "not measured training accuracy.",
        "",
        "## Mean oracle alignment",
        "",
        "| Direction | Cosine | Unit-oracle projection | First-order | Harmful |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("clip_kl", "DUET CLIP KL"),
        ("top1_set", "Top-1 set loss"),
        ("top2_set", "Top-2 set loss"),
    ):
        metric = methods[key]
        lines.append(
            f"| {label} | {metric['mean_cosine']:.8f} | "
            f"{metric['mean_oracle_unit_projection']:.8f} | "
            f"{metric['mean_first_order']:.8f} | "
            f"{metric['harmful_fraction_pct']:.4f}% |"
        )
    lines.extend(["", "## Paired differences: top-2 minus baseline", ""])
    for comparison, result in comparisons.items():
        lines.append(f"- `{comparison}`:")
        for metric, values in result.items():
            lines.append(
                f"  - {metric}: mean `{values['mean_difference']:.10f}`, "
                f"95% CI `{values['paired_bootstrap_95_ci']}`."
            )
    lines.extend(["", "## Gate", ""])
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend(
        [
            "",
            "Passing authorizes only one matched proxy design; it never starts",
            "or authorizes VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing candidate-gradient input: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    class_names = _load_class_names(args.class_names)
    source = _load_label_free_inputs(args.probability_dir, args.candidate_dir)
    if not all(source["checks"].values()):
        failed = [name for name, passed in source["checks"].items() if not passed]
        raise RuntimeError(f"Locked candidate-gradient input contract failed: {failed}")

    directions = {
        "clip_kl": kl_logit_descent(
            source["task_probability"], source["clip_probability"]
        ),
        "top1_set": set_mass_logit_descent(
            source["task_probability"], source["top1_mask"]
        ),
        "top2_set": set_mass_logit_descent(
            source["task_probability"], source["top2_mask"]
        ),
    }
    direction_checks = {
        f"{name}_finite": bool(np.isfinite(direction).all())
        for name, direction in directions.items()
    }
    direction_checks.update(
        {
            f"{name}_row_sum_zero": bool(
                np.allclose(direction.sum(axis=1), 0.0, atol=1e-10, rtol=1e-10)
            )
            for name, direction in directions.items()
        }
    )
    input_contract_valid = all({**source["checks"], **direction_checks}.values())
    if not input_contract_valid:
        failed = [name for name, passed in direction_checks.items() if not passed]
        raise RuntimeError(f"Candidate-gradient construction failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_candidate_set_gradient"
    direction_path = args.output_dir / f"{stem}_label_free.npz"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"

    np.savez_compressed(
        direction_path,
        index=source["index"],
        clip_kl_descent=directions["clip_kl"],
        top1_set_descent=directions["top1_set"],
        top2_set_descent=directions["top2_set"],
    )
    signal_lock = {
        "phase": "LABEL_FREE_CANDIDATE_SET_GRADIENT_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "target_list_read_before_lock": False,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate": "top2_union_set_mass_loss",
        "direction_contract": {
            "duet_clip_kl_descent": "clip_probability - task_probability",
            "set_loss": "-log(sum(task_probability[c] for c in candidate_set))",
            "set_loss_descent": (
                "1[c in S] * task_probability[c] / task_set_mass "
                "- task_probability[c]"
            ),
            "top1_set": "task top-1 union CLIP top-1",
            "top2_set": "task top-2 union CLIP top-2",
            "loss_weight_for_direction_comparison": (
                "common positive scalar omitted; cosine invariant and magnitude "
                "compared at equal coefficient"
            ),
        },
        "predeclared_oracle_gate": {
            "primary_metrics": [
                "rowwise cosine",
                "oracle-unit projection",
                "first-order oracle log-probability increase",
            ],
            "top2_minus_clip_mean": "> 0 for all primary metrics",
            "top2_minus_clip_paired_bootstrap_95_ci_lower": (
                "> 0 for all primary metrics"
            ),
            "top2_minus_top1_mean": "> 0 for all primary metrics",
            "top2_minus_top1_paired_bootstrap_95_ci_lower": (
                "> 0 for all primary metrics"
            ),
            "class_macro_first_order_delta_vs_clip": "> 0",
            "car_person_truck_first_order_delta_vs_clip": ">= 0 each",
            "top2_harmful_fraction_minus_clip": "<= 0",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "fitted_thresholds": False,
        },
        "input_contract_checks": {**source["checks"], **direction_checks},
        "inputs": {
            name: {"path": str(source["paths"][name]), "sha256": digest}
            for name, digest in source["hashes"].items()
        },
        "target_list_expected_opaque_sha256": source["expected_target_hash"],
        "class_names_sha256": _sha256(args.class_names),
        "output": {
            "direction_npz": {
                "path": str(direction_path),
                "sha256": _sha256(direction_path),
            }
        },
        "contract_sha256": {
            "src/utils/candidate_set_gradient_audit.py": _sha256(
                REPO_ROOT / "src/utils/candidate_set_gradient_audit.py"
            ),
            "tools/audit_visda_conflict_candidate_set_gradient.py": _sha256(
                Path(__file__).resolve()
            ),
            "src/methods/oh/plmatch.py": _sha256(
                REPO_ROOT / "src/methods/oh/plmatch.py"
            ),
            "cfgs/visda/plmatch.yaml": _sha256(REPO_ROOT / "cfgs/visda/plmatch.yaml"),
        },
    }
    lock_path.write_text(json.dumps(signal_lock, indent=2) + "\n")

    # Oracle phase: target-list content is first read after the signal lock.
    target_list_hash = _sha256(args.target_list)
    target_hash_matches = target_list_hash == source["expected_target_hash"]
    if not target_hash_matches:
        raise RuntimeError("Target-list hash does not match locked probability input")
    labels = _parse_labels_after_lock(args.target_list)[source["index"]]
    oracle_direction = oracle_ce_logit_descent(source["task_probability"], labels)
    alignments = {
        name: rowwise_oracle_alignment(direction, oracle_direction)
        for name, direction in directions.items()
    }
    method_metrics = {
        name: _method_metrics(alignment) for name, alignment in alignments.items()
    }
    comparisons = {
        "versus_clip_kl": _comparison(
            alignments["top2_set"], alignments["clip_kl"], seed=args.seed
        ),
        "versus_top1_set": _comparison(
            alignments["top2_set"], alignments["top1_set"], seed=args.seed + 10
        ),
    }

    class_rows = []
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        row: dict[str, Any] = {
            "class_index": class_index,
            "class": class_name,
            "conflict_samples": int(mask.sum()),
        }
        for method in ("clip_kl", "top1_set", "top2_set"):
            alignment = alignments[method]
            first_order = alignment["first_order"][mask]
            row.update(
                {
                    f"{method}_mean_cosine": float(alignment["cosine"][mask].mean()),
                    f"{method}_mean_oracle_unit_projection": float(
                        alignment["oracle_unit_projection"][mask].mean()
                    ),
                    f"{method}_mean_first_order": float(first_order.mean()),
                    f"{method}_harmful_fraction_pct": _pct(first_order < -1e-15),
                    f"{method}_joint_nonzero_fraction_pct": _pct(
                        alignment["joint_nonzero"][mask]
                    ),
                }
            )
        row.update(
            {
                "top2_minus_clip_mean_cosine": (
                    row["top2_set_mean_cosine"] - row["clip_kl_mean_cosine"]
                ),
                "top2_minus_clip_mean_oracle_unit_projection": (
                    row["top2_set_mean_oracle_unit_projection"]
                    - row["clip_kl_mean_oracle_unit_projection"]
                ),
                "top2_minus_clip_mean_first_order": (
                    row["top2_set_mean_first_order"] - row["clip_kl_mean_first_order"]
                ),
                "top2_minus_top1_mean_first_order": (
                    row["top2_set_mean_first_order"] - row["top1_set_mean_first_order"]
                ),
            }
        )
        class_rows.append(row)
    _write_csv(class_path, class_rows)

    oracle_rows = []
    for position, target_index in enumerate(source["index"]):
        label = int(labels[position])
        row: dict[str, Any] = {
            "index": int(target_index),
            "label": label,
            "label_name": class_names[label],
        }
        for method in ("clip_kl", "top1_set", "top2_set"):
            alignment = alignments[method]
            row.update(
                {
                    f"{method}_cosine": float(alignment["cosine"][position]),
                    f"{method}_oracle_unit_projection": float(
                        alignment["oracle_unit_projection"][position]
                    ),
                    f"{method}_first_order": float(alignment["first_order"][position]),
                    f"{method}_direction_norm": float(
                        alignment["candidate_norm"][position]
                    ),
                }
            )
        row.update(
            {
                "top2_minus_clip_cosine": (
                    row["top2_set_cosine"] - row["clip_kl_cosine"]
                ),
                "top2_minus_clip_oracle_unit_projection": (
                    row["top2_set_oracle_unit_projection"]
                    - row["clip_kl_oracle_unit_projection"]
                ),
                "top2_minus_clip_first_order": (
                    row["top2_set_first_order"] - row["clip_kl_first_order"]
                ),
                "top2_minus_top1_first_order": (
                    row["top2_set_first_order"] - row["top1_set_first_order"]
                ),
            }
        )
        oracle_rows.append(row)
    _write_csv(oracle_path, oracle_rows)

    macro_first_order_delta = float(
        np.mean([row["top2_minus_clip_mean_first_order"] for row in class_rows])
    )
    by_name = {row["class"]: row for row in class_rows}
    hard_class_delta = {
        name: float(by_name[name]["top2_minus_clip_mean_first_order"])
        for name in ("car", "person", "truck")
    }
    gate = evaluate_candidate_gradient_gate(
        input_contract_valid=input_contract_valid and target_hash_matches,
        comparisons=comparisons,
        macro_first_order_delta_vs_clip=macro_first_order_delta,
        hard_class_first_order_delta_vs_clip=hard_class_delta,
        top2_harmful_pct=method_metrics["top2_set"]["harmful_fraction_pct"],
        clip_harmful_pct=method_metrics["clip_kl"]["harmful_fraction_pct"],
    )
    summary = {
        "dataset": "VISDA-C",
        "task": "train->validation",
        "seed": args.seed,
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "candidate": "top2_union_set_mass_loss",
        "input_contract": {
            "passed": input_contract_valid and target_hash_matches,
            "checks": {
                **source["checks"],
                **direction_checks,
                "target_list_hash_matches_after_lock": target_hash_matches,
            },
        },
        "label_free_metrics": {
            "conflict_samples": EXPECTED_CONFLICT_SAMPLES,
            "top1_mean_set_mass": float(
                np.where(source["top1_mask"], source["task_probability"], 0.0)
                .sum(axis=1)
                .mean()
            ),
            "top2_mean_set_mass": float(
                np.where(source["top2_mask"], source["task_probability"], 0.0)
                .sum(axis=1)
                .mean()
            ),
        },
        "oracle_metrics": {
            "metric_definitions": {
                "first_order": (
                    "candidate_descent dot oracle_CE_descent; positive predicts "
                    "an infinitesimal oracle log-probability increase"
                ),
                "oracle_unit_projection": (
                    "first_order divided by oracle direction norm"
                ),
                "cosine": (
                    "direction-only alignment; zero when either direction norm "
                    "is at most 1e-15"
                ),
                "harmful": "first_order < -1e-15",
            },
            "methods": method_metrics,
            "comparisons": comparisons,
            "class_macro_first_order_delta_vs_clip": macro_first_order_delta,
            "hard_class_first_order_delta_vs_clip": hard_class_delta,
            "classwise": class_rows,
        },
        "gate": gate,
        "decision": gate["decision"],
        "next": (
            "eligible to design one matched proxy; no training is authorized"
            if gate["decision"] == "PASS_SET_GRADIENT_PREFLIGHT"
            else "reject set-mass supervision; do not run a proxy or full training"
        ),
        "scope_limit": (
            "Exact only for task-logit directions at the locked first-cycle view; "
            "does not identify parameter-gradient interactions or later dynamics."
        ),
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
        ("label-free directions", direction_path),
        ("signal lock", lock_path),
        ("oracle diagnostic", oracle_path),
        ("classwise oracle diagnostic", class_path),
        ("summary", summary_path),
        ("markdown summary", markdown_path),
    ):
        print(f"Wrote {label}: {output}")
    print(
        json.dumps(
            {"decision": summary["decision"], "checks": gate["checks"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
