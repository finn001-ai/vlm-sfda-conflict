#!/usr/bin/env python
"""CPU-only audit of task/CLIP top-1 versus top-2 candidate-set coverage.

Phase 1 reads a previously locked label-free probability NPZ, constructs fixed
candidate sets, and locks them. Phase 2 reads VisDA labels only for explicitly
marked oracle coverage diagnostics. No target image, checkpoint, model forward,
optimizer, backward pass, or training is involved.
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

from src.utils.candidate_set_audit import (  # noqa: E402
    candidate_coverage,
    evaluate_candidate_set_gate,
    union_candidate_mask,
)


EXPECTED_CONFLICT_SAMPLES = 28_255
EXPECTED_TARGET_SAMPLES = 55_388
EXPECTED_CLASSES = 12
DEFAULT_INPUT_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_pairwise_attribute_audit_seed2020/pairwise_attribute_audit"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_candidate_set_audit_seed2020/candidate_set_audit"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-list",
        type=Path,
        default=Path("data/VISDA-C/validation_list.txt"),
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=Path("data/VISDA-C/classname.txt"),
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


def _load_locked_probabilities(input_dir: Path) -> dict[str, Any]:
    stem = "visda_conflict_pairwise_attribute"
    signal_path = input_dir / f"{stem}_signals.npz"
    lock_path = input_dir / f"{stem}_signal_lock.json"
    summary_path = input_dir / f"{stem}_summary.json"
    for path in (signal_path, lock_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing candidate-set input: {path}")

    lock = json.loads(lock_path.read_text())
    summary = json.loads(summary_path.read_text())
    signal_hash = _sha256(signal_path)
    lock_hash = _sha256(lock_path)
    checks = {
        "source_signal_declared_label_free": not bool(
            lock.get("contains_target_labels", True)
        ),
        "source_loader_labels_ignored": bool(lock.get("loader_labels_ignored")),
        "source_signal_hash_matches_lock": (
            signal_hash == lock.get("signal_npz", {}).get("sha256")
        ),
        "source_lock_hash_matches_summary": (
            lock_hash == summary.get("signal_lock_sha256")
        ),
        "source_labels_locked_before_oracle": bool(
            summary.get("labels_used_only_after_signal_lock")
        ),
        "source_baseline_reproduced": bool(
            summary.get("baseline_reproduction", {}).get("passed")
        ),
        "source_target_list_hash_present": isinstance(
            lock.get("target_list_sha256"), str
        )
        and len(lock.get("target_list_sha256", "")) == 64,
    }
    with np.load(signal_path, allow_pickle=False) as archive:
        forbidden = {"label", "labels", "target_label", "path", "paths"}
        checks["source_npz_has_no_label_or_path_key"] = not bool(
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
            raise ValueError(f"Source signal is missing: {sorted(missing)}")
        arrays = {key: archive[key].copy() for key in required}

    index = np.asarray(arrays["index"], dtype=np.int64)
    task_probability = np.asarray(arrays["task_probability"], dtype=np.float64)
    clip_probability = np.asarray(arrays["clip_probability"], dtype=np.float64)
    task_prediction = np.asarray(arrays["task_prediction"], dtype=np.int64)
    clip_prediction = np.asarray(arrays["clip_prediction"], dtype=np.int64)
    checks.update(
        {
            "expected_conflict_count": index.shape == (EXPECTED_CONFLICT_SAMPLES,),
            "indices_strictly_increasing": bool(np.all(np.diff(index) > 0)),
            "indices_inside_full_target": bool(
                np.all(index >= 0) and np.all(index < EXPECTED_TARGET_SAMPLES)
            ),
            "probability_shapes_valid": task_probability.shape
            == clip_probability.shape
            == (EXPECTED_CONFLICT_SAMPLES, EXPECTED_CLASSES),
            "probabilities_finite": bool(
                np.isfinite(task_probability).all()
                and np.isfinite(clip_probability).all()
            ),
            "probabilities_nonnegative": bool(
                np.all(task_probability >= 0.0) and np.all(clip_probability >= 0.0)
            ),
            "probability_rows_normalized": bool(
                np.allclose(task_probability.sum(axis=1), 1.0, atol=1e-5)
                and np.allclose(clip_probability.sum(axis=1), 1.0, atol=1e-5)
            ),
            "saved_task_top1_matches_probability": np.array_equal(
                task_prediction, task_probability.argmax(axis=1)
            ),
            "saved_clip_top1_matches_probability": np.array_equal(
                clip_prediction, clip_probability.argmax(axis=1)
            ),
            "all_rows_are_task_clip_conflicts": bool(
                np.all(task_prediction != clip_prediction)
            ),
        }
    )
    return {
        "index": index,
        "task_probability": task_probability,
        "clip_probability": clip_probability,
        "task_prediction": task_prediction,
        "clip_prediction": clip_prediction,
        "checks": checks,
        "signal_path": signal_path,
        "signal_sha256": signal_hash,
        "lock_path": lock_path,
        "lock_sha256": lock_hash,
        "summary_path": summary_path,
        "summary_sha256": _sha256(summary_path),
        "expected_target_list_sha256": lock["target_list_sha256"],
    }


def _pct(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=np.float64).mean() * 100.0)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["oracle_metrics"]
    lines = [
        "# VisDA Conflict Candidate-Set Coverage Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The top-1 and top-2 task/CLIP candidate sets are constructed and",
        "locked before target labels are parsed. All coverage values below are",
        "explicit oracle diagnostics. No model or training is involved.",
        "",
        "## Coverage",
        "",
        f"- Top-1 union: `{metrics['top1_union_coverage_pct']:.6f}%`.",
        f"- Top-2 union: `{metrics['top2_union_coverage_pct']:.6f}%`.",
        f"- Top-2 gain: `{metrics['top2_minus_top1_coverage_pp']:.6f} pp`.",
        f"- Recovered top-1 misses: " f"`{metrics['recovered_top1_misses_pct']:.6f}%`.",
        f"- Mean top-2 union size: "
        f"`{summary['label_free_metrics']['top2_union_set_size_mean']:.6f}`.",
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend(
        [
            "",
            "Passing authorizes only method design. It does not authorize proxy",
            "or full VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing candidate-set input: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    class_names = _load_class_names(args.class_names)
    source = _load_locked_probabilities(args.input_dir)
    if not all(source["checks"].values()):
        failed = [name for name, passed in source["checks"].items() if not passed]
        raise RuntimeError(f"Locked source contract failed: {failed}")

    top1 = union_candidate_mask(
        source["task_probability"], source["clip_probability"], k=1
    )
    top2 = union_candidate_mask(
        source["task_probability"], source["clip_probability"], k=2
    )
    construction_checks = {
        "top1_union_has_exactly_two_classes": bool(np.all(top1["set_size"] == 2)),
        "top2_union_contains_top1_union": bool(
            np.all(~top1["union_mask"] | top2["union_mask"])
        ),
        "top2_union_size_between_two_and_four": bool(
            np.all((top2["set_size"] >= 2) & (top2["set_size"] <= 4))
        ),
        "top1_task_matches_saved_prediction": np.array_equal(
            top1["task_topk"][:, 0], source["task_prediction"]
        ),
        "top1_clip_matches_saved_prediction": np.array_equal(
            top1["clip_topk"][:, 0], source["clip_prediction"]
        ),
    }
    input_contract_valid = all({**source["checks"], **construction_checks}.values())
    if not input_contract_valid:
        failed = [name for name, passed in construction_checks.items() if not passed]
        raise RuntimeError(f"Candidate-set construction failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_candidate_set"
    candidate_path = args.output_dir / f"{stem}_label_free.npz"
    size_path = args.output_dir / f"{stem}_size_distribution.csv"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"

    np.savez_compressed(
        candidate_path,
        index=source["index"],
        task_top1=top1["task_topk"][:, 0],
        clip_top1=top1["clip_topk"][:, 0],
        task_top2=top2["task_topk"],
        clip_top2=top2["clip_topk"],
        top1_union_mask=top1["union_mask"],
        top2_union_mask=top2["union_mask"],
        top2_union_set_size=top2["set_size"],
    )
    size_rows = []
    for size in range(2, 5):
        mask = top2["set_size"] == size
        size_rows.append(
            {
                "candidate_set_size": size,
                "samples": int(mask.sum()),
                "fraction_pct": _pct(mask),
            }
        )
    _write_csv(size_path, size_rows)

    signal_lock = {
        "phase": "LABEL_FREE_CANDIDATE_SET_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "target_list_read_before_lock": False,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "top1_union": "task top-1 union CLIP top-1",
            "top2_union": "task top-2 union CLIP top-2",
            "tie_break": "stable class-index order",
            "target_label_thresholds": False,
            "class_specific_rules": False,
            "fitted_parameters": False,
        },
        "predeclared_gate": {
            "min_top2_union_coverage_pct": 90.0,
            "min_recovered_top1_misses_pct": 60.0,
            "min_per_class_top2_coverage_pct": 85.0,
            "min_car_truck_top2_coverage_pct": 90.0,
            "max_mean_candidate_set_size": 3.5,
        },
        "input_contract_checks": {
            **source["checks"],
            **construction_checks,
        },
        "inputs": {
            "source_signal_npz": {
                "path": str(source["signal_path"]),
                "sha256": source["signal_sha256"],
            },
            "source_signal_lock": {
                "path": str(source["lock_path"]),
                "sha256": source["lock_sha256"],
            },
            "source_summary": {
                "path": str(source["summary_path"]),
                "sha256": source["summary_sha256"],
            },
            "target_list_expected_opaque_sha256": source["expected_target_list_sha256"],
            "class_names_sha256": _sha256(args.class_names),
        },
        "outputs": {
            "candidate_set_npz": {
                "path": str(candidate_path),
                "sha256": _sha256(candidate_path),
            },
            "size_distribution_csv": {
                "path": str(size_path),
                "sha256": _sha256(size_path),
            },
        },
        "contract_sha256": {
            "src/utils/candidate_set_audit.py": _sha256(
                REPO_ROOT / "src/utils/candidate_set_audit.py"
            ),
            "tools/audit_visda_conflict_candidate_set.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(signal_lock, indent=2) + "\n")

    # Oracle diagnostic phase: target-list content is first read after lock.
    target_list_sha256 = _sha256(args.target_list)
    target_list_hash_matches = (
        target_list_sha256 == source["expected_target_list_sha256"]
    )
    if not target_list_hash_matches:
        raise RuntimeError("Target-list hash does not match the locked source signal")
    all_labels = _parse_labels_after_lock(args.target_list)
    labels = all_labels[source["index"]]
    top1_covered = candidate_coverage(top1["union_mask"], labels)
    top2_covered = candidate_coverage(top2["union_mask"], labels)
    task_top2_mask = np.zeros_like(top2["union_mask"])
    clip_top2_mask = np.zeros_like(top2["union_mask"])
    row = np.arange(labels.size)[:, None]
    task_top2_mask[row, top2["task_topk"]] = True
    clip_top2_mask[row, top2["clip_topk"]] = True
    task_top2_covered = candidate_coverage(task_top2_mask, labels)
    clip_top2_covered = candidate_coverage(clip_top2_mask, labels)

    top1_missed = ~top1_covered
    recovered = top1_missed & top2_covered
    recovered_top1_misses_pct = (
        float(recovered.sum() / top1_missed.sum() * 100.0) if top1_missed.any() else 0.0
    )
    task_rank2_adds_true = top1_missed & (top2["task_topk"][:, 1] == labels)
    clip_rank2_adds_true = top1_missed & (top2["clip_topk"][:, 1] == labels)

    class_rows = []
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        class_missed = mask & top1_missed
        class_recovered = mask & recovered
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "conflict_samples": int(mask.sum()),
                "top1_union_coverage_pct": _pct(top1_covered[mask]),
                "top2_union_coverage_pct": _pct(top2_covered[mask]),
                "top2_minus_top1_coverage_pp": _pct(top2_covered[mask])
                - _pct(top1_covered[mask]),
                "recovered_top1_misses_pct": (
                    float(class_recovered.sum() / class_missed.sum() * 100.0)
                    if class_missed.any()
                    else 0.0
                ),
                "task_top2_coverage_pct": _pct(task_top2_covered[mask]),
                "clip_top2_coverage_pct": _pct(clip_top2_covered[mask]),
                "task_rank2_additions": int((mask & task_rank2_adds_true).sum()),
                "clip_rank2_additions": int((mask & clip_rank2_adds_true).sum()),
                "unrecovered_after_top2": int((mask & ~top2_covered).sum()),
            }
        )
    _write_csv(class_path, class_rows)

    oracle_rows = []
    for position, target_index in enumerate(source["index"]):
        label = int(labels[position])
        oracle_rows.append(
            {
                "index": int(target_index),
                "label": label,
                "label_name": class_names[label],
                "task_top1": int(top1["task_topk"][position, 0]),
                "clip_top1": int(top1["clip_topk"][position, 0]),
                "task_top2": int(top2["task_topk"][position, 1]),
                "clip_top2": int(top2["clip_topk"][position, 1]),
                "top1_union_covers_label": bool(top1_covered[position]),
                "top2_union_covers_label": bool(top2_covered[position]),
                "recovered_by_top2": bool(recovered[position]),
                "task_rank2_adds_label": bool(task_rank2_adds_true[position]),
                "clip_rank2_adds_label": bool(clip_rank2_adds_true[position]),
                "top2_union_set_size": int(top2["set_size"][position]),
            }
        )
    _write_csv(oracle_path, oracle_rows)

    by_name = {row["class"]: row for row in class_rows}
    minimum_class_coverage = min(row["top2_union_coverage_pct"] for row in class_rows)
    gate = evaluate_candidate_set_gate(
        input_contract_valid=input_contract_valid and target_list_hash_matches,
        top2_coverage_pct=_pct(top2_covered),
        recovered_top1_misses_pct=recovered_top1_misses_pct,
        minimum_class_coverage_pct=minimum_class_coverage,
        car_coverage_pct=by_name["car"]["top2_union_coverage_pct"],
        truck_coverage_pct=by_name["truck"]["top2_union_coverage_pct"],
        mean_set_size=float(top2["set_size"].mean()),
    )
    summary = {
        "dataset": "VISDA-C",
        "task": "train->validation",
        "seed": args.seed,
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "candidate_contract": signal_lock["candidate_contract"],
        "input_contract": {
            "passed": input_contract_valid and target_list_hash_matches,
            "checks": {
                **source["checks"],
                **construction_checks,
                "target_list_hash_matches_after_lock": target_list_hash_matches,
            },
        },
        "label_free_metrics": {
            "conflict_samples": EXPECTED_CONFLICT_SAMPLES,
            "top1_union_set_size_mean": float(top1["set_size"].mean()),
            "top2_union_set_size_mean": float(top2["set_size"].mean()),
            "top2_union_set_size_median": float(np.median(top2["set_size"])),
            "top2_union_set_size_max": int(top2["set_size"].max()),
            "top2_union_size_distribution": size_rows,
        },
        "oracle_metrics": {
            "conflict_samples": EXPECTED_CONFLICT_SAMPLES,
            "top1_union_coverage_pct": _pct(top1_covered),
            "top2_union_coverage_pct": _pct(top2_covered),
            "top2_minus_top1_coverage_pp": _pct(top2_covered) - _pct(top1_covered),
            "recovered_top1_misses": int(recovered.sum()),
            "top1_misses": int(top1_missed.sum()),
            "recovered_top1_misses_pct": recovered_top1_misses_pct,
            "task_top2_coverage_pct": _pct(task_top2_covered),
            "clip_top2_coverage_pct": _pct(clip_top2_covered),
            "task_rank2_additions": int(task_rank2_adds_true.sum()),
            "clip_rank2_additions": int(clip_rank2_adds_true.sum()),
            "both_rank2_add_same_true_label": int(
                (task_rank2_adds_true & clip_rank2_adds_true).sum()
            ),
            "unrecovered_after_top2": int((~top2_covered).sum()),
            "minimum_class_top2_coverage_pct": minimum_class_coverage,
            "car_top2_coverage_pct": by_name["car"]["top2_union_coverage_pct"],
            "truck_top2_coverage_pct": by_name["truck"]["top2_union_coverage_pct"],
            "classwise": class_rows,
        },
        "gate": gate,
        "decision": gate["decision"],
        "next": (
            "eligible to design one set-valued conflict-supervision preflight; "
            "no training is authorized"
            if gate["decision"] == "PASS_CANDIDATE_SET_PREFLIGHT"
            else "reject top-2 candidate-set completion; do not train"
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
        ("label-free candidate set", candidate_path),
        ("label-free size distribution", size_path),
        ("signal lock", lock_path),
        ("oracle diagnostic", oracle_path),
        ("classwise oracle diagnostic", class_path),
        ("summary", summary_path),
    ):
        print(f"Wrote {label}: {output}")
    print(json.dumps({"decision": summary["decision"], "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
