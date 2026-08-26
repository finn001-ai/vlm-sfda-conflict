#!/usr/bin/env python
"""CPU-only offline audit of pairwise attribute probability redistribution.

Phase 1 reads only the previously locked label-free NPZ, loads CLIP ViT-B/32
solely to recover its frozen learned logit scale, constructs the soft target,
and locks it.  Phase 2 then reads VisDA target labels for explicitly marked
oracle diagnostics.  No target image is loaded, no model forward is called,
and no optimizer, backward pass, training loop, or parameter update exists.
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
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import clip  # noqa: E402
from src.utils.attribute_mass_audit import (  # noqa: E402
    evaluate_attribute_mass_gate,
    paired_mean_bootstrap_ci,
    redistribute_pairwise_attribute_mass,
)


EXPECTED_CONFLICT_SAMPLES = 28_255
EXPECTED_TARGET_SAMPLES = 55_388
EXPECTED_CLASSES = 12
BOOTSTRAP_REPEATS = 2_000
MAX_CLASS_MASS_SHIFT_PP = 1.0
DEFAULT_INPUT_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_pairwise_attribute_audit_seed2020/"
    "pairwise_attribute_audit"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_attribute_mass_audit_seed2020/attribute_mass_audit"
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
    parser.add_argument("--clip-architecture", default="ViT-B/32")
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
    with path.open() as handle:
        names = [line.strip().replace("_", " ") for line in handle if line.strip()]
    if len(names) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} VisDA classes, found {len(names)}"
        )
    return names


def _parse_labels_after_lock(path: Path) -> np.ndarray:
    labels = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
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
        raise ValueError("VisDA target label is outside the class range")
    return result


def _load_locked_inputs(input_dir: Path) -> dict[str, Any]:
    stem = "visda_conflict_pairwise_attribute"
    signal_npz_path = input_dir / f"{stem}_signals.npz"
    signal_lock_path = input_dir / f"{stem}_signal_lock.json"
    summary_path = input_dir / f"{stem}_summary.json"
    for path in (signal_npz_path, signal_lock_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing mass-audit input: {path}")

    signal_lock = json.loads(signal_lock_path.read_text())
    source_summary = json.loads(summary_path.read_text())
    signal_npz_sha256 = _sha256(signal_npz_path)
    signal_lock_sha256 = _sha256(signal_lock_path)
    expected_npz_hash = signal_lock.get("signal_npz", {}).get("sha256")
    expected_target_list_hash = signal_lock.get("target_list_sha256")
    checks = {
        "source_baseline_reproduced": bool(
            source_summary.get("baseline_reproduction", {}).get("passed")
        ),
        "source_labels_locked_before_oracle": bool(
            source_summary.get("labels_used_only_after_signal_lock")
        ),
        "source_signal_lock_hash_matches_summary": (
            signal_lock_sha256 == source_summary.get("signal_lock_sha256")
        ),
        "source_npz_hash_matches_lock": signal_npz_sha256 == expected_npz_hash,
        "expected_target_list_hash_present": (
            isinstance(expected_target_list_hash, str)
            and len(expected_target_list_hash) == 64
        ),
        "source_npz_declared_label_free": not bool(
            signal_lock.get("contains_target_labels", True)
        ),
    }

    with np.load(signal_npz_path) as archive:
        forbidden_keys = {"label", "labels", "target_label", "path", "paths"}
        checks["source_npz_has_no_label_or_path_key"] = not bool(
            forbidden_keys.intersection(archive.files)
        )
        required = {
            "index",
            "task_probability",
            "clip_probability",
            "task_prediction",
            "clip_prediction",
            "attribute_margin",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Source NPZ is missing keys: {sorted(missing)}")
        arrays = {key: archive[key].copy() for key in required}

    index = np.asarray(arrays["index"], dtype=np.int64)
    task_probability = np.asarray(arrays["task_probability"], dtype=np.float64)
    clip_probability = np.asarray(arrays["clip_probability"], dtype=np.float64)
    task_prediction = np.asarray(arrays["task_prediction"], dtype=np.int64)
    clip_prediction = np.asarray(arrays["clip_prediction"], dtype=np.int64)
    attribute_margin = np.asarray(arrays["attribute_margin"], dtype=np.float64)
    checks.update(
        {
            "expected_conflict_count": index.shape
            == (EXPECTED_CONFLICT_SAMPLES,),
            "indices_strictly_increasing": bool(np.all(np.diff(index) > 0)),
            "indices_inside_full_target": bool(
                np.all(index >= 0) and np.all(index < EXPECTED_TARGET_SAMPLES)
            ),
            "probability_shapes_valid": (
                task_probability.shape
                == clip_probability.shape
                == (EXPECTED_CONFLICT_SAMPLES, EXPECTED_CLASSES)
            ),
            "prediction_shapes_valid": (
                task_prediction.shape
                == clip_prediction.shape
                == (EXPECTED_CONFLICT_SAMPLES,)
            ),
            "all_rows_are_conflicts": bool(
                np.all(task_prediction != clip_prediction)
            ),
            "attribute_shape_valid": attribute_margin.shape
            == (EXPECTED_CONFLICT_SAMPLES, 2, 4),
        }
    )
    return {
        "index": index,
        "task_probability": task_probability,
        "clip_probability": clip_probability,
        "task_prediction": task_prediction,
        "clip_prediction": clip_prediction,
        "attribute_margin": attribute_margin,
        "checks": checks,
        "signal_npz_path": signal_npz_path,
        "signal_npz_sha256": signal_npz_sha256,
        "signal_lock_path": signal_lock_path,
        "signal_lock_sha256": signal_lock_sha256,
        "source_summary_path": summary_path,
        "source_summary_sha256": _sha256(summary_path),
        "expected_target_list_sha256": expected_target_list_hash,
    }


def _rms_probability(
    task_probability: np.ndarray, clip_probability: np.ndarray
) -> np.ndarray:
    result = np.sqrt(
        (np.square(task_probability) + np.square(clip_probability)) / 2.0
    )
    return result / result.sum(axis=1, keepdims=True)


def _metric_arrays(probability: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    row = np.arange(labels.size)
    true_probability = probability[row, labels]
    nll = -np.log(np.clip(true_probability, 1e-12, 1.0))
    squared_norm = np.square(probability).sum(axis=1)
    brier = squared_norm - 2.0 * true_probability + 1.0
    return {
        "true_probability": true_probability,
        "nll": nll,
        "brier": brier,
    }


def _compare_metrics(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    *,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    improvements = {
        "nll": baseline["nll"] - candidate["nll"],
        "brier": baseline["brier"] - candidate["brier"],
        "true_probability": (
            candidate["true_probability"] - baseline["true_probability"]
        ),
    }
    for offset, name in enumerate(("nll", "brier", "true_probability")):
        ci = paired_mean_bootstrap_ci(
            improvements[name],
            repeats=BOOTSTRAP_REPEATS,
            seed=seed + offset,
        )
        result[name] = {
            "candidate_mean": float(candidate[name].mean()),
            "baseline_mean": float(baseline[name].mean()),
            "improvement": float(improvements[name].mean()),
            "paired_bootstrap_95_ci": list(ci),
            "positive_means_candidate_better": True,
        }
    result["checks"] = {
        f"{name}_ci_lower_positive": result[name][
            "paired_bootstrap_95_ci"
        ][0]
        > 0.0
        for name in ("nll", "brier", "true_probability")
    }
    return result


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    fixed = summary["oracle_metrics"]["comparisons"]["fixed_clip"]
    safety = summary["safety_contract"]
    lines = [
        "# VisDA Pairwise Attribute Mass Offline Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The audit reads a previously locked label-free NPZ and uses CPU only.",
        "It loads no target image and performs no model forward, backward,",
        "optimizer step, adaptation, or training.",
        "",
        "## Candidate versus fixed CLIP",
        "",
        f"- NLL improvement: `{fixed['nll']['improvement']:.8f}`; 95% CI "
        f"`{fixed['nll']['paired_bootstrap_95_ci']}`.",
        f"- Brier improvement: `{fixed['brier']['improvement']:.8f}`; 95% CI "
        f"`{fixed['brier']['paired_bootstrap_95_ci']}`.",
        f"- True-probability gain: "
        f"`{fixed['true_probability']['improvement']:.8f}`; 95% CI "
        f"`{fixed['true_probability']['paired_bootstrap_95_ci']}`.",
        "",
        "## Class safety and label-free mass shift",
        "",
        f"- Car NLL improvement: `{summary['class_safety']['car']['nll_improvement']:.8f}`.",
        f"- Truck NLL improvement: `{summary['class_safety']['truck']['nll_improvement']:.8f}`.",
        f"- Maximum absolute class-mass shift: "
        f"`{summary['label_free_mass_diagnostic']['max_abs_shift_pp']:.8f} pp`.",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend(
        [
            "",
            f"- Target images loaded: `{safety['target_images_loaded']}`.",
            f"- Model forwards: `{safety['model_forward_calls']}`.",
            f"- Backward calls: `{safety['backward_calls']}`.",
            "",
            "Passing authorizes only a separately approved matched proxy run;",
            "it does not authorize or start proxy or full VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    if args.clip_architecture != "ViT-B/32":
        raise ValueError("Attribute mass audit is locked to CLIP ViT-B/32")
    for path in (args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing mass-audit input: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    class_names = _load_class_names(args.class_names)
    source = _load_locked_inputs(args.input_dir)
    input_contract_valid = all(source["checks"].values())
    if not input_contract_valid:
        failed = [name for name, passed in source["checks"].items() if not passed]
        raise RuntimeError(f"Locked input contract failed: {failed}")

    with torch.no_grad():
        clip_model, _preprocess, _ = clip.load(
            args.clip_architecture, device="cpu", jit=False
        )
        clip_logit_scale_log = float(clip_model.logit_scale.detach().cpu().item())
        clip_logit_scale = float(clip_model.logit_scale.exp().detach().cpu().item())
    del clip_model, _preprocess

    redistribution = redistribute_pairwise_attribute_mass(
        source["clip_probability"],
        source["task_prediction"],
        source["clip_prediction"],
        source["attribute_margin"],
        clip_logit_scale=clip_logit_scale,
    )
    candidate_probability = redistribution["probability"]
    class_mass_rows = []
    mass_shift_pp = (
        candidate_probability - source["clip_probability"]
    ).mean(axis=0) * 100.0
    for class_index, class_name in enumerate(class_names):
        class_mass_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "fixed_clip_expected_mass_pct": float(
                    source["clip_probability"][:, class_index].mean() * 100.0
                ),
                "candidate_expected_mass_pct": float(
                    candidate_probability[:, class_index].mean() * 100.0
                ),
                "delta_pp": float(mass_shift_pp[class_index]),
            }
        )
    max_abs_mass_shift_pp = float(np.abs(mass_shift_pp).max())

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_attribute_mass"
    target_npz_path = args.output_dir / f"{stem}_target.npz"
    mass_csv_path = args.output_dir / f"{stem}_class_mass.csv"
    lock_path = args.output_dir / f"{stem}_target_lock.json"
    oracle_csv_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_csv_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"

    np.savez_compressed(
        target_npz_path,
        index=source["index"],
        task_prediction=source["task_prediction"],
        clip_prediction=source["clip_prediction"],
        attribute_mean_margin=redistribution["attribute_mean_margin"],
        task_fraction=redistribution["task_fraction"],
        pair_mass=redistribution["pair_mass"],
        fixed_clip_probability=source["clip_probability"],
        candidate_probability=candidate_probability,
    )
    _write_csv(mass_csv_path, class_mass_rows)
    target_lock = {
        "phase": "LABEL_FREE_ATTRIBUTE_MASS_TARGET_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "oracle_labels_parsed_after_this_manifest": True,
        "source_signal_csv_read": False,
        "target_images_loaded": False,
        "clip_architecture": args.clip_architecture,
        "clip_checkpoint_use": "frozen logit_scale scalar only",
        "clip_logit_scale_log": clip_logit_scale_log,
        "clip_logit_scale_exp": clip_logit_scale,
        "formula": {
            "pair_mass": "M = p_clip(task_top1) + p_clip(clip_top1)",
            "task_fraction": "r = sigmoid(exp(logit_scale) * mean(attribute_margin))",
            "task_target": "q(task_top1) = M * r",
            "clip_target": "q(clip_top1) = M * (1-r)",
            "other_classes": "q(c) = p_clip(c)",
            "target_label_thresholds": False,
        },
        "input_contract_checks": source["checks"],
        "inputs": {
            "source_signal_npz": {
                "path": str(source["signal_npz_path"]),
                "sha256": source["signal_npz_sha256"],
            },
            "source_signal_lock": {
                "path": str(source["signal_lock_path"]),
                "sha256": source["signal_lock_sha256"],
            },
            "source_summary": {
                "path": str(source["source_summary_path"]),
                "sha256": source["source_summary_sha256"],
            },
            "target_list_expected_opaque_hash": source[
                "expected_target_list_sha256"
            ],
            "class_names_sha256": _sha256(args.class_names),
        },
        "outputs": {
            "target_npz": {
                "path": str(target_npz_path),
                "sha256": _sha256(target_npz_path),
            },
            "class_mass_csv": {
                "path": str(mass_csv_path),
                "sha256": _sha256(mass_csv_path),
            },
        },
        "contract_sha256": {
            "src/utils/attribute_mass_audit.py": _sha256(
                REPO_ROOT / "src/utils/attribute_mass_audit.py"
            ),
            "tools/audit_visda_conflict_attribute_mass.py": _sha256(
                Path(__file__).resolve()
            ),
            "src/methods/oh/plmatch.py": _sha256(
                REPO_ROOT / "src/methods/oh/plmatch.py"
            ),
        },
    }
    lock_path.write_text(json.dumps(target_lock, indent=2) + "\n")

    # Phase 2: the label-bearing target list is first accessed only after lock.
    target_list_sha256 = _sha256(args.target_list)
    target_list_hash_matches = (
        target_list_sha256 == source["expected_target_list_sha256"]
    )
    if not target_list_hash_matches:
        raise RuntimeError("Target-list hash does not match the locked source audit")
    all_labels = _parse_labels_after_lock(args.target_list)
    labels = all_labels[source["index"]]
    if labels.shape != (EXPECTED_CONFLICT_SAMPLES,):
        raise RuntimeError("Conflict labels do not match locked sample count")

    arithmetic_probability = (
        source["task_probability"] + source["clip_probability"]
    ) / 2.0
    rms_probability = _rms_probability(
        source["task_probability"], source["clip_probability"]
    )
    probabilities = {
        "candidate": candidate_probability,
        "fixed_clip": source["clip_probability"],
        "arithmetic": arithmetic_probability,
        "rms": rms_probability,
    }
    metrics = {
        name: _metric_arrays(probability, labels)
        for name, probability in probabilities.items()
    }
    comparisons = {
        name: _compare_metrics(
            metrics["candidate"], metrics[name], seed=args.seed + 10 * offset
        )
        for offset, name in enumerate(("fixed_clip", "arithmetic", "rms"))
    }

    row = np.arange(labels.size)
    predictions = {
        name: probability.argmax(axis=1)
        for name, probability in probabilities.items()
    }
    for name in ("fixed_clip", "arithmetic", "rms"):
        candidate_correct = predictions["candidate"] == labels
        baseline_correct = predictions[name] == labels
        accuracy_difference = candidate_correct.astype(np.float64) - baseline_correct
        accuracy_ci = paired_mean_bootstrap_ci(
            accuracy_difference,
            repeats=BOOTSTRAP_REPEATS,
            seed=args.seed + 100 + 10 * ("fixed_clip", "arithmetic", "rms").index(name),
        )
        comparisons[name]["accuracy"] = {
            "candidate_accuracy_pct": float(candidate_correct.mean() * 100.0),
            "baseline_accuracy_pct": float(baseline_correct.mean() * 100.0),
            "gain_pp": float(accuracy_difference.mean() * 100.0),
            "paired_bootstrap_95_ci_pp": [
                float(accuracy_ci[0] * 100.0),
                float(accuracy_ci[1] * 100.0),
            ],
        }

    class_rows = []
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        candidate_nll = metrics["candidate"]["nll"][mask]
        clip_nll = metrics["fixed_clip"]["nll"][mask]
        candidate_brier = metrics["candidate"]["brier"][mask]
        clip_brier = metrics["fixed_clip"]["brier"][mask]
        candidate_true = metrics["candidate"]["true_probability"][mask]
        clip_true = metrics["fixed_clip"]["true_probability"][mask]
        candidate_correct = predictions["candidate"][mask] == labels[mask]
        clip_correct = predictions["fixed_clip"][mask] == labels[mask]
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "conflict_samples": int(mask.sum()),
                "candidate_nll": float(candidate_nll.mean()),
                "fixed_clip_nll": float(clip_nll.mean()),
                "nll_improvement": float((clip_nll - candidate_nll).mean()),
                "candidate_brier": float(candidate_brier.mean()),
                "fixed_clip_brier": float(clip_brier.mean()),
                "brier_improvement": float(
                    (clip_brier - candidate_brier).mean()
                ),
                "candidate_true_probability": float(candidate_true.mean()),
                "fixed_clip_true_probability": float(clip_true.mean()),
                "true_probability_gain": float(
                    (candidate_true - clip_true).mean()
                ),
                "candidate_accuracy_pct": float(candidate_correct.mean() * 100.0),
                "fixed_clip_accuracy_pct": float(clip_correct.mean() * 100.0),
                "accuracy_gain_pp": float(
                    (candidate_correct.mean() - clip_correct.mean()) * 100.0
                ),
            }
        )
    _write_csv(class_csv_path, class_rows)

    oracle_rows = []
    for local_index, target_index in enumerate(source["index"]):
        oracle_rows.append(
            {
                "index": int(target_index),
                "label": int(labels[local_index]),
                "label_name": class_names[int(labels[local_index])],
                "task_top1": int(source["task_prediction"][local_index]),
                "clip_top1": int(source["clip_prediction"][local_index]),
                "attribute_mean_margin": float(
                    redistribution["attribute_mean_margin"][local_index]
                ),
                "task_fraction": float(
                    redistribution["task_fraction"][local_index]
                ),
                "pair_mass": float(redistribution["pair_mass"][local_index]),
                "candidate_true_probability": float(
                    metrics["candidate"]["true_probability"][local_index]
                ),
                "fixed_clip_true_probability": float(
                    metrics["fixed_clip"]["true_probability"][local_index]
                ),
                "arithmetic_true_probability": float(
                    metrics["arithmetic"]["true_probability"][local_index]
                ),
                "rms_true_probability": float(
                    metrics["rms"]["true_probability"][local_index]
                ),
                "candidate_nll": float(metrics["candidate"]["nll"][local_index]),
                "fixed_clip_nll": float(metrics["fixed_clip"]["nll"][local_index]),
                "candidate_brier": float(
                    metrics["candidate"]["brier"][local_index]
                ),
                "fixed_clip_brier": float(
                    metrics["fixed_clip"]["brier"][local_index]
                ),
            }
        )
    _write_csv(oracle_csv_path, oracle_rows)

    by_name = {row_data["class"]: row_data for row_data in class_rows}
    comparison_checks = {
        name: comparisons[name]["checks"]
        for name in ("fixed_clip", "arithmetic", "rms")
    }
    full_input_contract_valid = input_contract_valid and target_list_hash_matches
    gate = evaluate_attribute_mass_gate(
        input_contract_valid=full_input_contract_valid,
        comparison_checks=comparison_checks,
        car_nll_improvement=by_name["car"]["nll_improvement"],
        truck_nll_improvement=by_name["truck"]["nll_improvement"],
        car_brier_improvement=by_name["car"]["brier_improvement"],
        truck_brier_improvement=by_name["truck"]["brier_improvement"],
        max_abs_class_mass_shift_pp=max_abs_mass_shift_pp,
        max_allowed_class_mass_shift_pp=MAX_CLASS_MASS_SHIFT_PP,
    )
    summary = {
        "dataset": "VISDA-C",
        "task": "train->validation",
        "seed": args.seed,
        "oracle_diagnostic": True,
        "labels_used_only_after_target_lock": True,
        "target_lock": str(lock_path),
        "target_lock_sha256": _sha256(lock_path),
        "candidate_contract": target_lock["formula"],
        "compute_contract": {
            "conflict_samples": EXPECTED_CONFLICT_SAMPLES,
            "target_image_forwards": 0,
            "clip_image_forwards": 0,
            "clip_text_forwards": 0,
            "clip_checkpoint_use": "logit_scale scalar only",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
        },
        "input_contract": {
            "passed": full_input_contract_valid,
            "checks": {
                **source["checks"],
                "target_list_hash_matches_lock_after_target_lock": (
                    target_list_hash_matches
                ),
            },
        },
        "label_free_mass_diagnostic": {
            "max_abs_shift_pp": max_abs_mass_shift_pp,
            "max_abs_shift_class": class_names[int(np.abs(mass_shift_pp).argmax())],
            "class_rows": class_mass_rows,
        },
        "oracle_metrics": {
            "samples": EXPECTED_CONFLICT_SAMPLES,
            "comparisons": comparisons,
        },
        "classwise_oracle_diagnostic": class_rows,
        "class_safety": {
            "car": by_name["car"],
            "truck": by_name["truck"],
        },
        "gate": gate,
        "decision": gate["decision"],
        "next": (
            "eligible for separately approved matched proxy25 only; do not run full VisDA"
            if gate["decision"] == "PASS_OFFLINE_GATE"
            else "stop; do not run proxy or full VisDA training"
        ),
        "safety_contract": {
            "target_images_loaded": False,
            "model_forward_calls": 0,
            "optimizer_constructed": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "model_parameters_updated": False,
            "training_code_modified": False,
            "training_authorized": False,
        },
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)

    print(f"Wrote label-free soft target: {target_npz_path}")
    print(f"Wrote label-free class-mass audit: {mass_csv_path}")
    print(f"Locked target before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_csv_path}")
    print(f"Wrote classwise oracle diagnostic: {class_csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps({"decision": summary["decision"], "checks": gate["checks"]}, indent=2))


if __name__ == "__main__":
    main()
