#!/usr/bin/env python
"""CPU-only audit of entropy-anchored pairwise attribute evidence.

Phase 1 reads only a previously locked label-free signal NPZ plus the
label-free attribute-mass lock.  It constructs and locks a candidate target.
Phase 2 reads target labels solely for an explicitly marked oracle diagnostic.
No image, checkpoint, forward, backward, optimizer, or training is involved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.attribute_mass_audit import (  # noqa: E402
    entropy_anchored_attribute_mass,
    evaluate_attribute_reliability_gate,
    paired_mean_bootstrap_ci,
)
from tools.audit_visda_conflict_attribute_mass import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    EXPECTED_CONFLICT_SAMPLES,
    EXPECTED_TARGET_SAMPLES,
    _compare_metrics,
    _load_class_names,
    _load_locked_inputs,
    _metric_arrays,
    _parse_labels_after_lock,
    _rms_probability,
    _sha256,
    _write_csv,
)


EXPECTED_CLIP_ARCHITECTURE = "ViT-B/32"
MAX_CLASS_MASS_SHIFT_PP = 1.0
DEFAULT_INPUT_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_pairwise_attribute_audit_seed2020/"
    "pairwise_attribute_audit"
)
DEFAULT_MASS_LOCK = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_attribute_mass_audit_seed2020/attribute_mass_audit/"
    "visda_conflict_attribute_mass_target_lock.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_attribute_reliability_audit_seed2020/"
    "attribute_reliability_audit"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--mass-lock", type=Path, default=DEFAULT_MASS_LOCK)
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


def _load_label_free_scale_lock(
    path: Path, *, source_signal_npz_sha256: str
) -> tuple[dict[str, Any], dict[str, bool], float]:
    lock = json.loads(path.read_text())
    scale = lock.get("clip_logit_scale_exp")
    checks = {
        "mass_lock_is_label_free": not bool(lock.get("contains_target_labels", True)),
        "mass_lock_contains_no_paths": not bool(
            lock.get("contains_target_paths", True)
        ),
        "mass_lock_precedes_oracle_labels": bool(
            lock.get("oracle_labels_parsed_after_this_manifest")
        ),
        "mass_lock_uses_expected_clip": (
            lock.get("clip_architecture") == EXPECTED_CLIP_ARCHITECTURE
        ),
        "mass_lock_source_npz_matches": (
            lock.get("inputs", {}).get("source_signal_npz", {}).get("sha256")
            == source_signal_npz_sha256
        ),
        "mass_lock_has_finite_positive_scale": (
            isinstance(scale, (int, float))
            and math.isfinite(float(scale))
            and float(scale) > 0.0
        ),
        "mass_lock_has_no_label_fitted_threshold": not bool(
            lock.get("formula", {}).get("target_label_thresholds", True)
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Label-free mass-lock contract failed: {failed}")
    return lock, checks, float(scale)


def _accuracy_comparison(
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    candidate_correct = candidate_prediction == labels
    baseline_correct = baseline_prediction == labels
    difference = candidate_correct.astype(np.float64) - baseline_correct
    ci = paired_mean_bootstrap_ci(
        difference,
        repeats=BOOTSTRAP_REPEATS,
        seed=seed,
    )
    return {
        "candidate_accuracy_pct": float(candidate_correct.mean() * 100.0),
        "baseline_accuracy_pct": float(baseline_correct.mean() * 100.0),
        "gain_pp": float(difference.mean() * 100.0),
        "net_corrections": int(difference.sum()),
        "paired_bootstrap_95_ci_pp": [float(ci[0] * 100.0), float(ci[1] * 100.0)],
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    fixed = summary["oracle_metrics"]["comparisons"]["fixed_clip"]
    car = summary["class_safety"]["car"]
    truck = summary["class_safety"]["truck"]
    lines = [
        "# VisDA Entropy-Anchored Attribute Reliability Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "This audit reuses the locked pairwise attribute evidence. Its fixed,",
        "label-free weight is `H(CLIP pair) * (1 - H(task pair))`.",
        "No threshold, class exception, target image, model forward, or training is used.",
        "",
        "## Candidate versus fixed CLIP (oracle diagnostic)",
        "",
        f"- Accuracy gain: `{fixed['accuracy']['gain_pp']:.8f} pp`; 95% CI "
        f"`{fixed['accuracy']['paired_bootstrap_95_ci_pp']}`; net corrections "
        f"`{fixed['accuracy']['net_corrections']}`.",
        f"- NLL improvement: `{fixed['nll']['improvement']:.8f}`; 95% CI "
        f"`{fixed['nll']['paired_bootstrap_95_ci']}`.",
        f"- Brier improvement: `{fixed['brier']['improvement']:.8f}`; 95% CI "
        f"`{fixed['brier']['paired_bootstrap_95_ci']}`.",
        f"- True-probability gain: `{fixed['true_probability']['improvement']:.8f}`; "
        f"95% CI `{fixed['true_probability']['paired_bootstrap_95_ci']}`.",
        "",
        "## Exchange-risk checks (oracle diagnostic)",
        "",
        f"- Car accuracy gain: `{car['accuracy_gain_pp']:.8f} pp`; net "
        f"`{car['net_corrections']}`.",
        f"- Truck accuracy gain: `{truck['accuracy_gain_pp']:.8f} pp`; net "
        f"`{truck['net_corrections']}`.",
        f"- Non-car net corrections: `{summary['class_safety']['noncar_net_corrections']}`.",
        f"- Maximum label-free class-mass shift: "
        f"`{summary['label_free_diagnostic']['max_abs_class_mass_shift_pp']:.8f} pp`.",
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
            "Passing this audit authorizes no training. A matched proxy run would",
            "still require separate approval before any GPU use.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (args.mass_lock, args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing reliability-audit input: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    class_names = _load_class_names(args.class_names)
    source = _load_locked_inputs(args.input_dir)
    source_contract_valid = all(source["checks"].values())
    if not source_contract_valid:
        failed = [name for name, passed in source["checks"].items() if not passed]
        raise RuntimeError(f"Locked source contract failed: {failed}")
    mass_lock, mass_lock_checks, clip_logit_scale = _load_label_free_scale_lock(
        args.mass_lock,
        source_signal_npz_sha256=source["signal_npz_sha256"],
    )

    candidate = entropy_anchored_attribute_mass(
        source["task_probability"],
        source["clip_probability"],
        source["task_prediction"],
        source["clip_prediction"],
        source["attribute_margin"],
        clip_logit_scale=clip_logit_scale,
    )
    candidate_probability = candidate["probability"]
    mass_shift_pp = (candidate_probability - source["clip_probability"]).mean(
        axis=0
    ) * 100.0
    class_mass_rows = [
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
        for class_index, class_name in enumerate(class_names)
    ]
    max_abs_mass_shift_pp = float(np.abs(mass_shift_pp).max())

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_attribute_reliability"
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
        fixed_clip_probability=source["clip_probability"],
        candidate_probability=candidate_probability,
        pair_mass=candidate["pair_mass"],
        attribute_mean_margin=candidate["attribute_mean_margin"],
        attribute_fraction=candidate["attribute_fraction"],
        clip_pair_fraction=candidate["clip_pair_fraction"],
        task_pair_fraction=candidate["task_pair_fraction"],
        clip_pair_entropy=candidate["clip_pair_entropy"],
        task_pair_entropy=candidate["task_pair_entropy"],
        attribute_weight=candidate["attribute_weight"],
        anchored_fraction=candidate["anchored_fraction"],
    )
    _write_csv(mass_csv_path, class_mass_rows)
    target_lock = {
        "phase": "LABEL_FREE_ATTRIBUTE_RELIABILITY_TARGET_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "oracle_labels_parsed_after_this_manifest": True,
        "source_signal_csv_read": False,
        "target_images_loaded": False,
        "model_checkpoints_loaded": False,
        "model_forward_calls": 0,
        "clip_architecture": EXPECTED_CLIP_ARCHITECTURE,
        "clip_logit_scale_source": "previously locked label-free mass manifest",
        "clip_logit_scale_exp": clip_logit_scale,
        "formula": {
            "clip_pair_fraction": "c = p_clip(task)/(p_clip(task)+p_clip(clip))",
            "task_pair_fraction": "s = p_task(task)/(p_task(task)+p_task(clip))",
            "binary_entropy": "H(x) = -[x ln x + (1-x) ln(1-x)]/ln(2)",
            "attribute_weight": "w = H(c) * (1-H(s))",
            "attribute_log_odds": "a = exp(logit_scale)*mean(attribute_margin)",
            "anchored_log_odds": "z = (1-w)*logit(c) + w*a",
            "pair_redistribution": "q(task)=M*sigmoid(z); q(clip)=M-q(task)",
            "other_classes": "q(k)=p_clip(k)",
            "tunable_coefficients": False,
            "class_specific_rules": False,
            "target_label_thresholds": False,
        },
        "input_contract_checks": {
            **source["checks"],
            **mass_lock_checks,
        },
        "inputs": {
            "source_signal_npz": {
                "path": str(source["signal_npz_path"]),
                "sha256": source["signal_npz_sha256"],
            },
            "source_signal_lock": {
                "path": str(source["signal_lock_path"]),
                "sha256": source["signal_lock_sha256"],
            },
            "attribute_mass_lock": {
                "path": str(args.mass_lock),
                "sha256": _sha256(args.mass_lock),
            },
            "target_list_expected_opaque_hash": source["expected_target_list_sha256"],
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
            "tools/audit_visda_conflict_attribute_reliability.py": _sha256(
                Path(__file__).resolve()
            ),
            "src/methods/oh/plmatch.py": _sha256(
                REPO_ROOT / "src/methods/oh/plmatch.py"
            ),
        },
    }
    lock_path.write_text(json.dumps(target_lock, indent=2) + "\n")

    # Oracle diagnostic phase: no target-list content is accessed before lock.
    target_list_sha256 = _sha256(args.target_list)
    target_list_hash_matches = (
        target_list_sha256 == source["expected_target_list_sha256"]
    )
    if not target_list_hash_matches:
        raise RuntimeError("Target-list hash does not match the locked source audit")
    all_labels = _parse_labels_after_lock(args.target_list)
    labels = all_labels[source["index"]]
    if labels.shape != (EXPECTED_CONFLICT_SAMPLES,):
        raise RuntimeError("Conflict labels do not match the locked sample count")

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
    predictions = {
        name: probability.argmax(axis=1) for name, probability in probabilities.items()
    }
    comparisons: dict[str, Any] = {}
    for offset, name in enumerate(("fixed_clip", "arithmetic", "rms")):
        comparison = _compare_metrics(
            metrics["candidate"], metrics[name], seed=args.seed + 10 * offset
        )
        comparison["accuracy"] = _accuracy_comparison(
            predictions["candidate"],
            predictions[name],
            labels,
            seed=args.seed + 100 + 10 * offset,
        )
        comparisons[name] = comparison

    class_rows = []
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        candidate_correct = predictions["candidate"][mask] == labels[mask]
        clip_correct = predictions["fixed_clip"][mask] == labels[mask]
        correction = candidate_correct.astype(np.int64) - clip_correct.astype(np.int64)
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "conflict_samples": int(mask.sum()),
                "candidate_nll": float(metrics["candidate"]["nll"][mask].mean()),
                "fixed_clip_nll": float(metrics["fixed_clip"]["nll"][mask].mean()),
                "nll_improvement": float(
                    (
                        metrics["fixed_clip"]["nll"][mask]
                        - metrics["candidate"]["nll"][mask]
                    ).mean()
                ),
                "candidate_brier": float(metrics["candidate"]["brier"][mask].mean()),
                "fixed_clip_brier": float(metrics["fixed_clip"]["brier"][mask].mean()),
                "brier_improvement": float(
                    (
                        metrics["fixed_clip"]["brier"][mask]
                        - metrics["candidate"]["brier"][mask]
                    ).mean()
                ),
                "candidate_true_probability": float(
                    metrics["candidate"]["true_probability"][mask].mean()
                ),
                "fixed_clip_true_probability": float(
                    metrics["fixed_clip"]["true_probability"][mask].mean()
                ),
                "true_probability_gain": float(
                    (
                        metrics["candidate"]["true_probability"][mask]
                        - metrics["fixed_clip"]["true_probability"][mask]
                    ).mean()
                ),
                "candidate_accuracy_pct": float(candidate_correct.mean() * 100.0),
                "fixed_clip_accuracy_pct": float(clip_correct.mean() * 100.0),
                "accuracy_gain_pp": float(correction.mean() * 100.0),
                "net_corrections": int(correction.sum()),
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
                    candidate["attribute_mean_margin"][local_index]
                ),
                "attribute_fraction": float(
                    candidate["attribute_fraction"][local_index]
                ),
                "clip_pair_fraction": float(
                    candidate["clip_pair_fraction"][local_index]
                ),
                "task_pair_fraction": float(
                    candidate["task_pair_fraction"][local_index]
                ),
                "clip_pair_entropy": float(candidate["clip_pair_entropy"][local_index]),
                "task_pair_entropy": float(candidate["task_pair_entropy"][local_index]),
                "attribute_weight": float(candidate["attribute_weight"][local_index]),
                "anchored_fraction": float(candidate["anchored_fraction"][local_index]),
                "candidate_prediction": int(predictions["candidate"][local_index]),
                "fixed_clip_prediction": int(predictions["fixed_clip"][local_index]),
                "candidate_true_probability": float(
                    metrics["candidate"]["true_probability"][local_index]
                ),
                "fixed_clip_true_probability": float(
                    metrics["fixed_clip"]["true_probability"][local_index]
                ),
                "candidate_nll": float(metrics["candidate"]["nll"][local_index]),
                "fixed_clip_nll": float(metrics["fixed_clip"]["nll"][local_index]),
                "candidate_brier": float(metrics["candidate"]["brier"][local_index]),
                "fixed_clip_brier": float(metrics["fixed_clip"]["brier"][local_index]),
            }
        )
    _write_csv(oracle_csv_path, oracle_rows)

    by_name = {row["class"]: row for row in class_rows}
    car_index = class_names.index("car")
    noncar_net_corrections = int(
        sum(
            row["net_corrections"]
            for row in class_rows
            if row["class_index"] != car_index
        )
    )
    fixed_accuracy = comparisons["fixed_clip"]["accuracy"]
    full_input_contract_valid = (
        source_contract_valid
        and all(mass_lock_checks.values())
        and target_list_hash_matches
    )
    gate = evaluate_attribute_reliability_gate(
        input_contract_valid=full_input_contract_valid,
        fixed_clip_checks=comparisons["fixed_clip"]["checks"],
        accuracy_gain_pp=fixed_accuracy["gain_pp"],
        accuracy_ci_pp=tuple(fixed_accuracy["paired_bootstrap_95_ci_pp"]),
        car_metrics=by_name["car"],
        truck_metrics=by_name["truck"],
        noncar_net_corrections=noncar_net_corrections,
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
        "evidence_reused": {
            "source": "locked pairwise attribute margins and raw mass scale",
            "raw_attribute_mass_decision": mass_lock.get("phase"),
            "new_information": "per-sample task/CLIP pair entropy reliability",
        },
        "compute_contract": {
            "conflict_samples": EXPECTED_CONFLICT_SAMPLES,
            "target_image_forwards": 0,
            "model_checkpoint_loads": 0,
            "model_forwards": 0,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
        },
        "input_contract": {
            "passed": full_input_contract_valid,
            "checks": {
                **source["checks"],
                **mass_lock_checks,
                "target_list_hash_matches_lock_after_target_lock": target_list_hash_matches,
            },
        },
        "label_free_diagnostic": {
            "attribute_weight_mean": float(candidate["attribute_weight"].mean()),
            "attribute_weight_median": float(np.median(candidate["attribute_weight"])),
            "attribute_weight_positive_fraction": float(
                np.mean(candidate["attribute_weight"] > 0.0)
            ),
            "max_abs_class_mass_shift_pp": max_abs_mass_shift_pp,
            "max_abs_class_mass_shift_class": class_names[
                int(np.abs(mass_shift_pp).argmax())
            ],
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
            "noncar_net_corrections": noncar_net_corrections,
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
            "model_checkpoint_loads": 0,
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

    print(f"Wrote label-free reliability target: {target_npz_path}")
    print(f"Wrote label-free class-mass audit: {mass_csv_path}")
    print(f"Locked reliability target before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_csv_path}")
    print(f"Wrote classwise oracle diagnostic: {class_csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        json.dumps(
            {"decision": summary["decision"], "checks": gate["checks"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
