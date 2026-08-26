#!/usr/bin/env python3
"""CPU-only audit of continuous CE weighting for cycle-1 DUET agreements.

Phase 1 locks a label-free weight equal to CLIP agreed-label confidence divided
by its within-pseudo-class mean.  Phase 2 reads oracle labels only to diagnose
logit-space first-order alignment.  No training path is modified or executed.
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

from src.utils.agreement_confidence_weight_audit import (  # noqa: E402
    ce_logit_descent,
    class_balanced_bottom_fraction_reference_weight,
    class_mean_normalized_confidence_weight,
    evaluate_agreement_confidence_weight_gate,
    paired_mean_bootstrap_ci,
    weighted_logit_alignment,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_AGREEMENTS = 6_777
DELAY_FRACTION = 0.10
CLASS_NAMES = [
    "aeroplane", "bicycle", "bus", "car", "horse", "knife",
    "motorcycle", "person", "plant", "skateboard", "train", "truck",
]
DEFAULT_BASE = Path(
    "output/uda/VISDA-C/TV/"
    "duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_BASE / "cycle2_conflict_memory_snapshots/pre_cycle01.npz",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=(
            DEFAULT_BASE
            / "cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
        ),
    )
    parser.add_argument(
        "--target-list",
        type=Path,
        default=Path("data/VISDA-C/validation_proxy25_seed2020_list.txt"),
    )
    parser.add_argument(
        "--class-names", type=Path, default=Path("data/VISDA-C/classname.txt")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE / "agreement_confidence_weight_audit",
    )
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_labels_after_lock(path: Path) -> np.ndarray:
    labels = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            _image_path, label_text = stripped.rsplit(maxsplit=1)
            labels.append(int(label_text))
        except ValueError as error:
            raise ValueError(f"Malformed target row {line_number}: {stripped}") from error
    result = np.asarray(labels, dtype=np.int64)
    if result.shape != (EXPECTED_SAMPLES,):
        raise ValueError(f"Expected {EXPECTED_SAMPLES} labels, found {result.size}")
    if np.any(result < 0) or np.any(result >= EXPECTED_CLASSES):
        raise ValueError("Target label is outside the class range")
    return result


def _class_weight_means(
    weight: np.ndarray, agreement: np.ndarray, pseudo_label: np.ndarray
) -> dict[str, float]:
    return {
        CLASS_NAMES[class_index]: float(
            weight[agreement & (pseudo_label == class_index)].mean()
        )
        for class_index in range(EXPECTED_CLASSES)
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["oracle_metrics"]
    lines = [
        "# VisDA Agreement Confidence-Weight Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The continuous weights were locked before oracle labels were parsed.",
        "All correctness and gradient-alignment measurements are oracle diagnostics.",
        "No model, image, checkpoint, optimizer, backward pass, or training was run.",
        "",
        "## Candidate",
        "",
        "For every cycle-1 task/CLIP top-1 agreement, weight hard CE by CLIP",
        "agreed-label confidence divided by its mean in the same pseudo class.",
        "Every agreement remains supervised and every pseudo class retains mean",
        "weight one.",
        "",
        "## Oracle diagnostic",
        "",
        f"- Candidate first-order mean: `{metrics['candidate']['mean_first_order']:.9f}`.",
        f"- Unweighted first-order mean: `{metrics['unweighted']['mean_first_order']:.9f}`.",
        f"- Bottom-10 delay reference mean: `{metrics['hard_delay_reference']['mean_first_order']:.9f}`.",
        f"- Candidate minus unweighted 95% CI: `{metrics['delta_vs_unweighted_ci']}`.",
        f"- Candidate minus delay 95% CI: `{metrics['delta_vs_hard_delay_ci']}`.",
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend([
        "",
        "PASS would authorize design review only. It does not authorize proxy or",
        "full VisDA training.",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (args.snapshot, args.source_lock, args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing confidence-weight audit input: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    class_names = [
        line.strip().replace("_", " ")
        for line in args.class_names.read_text().splitlines()
        if line.strip()
    ]
    if class_names != CLASS_NAMES:
        raise RuntimeError("VisDA class-name contract changed")
    source_lock = json.loads(args.source_lock.read_text())
    snapshot_sha256 = _sha256(args.snapshot)

    # Phase 1: the snapshot target_label key is declared but deliberately not read.
    required = {
        "cycle", "label_mask", "source_label", "clip_label", "task_prob",
        "clip_prob", "sample_index", "target_label",
    }
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        missing = required.difference(snapshot.files)
        if missing:
            raise ValueError(f"Snapshot is missing keys: {sorted(missing)}")
        cycle = int(np.asarray(snapshot["cycle"]).item())
        label_mask = np.asarray(snapshot["label_mask"], dtype=bool).copy()
        source_label = np.asarray(snapshot["source_label"], dtype=np.int64).copy()
        clip_label = np.asarray(snapshot["clip_label"], dtype=np.int64).copy()
        task_probability = np.asarray(snapshot["task_prob"], dtype=np.float64).copy()
        clip_probability = np.asarray(snapshot["clip_prob"], dtype=np.float64).copy()
        sample_index = np.asarray(snapshot["sample_index"], dtype=np.int64).copy()

    agreement = source_label == clip_label
    pseudo_label = source_label
    row = np.arange(EXPECTED_SAMPLES)
    clip_confidence = clip_probability[row, pseudo_label]
    candidate_weight = class_mean_normalized_confidence_weight(
        clip_confidence, agreement, pseudo_label
    )
    hard_delay = class_balanced_bottom_fraction_reference_weight(
        clip_confidence,
        agreement,
        pseudo_label,
        fraction=DELAY_FRACTION,
    )
    unweighted = agreement.astype(np.float64)
    candidate_class_means = _class_weight_means(
        candidate_weight, agreement, pseudo_label
    )
    hard_delay_class_means = _class_weight_means(
        hard_delay["weight"], agreement, pseudo_label
    )
    max_class_mean_error = max(
        abs(value - 1.0) for value in candidate_class_means.values()
    )

    input_checks = {
        "source_snapshot_matches_cycle2_signal_lock": (
            snapshot_sha256
            == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
        ),
        "source_lock_declares_labels_after_manifest": bool(
            source_lock.get("labels_read_after_this_manifest")
        ),
        "cycle_is_one": cycle == 1,
        "expected_probability_shape": (
            task_probability.shape == clip_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "probabilities_finite_and_normalized": bool(
            np.isfinite(task_probability).all()
            and np.isfinite(clip_probability).all()
            and np.allclose(task_probability.sum(1), 1.0, atol=1e-5)
            and np.allclose(clip_probability.sum(1), 1.0, atol=1e-5)
        ),
        "sample_indices_are_proxy_order": np.array_equal(
            sample_index, np.arange(EXPECTED_SAMPLES)
        ),
        "saved_predictions_match_probabilities": (
            np.array_equal(source_label, task_probability.argmax(1))
            and np.array_equal(clip_label, clip_probability.argmax(1))
        ),
        "duet_mask_equals_top1_agreement": np.array_equal(label_mask, agreement),
        "expected_agreement_count": int(agreement.sum()) == EXPECTED_AGREEMENTS,
        "every_pseudo_class_has_agreements": all(
            np.any(agreement & (pseudo_label == class_index))
            for class_index in range(EXPECTED_CLASSES)
        ),
        "candidate_weights_positive_on_all_agreements": bool(
            np.all(candidate_weight[agreement] > 0.0)
        ),
        "candidate_weights_zero_outside_agreements": bool(
            np.all(candidate_weight[~agreement] == 0.0)
        ),
        "candidate_preserves_each_pseudo_class_mean": max_class_mean_error <= 1e-10,
        "hard_delay_reference_preserves_each_pseudo_class_mean": max(
            abs(value - 1.0) for value in hard_delay_class_means.values()
        ) <= 1e-10,
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Confidence-weight input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_agreement_confidence_weight"
    signal_path = args.output_dir / f"{stem}_label_free.npz"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"
    np.savez_compressed(
        signal_path,
        index=sample_index,
        agreement=agreement,
        pseudo_label=pseudo_label,
        clip_agreed_label_confidence=clip_confidence,
        candidate_weight=candidate_weight,
        hard_delay_reference_weight=hard_delay["weight"],
        hard_delay_selected=hard_delay["delayed"],
    )
    lock = {
        "phase": "LABEL_FREE_AGREEMENT_CONFIDENCE_WEIGHT_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "scope": "cycle1 task/CLIP top1 agreements only",
            "signal": "CLIP probability of the shared top1 pseudo label",
            "weight": "confidence divided by its within-pseudo-class mean",
            "all_agreements_retained": True,
            "per_pseudo_class_mean_weight": 1.0,
            "fitted_parameters": False,
            "target_label_thresholds": False,
            "training_change_in_this_audit": False,
        },
        "hard_delay_reference": {
            "selection": "bottom 10 percent CLIP confidence per pseudo class",
            "population_rescaling": "retained class mean weight equals one",
            "limitation": "not an exact replay of minibatch normalization",
        },
        "predeclared_gate": {
            "min_effective_sample_size_pct": 90.0,
            "paired_first_order_ci_lower_vs_unweighted_and_delay": "> 0",
            "negative_burden": "lower than unweighted",
            "positive_support": "nonworse than hard-delay reference",
            "car_truck_noncar_first_order_delta": ">= 0",
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": {
            "samples": EXPECTED_SAMPLES,
            "agreements": int(agreement.sum()),
            "candidate_weight_min": float(candidate_weight[agreement].min()),
            "candidate_weight_max": float(candidate_weight[agreement].max()),
            "candidate_weight_mean": float(candidate_weight[agreement].mean()),
            "candidate_class_means": candidate_class_means,
            "hard_delay_selected": int(hard_delay["delayed"].sum()),
        },
        "inputs": {
            "pre_cycle1_snapshot": {"path": str(args.snapshot), "sha256": snapshot_sha256},
            "cycle2_signal_lock": {
                "path": str(args.source_lock), "sha256": _sha256(args.source_lock)
            },
            "target_list_opaque_sha256": _sha256(args.target_list),
            "class_names_sha256": _sha256(args.class_names),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/agreement_confidence_weight_audit.py": _sha256(
                REPO_ROOT / "src/utils/agreement_confidence_weight_audit.py"
            ),
            "tools/audit_visda_agreement_confidence_weight.py": _sha256(
                Path(__file__).resolve()
            ),
            "src/methods/oh/plmatch.py": _sha256(
                REPO_ROOT / "src/methods/oh/plmatch.py"
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: labels are accessed only after the label-free artifact is locked.
    target_hash_matches = (
        _sha256(args.target_list) == lock["inputs"]["target_list_opaque_sha256"]
    )
    target_labels = _parse_labels_after_lock(args.target_list)
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        embedded_labels = np.asarray(snapshot["target_label"], dtype=np.int64).copy()
    labels_match_snapshot = np.array_equal(
        target_labels[sample_index], embedded_labels
    )
    labels = target_labels[sample_index]

    pseudo_descent = ce_logit_descent(task_probability, pseudo_label)
    oracle_descent = ce_logit_descent(task_probability, labels)
    summaries = {
        "unweighted": weighted_logit_alignment(
            pseudo_descent, oracle_descent, unweighted, agreement
        ),
        "candidate": weighted_logit_alignment(
            pseudo_descent, oracle_descent, candidate_weight, agreement
        ),
        "hard_delay_reference": weighted_logit_alignment(
            pseudo_descent, oracle_descent, hard_delay["weight"], agreement
        ),
    }
    delta_vs_unweighted_ci = paired_mean_bootstrap_ci(
        summaries["candidate"]["row_first_order"],
        summaries["unweighted"]["row_first_order"],
        agreement,
        seed=2020,
    )
    delta_vs_hard_delay_ci = paired_mean_bootstrap_ci(
        summaries["candidate"]["row_first_order"],
        summaries["hard_delay_reference"]["row_first_order"],
        agreement,
        seed=2021,
    )

    correct = pseudo_label == labels
    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_group = agreement & (labels == class_index)
        pseudo_group = agreement & (pseudo_label == class_index)
        candidate_delta = (
            summaries["candidate"]["row_first_order"]
            - summaries["unweighted"]["row_first_order"]
        )
        class_rows.append({
            "true_class_index": class_index,
            "true_class": class_name,
            "agreement_samples": int(true_group.sum()),
            "agreement_accuracy_pct": float(correct[true_group].mean() * 100.0),
            "candidate_first_order_delta": float(candidate_delta[true_group].mean()),
            "pseudo_class_samples": int(pseudo_group.sum()),
            "pseudo_class_candidate_weight_mean": float(
                candidate_weight[pseudo_group].mean()
            ),
        })
    _write_csv(class_path, class_rows)

    oracle_rows = []
    for index in np.flatnonzero(agreement):
        oracle_rows.append({
            "index": int(sample_index[index]),
            "label": int(labels[index]),
            "label_name": CLASS_NAMES[int(labels[index])],
            "pseudo_label": int(pseudo_label[index]),
            "pseudo_label_name": CLASS_NAMES[int(pseudo_label[index])],
            "pseudo_label_correct": bool(correct[index]),
            "clip_agreed_label_confidence": float(clip_confidence[index]),
            "candidate_weight": float(candidate_weight[index]),
            "hard_delay_selected": bool(hard_delay["delayed"][index]),
            "unweighted_first_order": float(
                summaries["unweighted"]["row_first_order"][index]
            ),
            "candidate_first_order": float(
                summaries["candidate"]["row_first_order"][index]
            ),
            "hard_delay_reference_first_order": float(
                summaries["hard_delay_reference"]["row_first_order"][index]
            ),
        })
    _write_csv(oracle_path, oracle_rows)

    candidate_delta = (
        summaries["candidate"]["row_first_order"]
        - summaries["unweighted"]["row_first_order"]
    )
    car_mask = agreement & (labels == CLASS_NAMES.index("car"))
    truck_mask = agreement & (labels == CLASS_NAMES.index("truck"))
    noncar_mask = agreement & ~np.isin(
        labels, [CLASS_NAMES.index("car"), CLASS_NAMES.index("truck")]
    )
    car_delta = float(candidate_delta[car_mask].mean())
    truck_delta = float(candidate_delta[truck_mask].mean())
    noncar_delta = float(candidate_delta[noncar_mask].mean())
    input_contract_valid = (
        all(input_checks.values()) and target_hash_matches and labels_match_snapshot
    )
    gate = evaluate_agreement_confidence_weight_gate(
        input_contract_valid=input_contract_valid,
        max_pseudo_class_mean_weight_error=max_class_mean_error,
        effective_sample_size_pct=summaries["candidate"][
            "effective_sample_size_pct"
        ],
        delta_vs_unweighted_ci=delta_vs_unweighted_ci,
        delta_vs_hard_delay_ci=delta_vs_hard_delay_ci,
        candidate_negative_burden=summaries["candidate"]["negative_burden"],
        baseline_negative_burden=summaries["unweighted"]["negative_burden"],
        candidate_positive_support=summaries["candidate"]["positive_support"],
        hard_delay_positive_support=summaries["hard_delay_reference"][
            "positive_support"
        ],
        car_first_order_delta=car_delta,
        truck_first_order_delta=truck_delta,
        noncar_first_order_delta=noncar_delta,
    )
    serializable_summaries = {
        name: {key: value for key, value in metric.items() if key != "row_first_order"}
        for name, metric in summaries.items()
    }
    summary = {
        "dataset": "VisDA-C",
        "seed": 2020,
        "decision": gate["decision"],
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "candidate_contract": lock["candidate_contract"],
        "input_contract": {
            "passed": input_contract_valid,
            "checks": {
                **input_checks,
                "target_list_hash_matches_after_lock": target_hash_matches,
                "target_labels_match_embedded_snapshot_after_lock": labels_match_snapshot,
            },
        },
        "label_free_metrics": lock["label_free_metrics"],
        "oracle_metrics": {
            **serializable_summaries,
            "delta_vs_unweighted": (
                summaries["candidate"]["mean_first_order"]
                - summaries["unweighted"]["mean_first_order"]
            ),
            "delta_vs_unweighted_ci": list(delta_vs_unweighted_ci),
            "delta_vs_hard_delay": (
                summaries["candidate"]["mean_first_order"]
                - summaries["hard_delay_reference"]["mean_first_order"]
            ),
            "delta_vs_hard_delay_ci": list(delta_vs_hard_delay_ci),
            "car_first_order_delta": car_delta,
            "truck_first_order_delta": truck_delta,
            "noncar_first_order_delta": noncar_delta,
            "classwise": class_rows,
        },
        "gate": gate,
        "scope_limit": (
            "PASS authorizes only review of one cycle-1 CE-weight proxy design. "
            "It never authorizes or starts proxy/full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote label-free weights: {signal_path}")
    print(f"Locked weights before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
