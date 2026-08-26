#!/usr/bin/env python3
"""CPU-only audit of runner-up disagreement inside cycle-1 DUET agreements.

Phase 1 reads only probability, prediction, mask, and index arrays from an
existing pre-cycle-1 snapshot.  It locks a class-balanced rank-residual selector
and three matched confidence comparators.  Phase 2 then parses target labels for
explicit oracle diagnostics.  No image, checkpoint, model, forward, backward,
optimizer, or training operation is involved.
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

from src.utils.agreement_rank_residual_audit import (  # noqa: E402
    agreement_rank_residual,
    evaluate_agreement_rank_residual_gate,
    paired_selection_precision_bootstrap_ci,
    select_class_balanced_fraction,
    select_matched_counts,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_AGREEMENTS = 6_777
EXPECTED_AGREEMENT_ACCURACY_PCT = 94.11
SELECTION_FRACTION = 0.10
CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
]
DEFAULT_SNAPSHOT = Path(
    "output/uda/VISDA-C/TV/"
    "duet_support_conditioned_clip_cycle2_memory_audit_seed2020/"
    "cycle2_conflict_memory_snapshots/pre_cycle01.npz"
)
DEFAULT_SOURCE_LOCK = Path(
    "output/uda/VISDA-C/TV/"
    "duet_support_conditioned_clip_cycle2_memory_audit_seed2020/"
    "cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "duet_support_conditioned_clip_cycle2_memory_audit_seed2020/"
    "agreement_rank_residual_audit"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument(
        "--target-list",
        type=Path,
        default=Path("data/VISDA-C/validation_proxy25_seed2020_list.txt"),
    )
    parser.add_argument(
        "--class-names", type=Path, default=Path("data/VISDA-C/classname.txt")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
                f"Malformed target row {line_number}: {stripped}"
            ) from error
        labels.append(label)
    result = np.asarray(labels, dtype=np.int64)
    if result.shape != (EXPECTED_SAMPLES,):
        raise ValueError(f"Expected {EXPECTED_SAMPLES} target labels, found {result.size}")
    if np.any(result < 0) or np.any(result >= EXPECTED_CLASSES):
        raise ValueError("Target label is outside the class range")
    return result


def _accuracy(correct: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    return float(np.asarray(correct, dtype=np.float64)[selected].mean() * 100.0)


def _capture_metrics(selected: np.ndarray, wrong: np.ndarray) -> dict[str, Any]:
    count = int(selected.sum())
    captured = int((selected & wrong).sum())
    return {
        "selected": count,
        "wrong_captured": captured,
        "selection_error_precision_pct": float(captured / count * 100.0),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["oracle_metrics"]
    lines = [
        "# VisDA Agreement Rank-Residual Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The selector and matched confidence masks were locked before labels",
        "were parsed. All correctness measurements are oracle diagnostics.",
        "No training or model execution was performed.",
        "",
        "## Main result",
        "",
        f"- Cycle-1 agreements: `{metrics['agreement_samples']}`; accuracy "
        f"`{metrics['agreement_accuracy_pct']:.6f}%`.",
        f"- Rank-residual selection: `{metrics['candidate']['selected']}`; error "
        f"precision `{metrics['candidate']['selection_error_precision_pct']:.6f}%`.",
        f"- Error enrichment: `{metrics['candidate_error_enrichment']:.6f}x`.",
        f"- Retained pseudo-label accuracy: "
        f"`{metrics['candidate_retained_accuracy_pct']:.6f}%` "
        f"(`{metrics['candidate_retained_accuracy_gain_pp']:.6f} pp`).",
        "",
        "## Matched confidence comparisons",
        "",
    ]
    for name, comparison in metrics["comparisons"].items():
        lines.append(
            f"- {name}: captured-error gain `{comparison['captured_error_gain']}`; "
            f"precision gain `{comparison['selection_precision_gain_pp']:.6f} pp`; "
            f"95% CI `{comparison['paired_bootstrap_95_ci_pp']}`."
        )
    lines.extend(["", "## Gate", ""])
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend([
        "",
        "Passing authorizes only a one-variable matched proxy design. It does",
        "not authorize or start proxy/full VisDA training.",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (args.snapshot, args.source_lock, args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing agreement audit input: {path}")
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
    source_snapshot_matches_lock = (
        snapshot_sha256
        == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
    )

    # Phase 1: access only named label-free keys. The snapshot's target_label
    # key is deliberately not read until after the signal lock is on disk.
    required = {
        "cycle",
        "label_mask",
        "source_label",
        "clip_label",
        "task_prob",
        "clip_prob",
        "sample_index",
        "target_label",
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

    residual = agreement_rank_residual(task_probability, clip_probability)
    agreement = residual["common_top1"]
    common_label = residual["task_top2"][:, 0]
    candidate = select_class_balanced_fraction(
        residual["rank_residual"], agreement, common_label,
        fraction=SELECTION_FRACTION, largest=True,
    )
    candidate_selected = candidate["selected"]
    row = np.arange(EXPECTED_SAMPLES)
    task_confidence = task_probability[row, common_label]
    clip_confidence = clip_probability[row, common_label]
    arithmetic_confidence = 0.5 * (task_confidence + clip_confidence)
    rms_probability = np.sqrt(
        (np.square(task_probability) + np.square(clip_probability)) / 2.0
    )
    rms_probability /= rms_probability.sum(axis=1, keepdims=True)
    rms_confidence = rms_probability[row, common_label]
    baseline_selected = {
        "task_confidence": select_matched_counts(
            task_confidence, agreement, common_label,
            candidate["counts_by_group"], largest=False,
        ),
        "clip_confidence": select_matched_counts(
            clip_confidence, agreement, common_label,
            candidate["counts_by_group"], largest=False,
        ),
        "arithmetic_confidence": select_matched_counts(
            arithmetic_confidence, agreement, common_label,
            candidate["counts_by_group"], largest=False,
        ),
        "rms_confidence": select_matched_counts(
            rms_confidence,
            agreement,
            common_label,
            candidate["counts_by_group"],
            largest=False,
        ),
    }
    expected_by_class = {
        class_index: max(
            1,
            int(
                np.ceil(
                    (agreement & (common_label == class_index)).sum()
                    * SELECTION_FRACTION
                )
            ),
        )
        for class_index in range(EXPECTED_CLASSES)
    }
    input_checks = {
        "source_snapshot_matches_cycle2_signal_lock": source_snapshot_matches_lock,
        "source_lock_declares_labels_after_manifest": bool(
            source_lock.get("labels_read_after_this_manifest")
        ),
        "cycle_is_one": cycle == 1,
        "expected_probability_shape": (
            task_probability.shape == clip_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "probabilities_finite": bool(
            np.isfinite(task_probability).all() and np.isfinite(clip_probability).all()
        ),
        "probabilities_normalized": bool(
            np.allclose(task_probability.sum(1), 1.0, atol=1e-5)
            and np.allclose(clip_probability.sum(1), 1.0, atol=1e-5)
        ),
        "sample_indices_are_proxy_order": np.array_equal(
            sample_index, np.arange(EXPECTED_SAMPLES)
        ),
        "saved_task_prediction_matches_probability": np.array_equal(
            source_label, task_probability.argmax(1)
        ),
        "saved_clip_prediction_matches_probability": np.array_equal(
            clip_label, clip_probability.argmax(1)
        ),
        "duet_mask_equals_top1_agreement": np.array_equal(label_mask, agreement),
        "expected_agreement_count": int(agreement.sum()) == EXPECTED_AGREEMENTS,
        "every_class_has_agreements": len(candidate["counts_by_group"]) == EXPECTED_CLASSES,
        "candidate_counts_are_exact_classwise_ten_percent": (
            candidate["counts_by_group"] == expected_by_class
        ),
        "every_selected_row_has_positive_rank_residual": bool(
            np.all(residual["rank_residual"][candidate_selected] > 0.0)
        ),
        "matched_baseline_counts_equal_candidate_by_class": all(
            all(
                int((mask & (common_label == class_index)).sum()) == count
                for class_index, count in candidate["counts_by_group"].items()
            )
            for mask in baseline_selected.values()
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Agreement rank-residual input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_agreement_rank_residual"
    signal_path = args.output_dir / f"{stem}_label_free.npz"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"
    np.savez_compressed(
        signal_path,
        index=sample_index,
        common_top1=common_label,
        task_runner_up=residual["task_top2"][:, 1],
        clip_runner_up=residual["clip_top2"][:, 1],
        agreement=agreement,
        runner_up_disagreement=residual["runner_up_disagreement"],
        rank_residual=residual["rank_residual"],
        candidate_selected=candidate_selected,
        task_confidence_selected=baseline_selected["task_confidence"],
        clip_confidence_selected=baseline_selected["clip_confidence"],
        arithmetic_confidence_selected=baseline_selected["arithmetic_confidence"],
        rms_confidence_selected=baseline_selected["rms_confidence"],
    )
    lock = {
        "phase": "LABEL_FREE_AGREEMENT_RANK_RESIDUAL_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "scope": "cycle1 task/CLIP top1 agreements only",
            "signal": "geometric mean of opposing task/CLIP runner-up probability margins",
            "selection": "largest 10 percent independently within each common pseudo class",
            "comparators": (
                "lowest task, CLIP, arithmetic, and normalized-RMS agreed-label "
                "confidence at identical per-class counts"
            ),
            "target_label_thresholds": False,
            "class_specific_thresholds": False,
            "fitted_parameters": False,
            "weak_strong_signal": False,
            "clip_kl_signal": False,
        },
        "predeclared_gate": {
            "min_error_enrichment": 2.0,
            "min_retained_accuracy_gain_pp": 0.25,
            "paired_precision_ci_lower": "> 0 against all four confidence baselines",
            "car_truck_noncar_wrong_capture": "nonworse than all three matched baselines",
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": {
            "samples": EXPECTED_SAMPLES,
            "agreement_samples": int(agreement.sum()),
            "runner_up_disagreement_samples": int(residual["runner_up_disagreement"].sum()),
            "runner_up_disagreement_within_agreements_pct": float(
                residual["runner_up_disagreement"].sum() / agreement.sum() * 100.0
            ),
            "candidate_selected": int(candidate_selected.sum()),
            "candidate_selected_within_agreements_pct": float(
                candidate_selected.sum() / agreement.sum() * 100.0
            ),
            "selected_by_pseudo_class": {
                CLASS_NAMES[key]: value for key, value in candidate["counts_by_group"].items()
            },
        },
        "inputs": {
            "pre_cycle1_snapshot": {"path": str(args.snapshot), "sha256": snapshot_sha256},
            "cycle2_signal_lock": {
                "path": str(args.source_lock),
                "sha256": _sha256(args.source_lock),
            },
            "target_list_opaque_sha256": _sha256(args.target_list),
            "class_names_sha256": _sha256(args.class_names),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/agreement_rank_residual_audit.py": _sha256(
                REPO_ROOT / "src/utils/agreement_rank_residual_audit.py"
            ),
            "tools/audit_visda_agreement_rank_residual.py": _sha256(Path(__file__).resolve()),
            "src/methods/oh/plmatch.py": _sha256(REPO_ROOT / "src/methods/oh/plmatch.py"),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: target labels are accessed only after the label-free lock exists.
    target_list_hash_matches = (
        _sha256(args.target_list)
        == lock["inputs"]["target_list_opaque_sha256"]
    )
    target_labels = _parse_labels_after_lock(args.target_list)
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        embedded_labels = np.asarray(snapshot["target_label"], dtype=np.int64).copy()
    labels_match_snapshot = np.array_equal(target_labels[sample_index], embedded_labels)
    labels = target_labels[sample_index]
    correct = common_label == labels
    wrong = agreement & ~correct
    agreement_accuracy = _accuracy(correct, agreement)
    baseline_reproduced = (
        int(agreement.sum()) == EXPECTED_AGREEMENTS
        and round(agreement_accuracy, 2) == EXPECTED_AGREEMENT_ACCURACY_PCT
        and target_list_hash_matches
        and labels_match_snapshot
    )

    candidate_metrics = _capture_metrics(candidate_selected, wrong)
    base_error_rate = float(wrong.sum() / agreement.sum())
    candidate_enrichment = (
        candidate_metrics["selection_error_precision_pct"] / 100.0 / base_error_rate
    )
    retained = agreement & ~candidate_selected
    retained_accuracy = _accuracy(correct, retained)
    comparisons = {}
    baseline_metrics = {}
    for offset, (name, selected) in enumerate(baseline_selected.items()):
        metrics = _capture_metrics(selected, wrong)
        baseline_metrics[name] = metrics
        precision_gain = (
            candidate_metrics["selection_error_precision_pct"]
            - metrics["selection_error_precision_pct"]
        )
        interval = paired_selection_precision_bootstrap_ci(
            candidate_selected, selected, wrong, agreement, seed=2020 + offset,
        )
        comparisons[name] = {
            "captured_error_gain": (
                candidate_metrics["wrong_captured"] - metrics["wrong_captured"]
            ),
            "selection_precision_gain_pp": precision_gain,
            "paired_bootstrap_95_ci_pp": list(interval),
            "baseline": metrics,
        }

    selected_masks = {"candidate": candidate_selected, **baseline_selected}
    capture_groups: dict[str, dict[str, int]] = {}
    for group_name, group_mask in {
        "car": common_label == CLASS_NAMES.index("car"),
        "truck": common_label == CLASS_NAMES.index("truck"),
        "noncar": common_label != CLASS_NAMES.index("car"),
    }.items():
        if group_name == "noncar":
            group_mask &= common_label != CLASS_NAMES.index("truck")
        capture_groups[group_name] = {
            name: int((mask & wrong & group_mask).sum())
            for name, mask in selected_masks.items()
        }

    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        group_mask = agreement & (common_label == class_index)
        candidate_class = candidate_selected & group_mask
        row_data: dict[str, Any] = {
            "pseudo_class_index": class_index,
            "pseudo_class": class_name,
            "agreements": int(group_mask.sum()),
            "agreement_errors": int((wrong & group_mask).sum()),
            "agreement_accuracy_pct": _accuracy(correct, group_mask),
            "candidate_selected": int(candidate_class.sum()),
            "candidate_wrong_captured": int((candidate_class & wrong).sum()),
            "candidate_selection_error_precision_pct": _capture_metrics(
                candidate_class, wrong
            )["selection_error_precision_pct"],
        }
        for name, selected in baseline_selected.items():
            baseline_class = selected & group_mask
            row_data[f"{name}_wrong_captured"] = int((baseline_class & wrong).sum())
        class_rows.append(row_data)
    _write_csv(class_path, class_rows)

    oracle_rows = []
    for index in np.flatnonzero(agreement):
        oracle_rows.append({
            "index": int(sample_index[index]),
            "label": int(labels[index]),
            "label_name": CLASS_NAMES[int(labels[index])],
            "common_top1": int(common_label[index]),
            "common_top1_name": CLASS_NAMES[int(common_label[index])],
            "common_top1_correct": bool(correct[index]),
            "task_runner_up": int(residual["task_top2"][index, 1]),
            "clip_runner_up": int(residual["clip_top2"][index, 1]),
            "runner_up_disagreement": bool(residual["runner_up_disagreement"][index]),
            "rank_residual": float(residual["rank_residual"][index]),
            "candidate_selected": bool(candidate_selected[index]),
            "task_confidence_selected": bool(baseline_selected["task_confidence"][index]),
            "clip_confidence_selected": bool(baseline_selected["clip_confidence"][index]),
            "arithmetic_confidence_selected": bool(
                baseline_selected["arithmetic_confidence"][index]
            ),
            "rms_confidence_selected": bool(
                baseline_selected["rms_confidence"][index]
            ),
        })
    _write_csv(oracle_path, oracle_rows)

    gate = evaluate_agreement_rank_residual_gate(
        input_contract_valid=(
            all(input_checks.values())
            and target_list_hash_matches
            and labels_match_snapshot
        ),
        baseline_reproduced=baseline_reproduced,
        selected_fraction_pct=float(candidate_selected.sum() / agreement.sum() * 100.0),
        error_enrichment=candidate_enrichment,
        retained_accuracy_gain_pp=retained_accuracy - agreement_accuracy,
        comparisons=comparisons,
        car_wrong_captures=capture_groups["car"],
        truck_wrong_captures=capture_groups["truck"],
        noncar_wrong_captures=capture_groups["noncar"],
    )
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
            "passed": (
                all(input_checks.values())
                and target_list_hash_matches
                and labels_match_snapshot
            ),
            "checks": {
                **input_checks,
                "target_list_hash_matches_after_lock": target_list_hash_matches,
                "target_list_labels_match_embedded_snapshot_after_lock": labels_match_snapshot,
            },
        },
        "label_free_metrics": lock["label_free_metrics"],
        "baseline_reproduction": {
            "passed": baseline_reproduced,
            "expected_agreements": EXPECTED_AGREEMENTS,
            "observed_agreements": int(agreement.sum()),
            "expected_accuracy_pct_rounded": EXPECTED_AGREEMENT_ACCURACY_PCT,
            "observed_accuracy_pct": agreement_accuracy,
        },
        "oracle_metrics": {
            "agreement_samples": int(agreement.sum()),
            "agreement_errors": int(wrong.sum()),
            "agreement_accuracy_pct": agreement_accuracy,
            "candidate": candidate_metrics,
            "candidate_error_enrichment": candidate_enrichment,
            "candidate_retained_accuracy_pct": retained_accuracy,
            "candidate_retained_accuracy_gain_pp": retained_accuracy - agreement_accuracy,
            "baselines": baseline_metrics,
            "comparisons": comparisons,
            "wrong_captures_by_pseudo_class_group": capture_groups,
            "classwise": class_rows,
        },
        "gate": gate,
        "scope_limit": (
            "PASS authorizes only design of one matched proxy that changes the "
            "cycle-1 admission mask. It never authorizes or starts training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote label-free signal: {signal_path}")
    print(f"Locked signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
