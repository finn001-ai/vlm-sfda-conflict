#!/usr/bin/env python3
"""CPU-only audit of a shared-runner-up partial-label set in DUET agreements.

The label-free phase selects cycle-1 task/CLIP top-1 agreements for which both
models also name the same runner-up.  It locks the two-label set before target
labels are parsed.  The oracle phase measures coverage and exact logit-space
gradient alignment against both hard top-1 CE and dropping the selected rows.
No training path is changed or executed.
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
    paired_selection_precision_bootstrap_ci,
    select_matched_counts,
)
from src.utils.agreement_shared_runner_up_audit import (  # noqa: E402
    evaluate_shared_runner_up_gate,
    shared_runner_up_candidate,
)
from src.utils.candidate_set_gradient_audit import (  # noqa: E402
    oracle_ce_logit_descent,
    paired_mean_bootstrap_ci,
    rowwise_oracle_alignment,
    set_mass_logit_descent,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_AGREEMENTS = 6_777
CLASS_NAMES = [
    "aeroplane", "bicycle", "bus", "car", "horse", "knife",
    "motorcycle", "person", "plant", "skateboard", "train", "truck",
]
HARD_CLASSES = ("car", "person", "truck")
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
        default=DEFAULT_BASE / "agreement_shared_runner_up_audit",
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


def _capture_metrics(selected: np.ndarray, wrong: np.ndarray) -> dict[str, Any]:
    count = int(selected.sum())
    captured = int((selected & wrong).sum())
    return {
        "selected": count,
        "wrong_captured": captured,
        "selection_error_precision_pct": float(captured / count * 100.0),
    }


def _alignment_summary(alignment: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    return {
        "mean_first_order": float(alignment["first_order"][selected].mean()),
        "mean_oracle_unit_projection": float(
            alignment["oracle_unit_projection"][selected].mean()
        ),
        "mean_cosine": float(alignment["cosine"][selected].mean()),
        "harmful_fraction_pct": float(
            (alignment["first_order"][selected] < 0.0).mean() * 100.0
        ),
        "nonzero_fraction_pct": float(
            alignment["joint_nonzero"][selected].mean() * 100.0
        ),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["oracle_metrics"]
    lines = [
        "# VisDA Agreement Shared Runner-Up Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The common top-2 sets were locked before oracle labels were parsed.",
        "All coverage and gradient measurements are oracle diagnostics.",
        "No image, model, checkpoint, optimizer, backward pass, or training was run.",
        "",
        "## Evidence",
        "",
        f"- Selected agreements: `{metrics['selected_samples']}` ",
        f"(`{metrics['selected_fraction_pct']:.6f}%`).",
        f"- Selected two-label oracle coverage: ",
        f"`{metrics['selected_candidate_coverage_pct']:.6f}%`.",
        f"- Selected top-1 miss recovery: ",
        f"`{metrics['selected_top1_miss_recovery_pct']:.6f}%`.",
        f"- First-order delta versus hard top-1: ",
        f"`{metrics['candidate_delta_vs_top1']:.9f}`; ",
        f"95% CI `{metrics['candidate_delta_vs_top1_ci']}`.",
        f"- First-order delta versus dropping the selected rows: ",
        f"`{metrics['candidate_delta_vs_zero_delay']:.9f}`; ",
        f"95% CI `{metrics['candidate_delta_vs_zero_delay_ci']}`.",
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend([
        "",
        "PASS authorizes method design review only; it never starts training.",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (args.snapshot, args.source_lock, args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing shared runner-up audit input: {path}")
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

    # Phase 1: the target_label key is declared but deliberately not accessed.
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

    candidate = shared_runner_up_candidate(task_probability, clip_probability)
    agreement = candidate["agreement"]
    selected = candidate["selected"]
    common_top1 = candidate["common_top1"]
    shared_runner_up = candidate["shared_runner_up"]
    row = np.arange(EXPECTED_SAMPLES)
    clip_confidence = clip_probability[row, common_top1]
    selected_counts = {
        class_index: int((selected & (common_top1 == class_index)).sum())
        for class_index in range(EXPECTED_CLASSES)
    }
    confidence_selected = select_matched_counts(
        clip_confidence,
        agreement,
        common_top1,
        selected_counts,
        largest=False,
    )
    candidate_sizes = candidate["candidate_mask"].sum(axis=1)
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
        "selected_is_nonempty_subset_of_agreements": bool(
            selected.any() and np.all(agreement[selected])
        ),
        "selected_rows_have_exactly_two_candidates": bool(
            np.all(candidate_sizes[selected] == 2)
        ),
        "unselected_rows_have_exactly_one_candidate": bool(
            np.all(candidate_sizes[~selected] == 1)
        ),
        "matched_confidence_counts_equal_by_pseudo_class": all(
            int((confidence_selected & (common_top1 == class_index)).sum()) == count
            for class_index, count in selected_counts.items()
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Shared runner-up input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_agreement_shared_runner_up"
    signal_path = args.output_dir / f"{stem}_label_free.npz"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    pair_path = args.output_dir / f"{stem}_pairwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"
    np.savez_compressed(
        signal_path,
        index=sample_index,
        agreement=agreement,
        selected=selected,
        common_top1=common_top1,
        shared_runner_up=shared_runner_up,
        candidate_mask=candidate["candidate_mask"],
        clip_confidence_selected=confidence_selected,
    )
    lock = {
        "phase": "LABEL_FREE_AGREEMENT_SHARED_RUNNER_UP_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "scope": "cycle1 task/CLIP top1 agreements only",
            "selection": "task and CLIP also share the same runner-up class",
            "candidate_set": "shared top1 plus shared runner-up",
            "candidate_loss_diagnostic": "negative log student mass on the two-label set",
            "comparison": "hard top1 CE and zero gradient on the same selected rows",
            "target_label_thresholds": False,
            "class_specific_rules": False,
            "fitted_parameters": False,
            "training_change_in_this_audit": False,
        },
        "novel_scope_relative_to_prior_audits": (
            "prior candidate-set audits covered task/CLIP top1 conflicts; this audit "
            "covers false agreements with a shared second candidate"
        ),
        "predeclared_gate": {
            "selected_fraction_pct": [5.0, 80.0],
            "min_candidate_coverage_pct": 98.0,
            "min_top1_miss_recovery_pct": 50.0,
            "paired_first_order_ci_lower_vs_top1_and_zero_delay": "> 0",
            "car_person_truck_other9_first_order_delta": ">= 0",
            "max_full_mass_shift_pp": 1.0,
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": {
            "samples": EXPECTED_SAMPLES,
            "agreements": int(agreement.sum()),
            "selected": int(selected.sum()),
            "selected_fraction_pct": float(selected.sum() / agreement.sum() * 100.0),
            "selected_by_pseudo_class": {
                CLASS_NAMES[key]: value for key, value in selected_counts.items()
            },
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
            "src/utils/agreement_shared_runner_up_audit.py": _sha256(
                REPO_ROOT / "src/utils/agreement_shared_runner_up_audit.py"
            ),
            "tools/audit_visda_agreement_shared_runner_up.py": _sha256(
                Path(__file__).resolve()
            ),
            "src/methods/oh/plmatch.py": _sha256(
                REPO_ROOT / "src/methods/oh/plmatch.py"
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: target labels are read only after the label-free lock exists.
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
    correct_top1 = common_top1 == labels
    wrong = agreement & ~correct_top1
    candidate_contains_label = candidate["candidate_mask"][row, labels]
    selected_wrong = selected & wrong
    recovered = selected_wrong & (shared_runner_up == labels)

    top1_mask = np.zeros_like(candidate["candidate_mask"])
    top1_mask[row, common_top1] = True
    top1_descent = set_mass_logit_descent(task_probability, top1_mask)
    candidate_descent = set_mass_logit_descent(
        task_probability, candidate["candidate_mask"]
    )
    zero_delay_descent = top1_descent.copy()
    zero_delay_descent[selected] = 0.0
    oracle_descent = oracle_ce_logit_descent(task_probability, labels)
    alignments = {
        "top1": rowwise_oracle_alignment(top1_descent, oracle_descent),
        "candidate": rowwise_oracle_alignment(candidate_descent, oracle_descent),
        "zero_delay": rowwise_oracle_alignment(zero_delay_descent, oracle_descent),
    }
    delta_vs_top1 = (
        alignments["candidate"]["first_order"]
        - alignments["top1"]["first_order"]
    )
    delta_vs_zero = (
        alignments["candidate"]["first_order"]
        - alignments["zero_delay"]["first_order"]
    )
    delta_vs_top1_ci = paired_mean_bootstrap_ci(
        delta_vs_top1[agreement], seed=2020
    )
    delta_vs_zero_ci = paired_mean_bootstrap_ci(
        delta_vs_zero[agreement], seed=2021
    )
    class_mass_shift = (
        candidate_descent[agreement].mean(axis=0)
        - top1_descent[agreement].mean(axis=0)
    ) * 100.0
    max_mass_shift = float(np.abs(class_mass_shift).max())

    candidate_capture = _capture_metrics(selected, wrong)
    confidence_capture = _capture_metrics(confidence_selected, wrong)
    precision_interval = paired_selection_precision_bootstrap_ci(
        selected,
        confidence_selected,
        wrong,
        agreement,
        seed=2022,
    )
    base_error_rate = float(wrong.sum() / agreement.sum())
    selected_candidate_coverage = float(
        candidate_contains_label[selected].mean() * 100.0
    )
    selected_recovery = float(
        recovered.sum() / selected_wrong.sum() * 100.0
    ) if selected_wrong.any() else 0.0

    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_group = agreement & (labels == class_index)
        selected_group = selected & true_group
        class_rows.append({
            "true_class_index": class_index,
            "true_class": class_name,
            "agreement_samples": int(true_group.sum()),
            "agreement_top1_accuracy_pct": float(correct_top1[true_group].mean() * 100.0),
            "selected_samples": int(selected_group.sum()),
            "selected_candidate_coverage_pct": float(
                candidate_contains_label[selected_group].mean() * 100.0
            ) if selected_group.any() else 0.0,
            "selected_top1_errors": int((selected_group & wrong).sum()),
            "selected_runner_up_recovers": int((selected_group & recovered).sum()),
            "candidate_first_order_delta": float(delta_vs_top1[true_group].mean()),
        })
    _write_csv(class_path, class_rows)

    pair_counter: dict[tuple[int, int], dict[str, int]] = {}
    for top1, runner, label in zip(
        common_top1[selected], shared_runner_up[selected], labels[selected]
    ):
        key = (int(top1), int(runner))
        counts = pair_counter.setdefault(
            key, {"samples": 0, "top1_correct": 0, "runner_up_correct": 0, "outside": 0}
        )
        counts["samples"] += 1
        if int(label) == key[0]:
            counts["top1_correct"] += 1
        elif int(label) == key[1]:
            counts["runner_up_correct"] += 1
        else:
            counts["outside"] += 1
    pair_rows = []
    for (top1, runner), counts in sorted(
        pair_counter.items(), key=lambda item: (-item[1]["samples"], item[0])
    ):
        pair_rows.append({
            "top1_index": top1,
            "top1": CLASS_NAMES[top1],
            "runner_up_index": runner,
            "runner_up": CLASS_NAMES[runner],
            **counts,
            "candidate_coverage_pct": float(
                (counts["top1_correct"] + counts["runner_up_correct"])
                / counts["samples"] * 100.0
            ),
        })
    _write_csv(pair_path, pair_rows)

    oracle_rows = []
    for index in np.flatnonzero(agreement):
        oracle_rows.append({
            "index": int(sample_index[index]),
            "label": int(labels[index]),
            "label_name": CLASS_NAMES[int(labels[index])],
            "common_top1": int(common_top1[index]),
            "common_top1_name": CLASS_NAMES[int(common_top1[index])],
            "common_top1_correct": bool(correct_top1[index]),
            "shared_runner_up": int(shared_runner_up[index]),
            "shared_runner_up_name": CLASS_NAMES[int(shared_runner_up[index])],
            "selected": bool(selected[index]),
            "candidate_contains_label": bool(candidate_contains_label[index]),
            "runner_up_recovers_top1_error": bool(recovered[index]),
            "clip_confidence_selected": bool(confidence_selected[index]),
            "top1_first_order": float(alignments["top1"]["first_order"][index]),
            "candidate_first_order": float(
                alignments["candidate"]["first_order"][index]
            ),
            "zero_delay_first_order": float(
                alignments["zero_delay"]["first_order"][index]
            ),
        })
    _write_csv(oracle_path, oracle_rows)

    class_delta = {
        name: float(
            delta_vs_top1[
                agreement & (labels == CLASS_NAMES.index(name))
            ].mean()
        )
        for name in HARD_CLASSES
    }
    nonhard = agreement & ~np.isin(
        labels, [CLASS_NAMES.index(name) for name in HARD_CLASSES]
    )
    nonhard_delta = float(delta_vs_top1[nonhard].mean())
    input_contract_valid = (
        all(input_checks.values()) and target_hash_matches and labels_match_snapshot
    )
    gate = evaluate_shared_runner_up_gate(
        input_contract_valid=input_contract_valid,
        selected_fraction_pct=float(selected.sum() / agreement.sum() * 100.0),
        selected_candidate_coverage_pct=selected_candidate_coverage,
        selected_top1_miss_recovery_pct=selected_recovery,
        delta_vs_top1_ci=delta_vs_top1_ci,
        delta_vs_zero_delay_ci=delta_vs_zero_ci,
        car_first_order_delta=class_delta["car"],
        person_first_order_delta=class_delta["person"],
        truck_first_order_delta=class_delta["truck"],
        nonhard_first_order_delta=nonhard_delta,
        max_full_mass_shift_pp=max_mass_shift,
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
            "passed": input_contract_valid,
            "checks": {
                **input_checks,
                "target_list_hash_matches_after_lock": target_hash_matches,
                "target_labels_match_embedded_snapshot_after_lock": labels_match_snapshot,
            },
        },
        "label_free_metrics": lock["label_free_metrics"],
        "oracle_metrics": {
            "agreement_samples": int(agreement.sum()),
            "agreement_errors": int(wrong.sum()),
            "agreement_error_rate_pct": float(base_error_rate * 100.0),
            "selected_samples": int(selected.sum()),
            "selected_fraction_pct": float(selected.sum() / agreement.sum() * 100.0),
            "selected_errors": int(selected_wrong.sum()),
            "selected_error_enrichment": float(
                candidate_capture["selection_error_precision_pct"] / 100.0
                / base_error_rate
            ),
            "selected_candidate_coverage_pct": selected_candidate_coverage,
            "selected_top1_miss_recovery_pct": selected_recovery,
            "shared_runner_up_selector": candidate_capture,
            "matched_low_clip_confidence_selector": confidence_capture,
            "shared_minus_confidence_precision_pp": (
                candidate_capture["selection_error_precision_pct"]
                - confidence_capture["selection_error_precision_pct"]
            ),
            "shared_minus_confidence_precision_95_ci_pp": list(precision_interval),
            "alignments": {
                name: _alignment_summary(alignment, agreement)
                for name, alignment in alignments.items()
            },
            "candidate_delta_vs_top1": float(delta_vs_top1[agreement].mean()),
            "candidate_delta_vs_top1_ci": list(delta_vs_top1_ci),
            "candidate_delta_vs_zero_delay": float(delta_vs_zero[agreement].mean()),
            "candidate_delta_vs_zero_delay_ci": list(delta_vs_zero_ci),
            "hard_class_first_order_delta": class_delta,
            "other_nine_first_order_delta": nonhard_delta,
            "class_mass_shift_pp": {
                CLASS_NAMES[index]: float(value)
                for index, value in enumerate(class_mass_shift)
            },
            "max_full_mass_shift_pp": max_mass_shift,
            "classwise": class_rows,
            "pairwise": pair_rows,
        },
        "gate": gate,
        "scope_limit": (
            "PASS authorizes only review of one partial-label proxy design. "
            "It never authorizes or starts proxy/full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote label-free shared top-2 sets: {signal_path}")
    print(f"Locked sets before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote pairwise oracle diagnostic: {pair_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
