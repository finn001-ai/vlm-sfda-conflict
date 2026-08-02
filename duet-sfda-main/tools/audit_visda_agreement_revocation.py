#!/usr/bin/env python3
"""CPU-only audit of revoking stale DUET agreements at cycle 2.

DUET's production mask is monotonic: a cycle-1 agreement remains admitted even
when task and CLIP disagree at cycle 2.  Phase 1 locks those stale admissions
and four matched confidence-removal comparators without reading target labels.
Phase 2 uses labels only for oracle accuracy and exact logit-gradient diagnosis.
No model, image, optimizer, backward pass, or training is involved.
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
    paired_mean_bootstrap_ci,
    weighted_logit_alignment,
)
from src.utils.agreement_rank_residual_audit import (  # noqa: E402
    paired_selection_precision_bootstrap_ci,
    select_matched_counts,
)
from src.utils.agreement_revocation_audit import (  # noqa: E402
    evaluate_agreement_revocation_gate,
    normalized_mask_weight,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_CYCLE1_AGREEMENTS = 6_777
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
    snapshot_dir = DEFAULT_BASE / "cycle2_conflict_memory_snapshots"
    parser.add_argument(
        "--pre-cycle1", type=Path, default=snapshot_dir / "pre_cycle01.npz"
    )
    parser.add_argument(
        "--pre-cycle2", type=Path, default=snapshot_dir / "pre_cycle02.npz"
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
        default=DEFAULT_BASE / "agreement_revocation_audit",
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


def _accuracy(prediction: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    return float((prediction[selected] == labels[selected]).mean() * 100.0)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["oracle_metrics"]
    lines = [
        "# VisDA Agreement Revocation Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "Cycle-2 stale admissions and confidence comparators were locked before",
        "oracle labels were parsed. No training or model execution was performed.",
        "",
        "## Result",
        "",
        f"- Monotonic-mask admitted samples: `{metrics['baseline_admitted']}`.",
        f"- Stale cycle-1 agreements: `{metrics['stale_samples']}` ",
        f"(`{metrics['stale_fraction_of_admitted_pct']:.6f}%`).",
        f"- Stale error precision: ",
        f"`{metrics['stale_selector']['selection_error_precision_pct']:.6f}%`.",
        f"- Retained pseudo-label accuracy gain: ",
        f"`{metrics['retained_accuracy_gain_pp']:.6f} pp`.",
        f"- First-order delta versus monotonic mask: ",
        f"`{metrics['first_order_delta_vs_baseline']:.9f}`; ",
        f"95% CI `{metrics['first_order_delta_vs_baseline_ci']}`.",
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend([
        "",
        "PASS authorizes design review only; it never starts proxy/full training.",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (
        args.pre_cycle1, args.pre_cycle2, args.source_lock,
        args.target_list, args.class_names,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing agreement revocation input: {path}")
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
    pre1_sha256 = _sha256(args.pre_cycle1)
    pre2_sha256 = _sha256(args.pre_cycle2)

    required = {
        "cycle", "mix_label", "label_mask", "source_label", "clip_label",
        "task_prob", "clip_prob", "sample_index", "target_label",
    }
    snapshots: list[dict[str, np.ndarray | int]] = []
    for path in (args.pre_cycle1, args.pre_cycle2):
        with np.load(path, allow_pickle=False) as snapshot:
            missing = required.difference(snapshot.files)
            if missing:
                raise ValueError(f"Snapshot is missing keys: {sorted(missing)}")
            snapshots.append({
                "cycle": int(np.asarray(snapshot["cycle"]).item()),
                "mix_label": np.asarray(snapshot["mix_label"], dtype=np.int64).copy(),
                "label_mask": np.asarray(snapshot["label_mask"], dtype=bool).copy(),
                "source_label": np.asarray(snapshot["source_label"], dtype=np.int64).copy(),
                "clip_label": np.asarray(snapshot["clip_label"], dtype=np.int64).copy(),
                "task_prob": np.asarray(snapshot["task_prob"], dtype=np.float64).copy(),
                "clip_prob": np.asarray(snapshot["clip_prob"], dtype=np.float64).copy(),
                "sample_index": np.asarray(snapshot["sample_index"], dtype=np.int64).copy(),
            })
    pre1, pre2 = snapshots
    index1 = np.asarray(pre1["sample_index"])
    index2 = np.asarray(pre2["sample_index"])
    mask1 = np.asarray(pre1["label_mask"])
    mask2 = np.asarray(pre2["label_mask"])
    task1_label = np.asarray(pre1["source_label"])
    clip1_label = np.asarray(pre1["clip_label"])
    task2_label = np.asarray(pre2["source_label"])
    clip2_label = np.asarray(pre2["clip_label"])
    task2_probability = np.asarray(pre2["task_prob"])
    clip2_probability = np.asarray(pre2["clip_prob"])
    mix2_label = np.asarray(pre2["mix_label"])
    cycle2_agreement = task2_label == clip2_label
    expected_monotonic_mask = mask1 | (~mask1 & cycle2_agreement)
    stale = mask1 & ~cycle2_agreement
    reversible_mask = cycle2_agreement

    row = np.arange(EXPECTED_SAMPLES)
    task_confidence = task2_probability[row, mix2_label]
    clip_confidence = clip2_probability[row, mix2_label]
    arithmetic_probability = 0.5 * (task2_probability + clip2_probability)
    arithmetic_confidence = arithmetic_probability[row, mix2_label]
    rms_probability = np.sqrt(
        (np.square(task2_probability) + np.square(clip2_probability)) / 2.0
    )
    rms_probability /= rms_probability.sum(axis=1, keepdims=True)
    rms_confidence = rms_probability[row, mix2_label]
    stale_counts = {
        class_index: int((stale & (mix2_label == class_index)).sum())
        for class_index in range(EXPECTED_CLASSES)
    }
    confidence_selected = {
        "task_confidence": select_matched_counts(
            task_confidence, mask2, mix2_label, stale_counts, largest=False
        ),
        "clip_confidence": select_matched_counts(
            clip_confidence, mask2, mix2_label, stale_counts, largest=False
        ),
        "arithmetic_confidence": select_matched_counts(
            arithmetic_confidence, mask2, mix2_label, stale_counts, largest=False
        ),
        "rms_confidence": select_matched_counts(
            rms_confidence, mask2, mix2_label, stale_counts, largest=False
        ),
    }
    confidence_retain = {
        name: mask2 & ~selected for name, selected in confidence_selected.items()
    }
    input_checks = {
        "pre_cycle1_matches_source_lock": (
            pre1_sha256 == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
        ),
        "pre_cycle2_matches_source_lock": (
            pre2_sha256 == source_lock.get("inputs", {}).get("pre_cycle2_sha256")
        ),
        "source_lock_declares_labels_after_manifest": bool(
            source_lock.get("labels_read_after_this_manifest")
        ),
        "cycles_are_one_and_two": pre1["cycle"] == 1 and pre2["cycle"] == 2,
        "sample_indices_align_and_match_proxy_order": (
            np.array_equal(index1, index2)
            and np.array_equal(index2, np.arange(EXPECTED_SAMPLES))
        ),
        "expected_probability_shapes": (
            task2_probability.shape == clip2_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "cycle1_mask_equals_cycle1_agreement": np.array_equal(
            mask1, task1_label == clip1_label
        ),
        "expected_cycle1_agreement_count": (
            int(mask1.sum()) == EXPECTED_CYCLE1_AGREEMENTS
        ),
        "cycle2_mask_reproduces_production_monotonic_rule": np.array_equal(
            mask2, expected_monotonic_mask
        ),
        "stale_admissions_nonempty": bool(stale.any()),
        "reversible_mask_is_strict_subset_of_monotonic_mask": bool(
            np.all(~reversible_mask | mask2) and reversible_mask.sum() < mask2.sum()
        ),
        "matched_confidence_counts_equal_stale_by_pseudo_class": all(
            all(
                int((selected & (mix2_label == class_index)).sum()) == count
                for class_index, count in stale_counts.items()
            )
            for selected in confidence_selected.values()
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Agreement revocation input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_agreement_revocation"
    signal_path = args.output_dir / f"{stem}_label_free.npz"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    pair_path = args.output_dir / f"{stem}_pairwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"
    np.savez_compressed(
        signal_path,
        index=index2,
        monotonic_mask=mask2,
        current_agreement=cycle2_agreement,
        stale_admission=stale,
        reversible_mask=reversible_mask,
        current_mix_label=mix2_label,
        task_confidence_selected=confidence_selected["task_confidence"],
        clip_confidence_selected=confidence_selected["clip_confidence"],
        arithmetic_confidence_selected=confidence_selected["arithmetic_confidence"],
        rms_confidence_selected=confidence_selected["rms_confidence"],
    )
    lock = {
        "phase": "LABEL_FREE_AGREEMENT_REVOCATION_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "snapshot_target_label_keys_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "scope": "cycle1 admissions that are task/CLIP conflicts at cycle2",
            "candidate": "replace monotonic admission mask with current agreement mask",
            "hard_pseudo_label": "unchanged current arithmetic-mix top1",
            "clip_kl": "unchanged",
            "consistency": "unchanged",
            "target_label_thresholds": False,
            "class_specific_rules": False,
            "fitted_parameters": False,
            "training_change_in_this_audit": False,
        },
        "trajectory_provenance": {
            "source": "support-conditioned CLIP cycle2 failure-audit snapshots",
            "cycle1_changes_only_initial_conflict_KL_targets": True,
            "known_cycle1_replay_max_accuracy_error_pp": 0.06,
            "limitation": "not an exact matched arithmetic-DUET cycle2 snapshot",
        },
        "predeclared_gate": {
            "stale_fraction_of_admitted_pct": [1.0, 30.0],
            "min_stale_error_enrichment": 2.0,
            "paired_precision_ci_lower_vs_all_confidence_baselines": "> 0",
            "min_retained_accuracy_gain_pp": 0.25,
            "paired_first_order_ci_lower_vs_monotonic_and_confidence": "> 0",
            "car_person_truck_other9_first_order_delta": ">= 0",
            "max_class_mass_shift_pp": 1.0,
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": {
            "samples": EXPECTED_SAMPLES,
            "cycle1_admitted": int(mask1.sum()),
            "cycle2_monotonic_admitted": int(mask2.sum()),
            "cycle2_current_agreements": int(cycle2_agreement.sum()),
            "stale_admissions": int(stale.sum()),
            "stale_fraction_of_admitted_pct": float(stale.sum() / mask2.sum() * 100.0),
            "stale_by_current_pseudo_class": {
                CLASS_NAMES[key]: value for key, value in stale_counts.items()
            },
        },
        "inputs": {
            "pre_cycle1": {"path": str(args.pre_cycle1), "sha256": pre1_sha256},
            "pre_cycle2": {"path": str(args.pre_cycle2), "sha256": pre2_sha256},
            "cycle2_signal_lock": {
                "path": str(args.source_lock), "sha256": _sha256(args.source_lock)
            },
            "target_list_opaque_sha256": _sha256(args.target_list),
            "class_names_sha256": _sha256(args.class_names),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/agreement_revocation_audit.py": _sha256(
                REPO_ROOT / "src/utils/agreement_revocation_audit.py"
            ),
            "tools/audit_visda_agreement_revocation.py": _sha256(Path(__file__).resolve()),
            "src/methods/oh/plmatch.py": _sha256(
                REPO_ROOT / "src/methods/oh/plmatch.py"
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    target_hash_matches = (
        _sha256(args.target_list) == lock["inputs"]["target_list_opaque_sha256"]
    )
    target_labels = _parse_labels_after_lock(args.target_list)
    with np.load(args.pre_cycle1, allow_pickle=False) as snapshot1:
        labels1 = np.asarray(snapshot1["target_label"], dtype=np.int64).copy()
    with np.load(args.pre_cycle2, allow_pickle=False) as snapshot2:
        labels2 = np.asarray(snapshot2["target_label"], dtype=np.int64).copy()
    labels_match_snapshots = (
        np.array_equal(labels1, labels2)
        and np.array_equal(target_labels[index2], labels2)
    )
    labels = target_labels[index2]
    correct = mix2_label == labels
    wrong = mask2 & ~correct
    base_error_rate = float(wrong.sum() / mask2.sum())
    stale_metrics = _capture_metrics(stale, wrong)
    confidence_metrics = {
        name: _capture_metrics(selected, wrong)
        for name, selected in confidence_selected.items()
    }
    selection_comparisons = {}
    for offset, (name, selected) in enumerate(confidence_selected.items()):
        selection_comparisons[name] = {
            "captured_error_gain": (
                stale_metrics["wrong_captured"]
                - confidence_metrics[name]["wrong_captured"]
            ),
            "selection_precision_gain_pp": (
                stale_metrics["selection_error_precision_pct"]
                - confidence_metrics[name]["selection_error_precision_pct"]
            ),
            "paired_bootstrap_95_ci_pp": list(
                paired_selection_precision_bootstrap_ci(
                    stale, selected, wrong, mask2, seed=2020 + offset
                )
            ),
        }

    pseudo_descent = ce_logit_descent(task2_probability, mix2_label)
    oracle_descent = ce_logit_descent(task2_probability, labels)
    masks = {
        "monotonic": mask2,
        "reversible": reversible_mask,
        **{f"revoke_{name}": value for name, value in confidence_retain.items()},
    }
    alignments = {
        name: weighted_logit_alignment(
            pseudo_descent,
            oracle_descent,
            normalized_mask_weight(mask),
            np.ones(EXPECTED_SAMPLES, dtype=bool),
        )
        for name, mask in masks.items()
    }
    candidate_score = alignments["reversible"]["row_first_order"]
    baseline_score = alignments["monotonic"]["row_first_order"]
    delta_vs_baseline = candidate_score - baseline_score
    delta_vs_baseline_ci = paired_mean_bootstrap_ci(
        candidate_score,
        baseline_score,
        np.ones(EXPECTED_SAMPLES, dtype=bool),
        seed=2030,
    )
    first_order_comparisons = {}
    for offset, name in enumerate(confidence_selected):
        reference_score = alignments[f"revoke_{name}"]["row_first_order"]
        first_order_comparisons[name] = {
            "mean_delta": float((candidate_score - reference_score).mean()),
            "paired_bootstrap_95_ci": list(
                paired_mean_bootstrap_ci(
                    candidate_score,
                    reference_score,
                    np.ones(EXPECTED_SAMPLES, dtype=bool),
                    seed=2040 + offset,
                )
            ),
        }

    baseline_accuracy = _accuracy(mix2_label, labels, mask2)
    retained_accuracy = _accuracy(mix2_label, labels, reversible_mask)
    candidate_weight = normalized_mask_weight(reversible_mask)
    baseline_weight = normalized_mask_weight(mask2)
    candidate_descent = pseudo_descent * candidate_weight[:, None]
    baseline_descent = pseudo_descent * baseline_weight[:, None]
    class_mass_shift = (candidate_descent - baseline_descent).mean(axis=0) * 100.0
    max_mass_shift = float(np.abs(class_mass_shift).max())

    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_group = labels == class_index
        admitted_group = mask2 & true_group
        stale_group = stale & true_group
        class_rows.append({
            "true_class_index": class_index,
            "true_class": class_name,
            "monotonic_admitted": int(admitted_group.sum()),
            "monotonic_accuracy_pct": _accuracy(
                mix2_label, labels, admitted_group
            ),
            "stale_samples": int(stale_group.sum()),
            "stale_error_precision_pct": float(
                (~correct[stale_group]).mean() * 100.0
            ) if stale_group.any() else 0.0,
            "reversible_first_order_delta": float(delta_vs_baseline[true_group].mean()),
        })
    _write_csv(class_path, class_rows)

    pair_counter: dict[tuple[int, int], dict[str, int]] = {}
    for task_label, clip_label, is_correct in zip(
        task2_label[stale], clip2_label[stale], correct[stale]
    ):
        key = (int(task_label), int(clip_label))
        counts = pair_counter.setdefault(key, {"samples": 0, "errors": 0})
        counts["samples"] += 1
        counts["errors"] += int(not is_correct)
    pair_rows = []
    for (task_label, clip_label), counts in sorted(
        pair_counter.items(), key=lambda item: (-item[1]["samples"], item[0])
    ):
        pair_rows.append({
            "cycle2_task_top1_index": task_label,
            "cycle2_task_top1": CLASS_NAMES[task_label],
            "cycle2_clip_top1_index": clip_label,
            "cycle2_clip_top1": CLASS_NAMES[clip_label],
            **counts,
            "error_precision_pct": float(counts["errors"] / counts["samples"] * 100.0),
        })
    _write_csv(pair_path, pair_rows)

    oracle_rows = []
    for index in np.flatnonzero(mask2):
        oracle_rows.append({
            "index": int(index2[index]),
            "label": int(labels[index]),
            "label_name": CLASS_NAMES[int(labels[index])],
            "mix_pseudo_label": int(mix2_label[index]),
            "mix_pseudo_label_name": CLASS_NAMES[int(mix2_label[index])],
            "pseudo_label_correct": bool(correct[index]),
            "cycle1_admitted": bool(mask1[index]),
            "cycle2_current_agreement": bool(cycle2_agreement[index]),
            "stale_admission": bool(stale[index]),
            "cycle2_task_top1": int(task2_label[index]),
            "cycle2_clip_top1": int(clip2_label[index]),
            "task_confidence_selected": bool(
                confidence_selected["task_confidence"][index]
            ),
            "clip_confidence_selected": bool(
                confidence_selected["clip_confidence"][index]
            ),
            "arithmetic_confidence_selected": bool(
                confidence_selected["arithmetic_confidence"][index]
            ),
            "rms_confidence_selected": bool(
                confidence_selected["rms_confidence"][index]
            ),
            "monotonic_first_order": float(baseline_score[index]),
            "reversible_first_order": float(candidate_score[index]),
        })
    _write_csv(oracle_path, oracle_rows)

    hard_class_delta = {
        name: float(
            delta_vs_baseline[labels == CLASS_NAMES.index(name)].mean()
        )
        for name in HARD_CLASSES
    }
    nonhard = ~np.isin(labels, [CLASS_NAMES.index(name) for name in HARD_CLASSES])
    nonhard_delta = float(delta_vs_baseline[nonhard].mean())
    input_contract_valid = (
        all(input_checks.values()) and target_hash_matches and labels_match_snapshots
    )
    gate = evaluate_agreement_revocation_gate(
        input_contract_valid=input_contract_valid,
        stale_fraction_of_admitted_pct=float(stale.sum() / mask2.sum() * 100.0),
        stale_error_enrichment=(
            stale_metrics["selection_error_precision_pct"] / 100.0 / base_error_rate
        ),
        captured_error_gains={
            name: values["captured_error_gain"]
            for name, values in selection_comparisons.items()
        },
        precision_gain_cis={
            name: tuple(values["paired_bootstrap_95_ci_pp"])
            for name, values in selection_comparisons.items()
        },
        retained_accuracy_gain_pp=retained_accuracy - baseline_accuracy,
        first_order_delta_vs_baseline_ci=delta_vs_baseline_ci,
        first_order_delta_vs_confidence_cis={
            name: tuple(values["paired_bootstrap_95_ci"])
            for name, values in first_order_comparisons.items()
        },
        car_first_order_delta=hard_class_delta["car"],
        person_first_order_delta=hard_class_delta["person"],
        truck_first_order_delta=hard_class_delta["truck"],
        nonhard_first_order_delta=nonhard_delta,
        max_class_mass_shift_pp=max_mass_shift,
    )
    serializable_alignments = {
        name: {key: value for key, value in result.items() if key != "row_first_order"}
        for name, result in alignments.items()
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
        "trajectory_provenance": lock["trajectory_provenance"],
        "input_contract": {
            "passed": input_contract_valid,
            "checks": {
                **input_checks,
                "target_list_hash_matches_after_lock": target_hash_matches,
                "target_labels_match_both_snapshots_after_lock": labels_match_snapshots,
            },
        },
        "label_free_metrics": lock["label_free_metrics"],
        "oracle_metrics": {
            "baseline_admitted": int(mask2.sum()),
            "baseline_errors": int(wrong.sum()),
            "baseline_error_rate_pct": float(base_error_rate * 100.0),
            "baseline_accuracy_pct": baseline_accuracy,
            "stale_samples": int(stale.sum()),
            "stale_fraction_of_admitted_pct": float(stale.sum() / mask2.sum() * 100.0),
            "stale_selector": stale_metrics,
            "stale_error_enrichment": float(
                stale_metrics["selection_error_precision_pct"] / 100.0 / base_error_rate
            ),
            "confidence_selectors": confidence_metrics,
            "selection_comparisons": selection_comparisons,
            "reversible_retained": int(reversible_mask.sum()),
            "retained_accuracy_pct": retained_accuracy,
            "retained_accuracy_gain_pp": retained_accuracy - baseline_accuracy,
            "alignments": serializable_alignments,
            "first_order_delta_vs_baseline": float(delta_vs_baseline.mean()),
            "first_order_delta_vs_baseline_ci": list(delta_vs_baseline_ci),
            "first_order_comparisons": first_order_comparisons,
            "hard_class_first_order_delta": hard_class_delta,
            "other_nine_first_order_delta": nonhard_delta,
            "class_mass_shift_pp": {
                CLASS_NAMES[index]: float(value)
                for index, value in enumerate(class_mass_shift)
            },
            "max_class_mass_shift_pp": max_mass_shift,
            "classwise": class_rows,
            "pairwise": pair_rows,
        },
        "gate": gate,
        "scope_limit": (
            "PASS authorizes only review of one reversible-mask proxy design. "
            "It never authorizes or starts proxy/full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))
    print(f"Wrote label-free revocation masks: {signal_path}")
    print(f"Locked masks before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote pairwise oracle diagnostic: {pair_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
