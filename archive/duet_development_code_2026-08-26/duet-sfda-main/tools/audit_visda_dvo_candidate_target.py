#!/usr/bin/env python3
"""CPU-only audit of a candidate-set soft target for DUET CLIP DVO/TMI.

Phase 1 reads the locked pre-cycle-1 probabilities, constructs a top-1-
preserving support-conditioned mixed target for task/CLIP conflicts, and
replays output-level TMI gradients across fixed batch permutations.  It locks
all signals before Phase 2 parses target labels for oracle diagnostics.  No
image, model, checkpoint, optimizer, parameter update, or training is used.
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

from src.utils.candidate_set_gradient_audit import (  # noqa: E402
    oracle_ce_logit_descent,
    paired_mean_bootstrap_ci,
    rowwise_oracle_alignment,
)
from src.utils.dvo_candidate_target_audit import (  # noqa: E402
    evaluate_dvo_candidate_target_gate,
    support_conditioned_mixed_target,
    tmi_logit_descent_replays,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_AGREEMENTS = 6_777
BATCH_SIZE = 64
INITIAL_Q = 1.05
BETA = 0.99
PERMUTATION_SEEDS = tuple(range(2_020, 2_028))
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
            DEFAULT_BASE / "cycle2_conflict_memory_audit/"
            "visda_cycle2_conflict_memory_signal_lock.json"
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
        default=DEFAULT_BASE / "dvo_candidate_target_audit",
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
            raise ValueError(
                f"Malformed target row {line_number}: {stripped}"
            ) from error
    result = np.asarray(labels, dtype=np.int64)
    if result.shape != (EXPECTED_SAMPLES,):
        raise ValueError(f"Expected {EXPECTED_SAMPLES} labels, found {result.size}")
    if np.any(result < 0) or np.any(result >= EXPECTED_CLASSES):
        raise ValueError("Target label is outside the class range")
    return result


def _comparison(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result = {}
    for offset, metric in enumerate(
        ("cosine", "oracle_unit_projection", "first_order")
    ):
        difference = candidate[metric][mask] - baseline[metric][mask]
        result[metric] = {
            "mean_difference": float(difference.mean()),
            "paired_bootstrap_95_ci": list(
                paired_mean_bootstrap_ci(difference, repeats=2_000, seed=2_020 + offset)
            ),
        }
    return result


def _negative_burden(first_order: np.ndarray, mask: np.ndarray) -> float:
    return float(np.maximum(-first_order[mask], 0.0).mean())


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["oracle_metrics"]
    lines = [
        "# VisDA DUET DVO Candidate-Target Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The candidate target and output-level TMI descents were locked before",
        "oracle labels were parsed. No image, model, checkpoint, optimizer,",
        "parameter update, backward-through-model pass, or training was run.",
        "",
        "## Candidate",
        "",
        "For cycle-1 task/CLIP top-1 conflicts only, condition DUET's arithmetic",
        "mixed DVO target on the task/CLIP top-2 union. The original mixed top-1",
        "is always included, so the hard target never changes. PLMatch CE, CLIP",
        "KL, consistency, agreement memory, and loss weights remain untouched.",
        "",
        "## Oracle diagnostic",
        "",
        f"- Conflict first-order delta: `{metrics['comparisons']['first_order']['mean_difference']:.9f}`.",
        f"- 95% CI: `{metrics['comparisons']['first_order']['paired_bootstrap_95_ci']}`.",
        f"- Minimum replay delta: `{metrics['minimum_replay_first_order_delta']:.9f}`.",
        f"- Candidate negative burden: `{metrics['candidate_negative_burden']:.9f}`.",
        f"- Baseline negative burden: `{metrics['baseline_negative_burden']:.9f}`.",
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
            "PASS authorizes review of one matched proxy design only. It never starts",
            "or authorizes proxy/full VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (args.snapshot, args.source_lock, args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing DVO candidate-target input: {path}")
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

    # Phase 1: target_label is required for later verification but not accessed.
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
        task_label = np.asarray(snapshot["source_label"], dtype=np.int64).copy()
        clip_label = np.asarray(snapshot["clip_label"], dtype=np.int64).copy()
        task_probability = np.asarray(snapshot["task_prob"], dtype=np.float64).copy()
        clip_probability = np.asarray(snapshot["clip_prob"], dtype=np.float64).copy()
        sample_index = np.asarray(snapshot["sample_index"], dtype=np.int64).copy()

    agreement = task_label == clip_label
    active_conflict = ~agreement
    target = support_conditioned_mixed_target(
        task_probability, clip_probability, active_conflict
    )
    baseline_replay = tmi_logit_descent_replays(
        clip_probability,
        target["baseline_probability"],
        permutation_seeds=PERMUTATION_SEEDS,
        batch_size=BATCH_SIZE,
        initial_q=INITIAL_Q,
        beta=BETA,
    )
    candidate_replay = tmi_logit_descent_replays(
        clip_probability,
        target["candidate_probability"],
        permutation_seeds=PERMUTATION_SEEDS,
        batch_size=BATCH_SIZE,
        initial_q=INITIAL_Q,
        beta=BETA,
    )
    baseline_mass = target["baseline_probability"].mean(axis=0)
    candidate_mass = target["candidate_probability"].mean(axis=0)
    class_mass_shift_pp = (candidate_mass - baseline_mass) * 100.0
    active_support = target["support"][active_conflict]
    input_checks = {
        "source_snapshot_matches_cycle2_signal_lock": (
            snapshot_sha256 == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
        ),
        "source_lock_declares_labels_after_manifest": bool(
            source_lock.get("labels_read_after_this_manifest")
        ),
        "cycle_is_one": cycle == 1,
        "expected_probability_shape": (
            task_probability.shape
            == clip_probability.shape
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
            np.array_equal(task_label, task_probability.argmax(1))
            and np.array_equal(clip_label, clip_probability.argmax(1))
        ),
        "duet_mask_equals_top1_agreement": np.array_equal(label_mask, agreement),
        "expected_agreement_count": int(agreement.sum()) == EXPECTED_AGREEMENTS,
        "candidate_changes_only_conflicts": np.array_equal(
            target["candidate_probability"][agreement],
            target["baseline_probability"][agreement],
        ),
        "candidate_mixed_top1_unchanged": np.array_equal(
            target["candidate_probability"].argmax(1),
            target["baseline_probability"].argmax(1),
        ),
        "tmi_q_trajectory_is_target_independent": np.allclose(
            baseline_replay["final_q"],
            candidate_replay["final_q"],
            atol=1e-12,
            rtol=0.0,
        ),
        "tmi_descents_finite": bool(
            np.isfinite(baseline_replay["descent_by_replay"]).all()
            and np.isfinite(candidate_replay["descent_by_replay"]).all()
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"DVO candidate-target input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_dvo_candidate_target"
    signal_path = args.output_dir / f"{stem}_label_free.npz"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"
    np.savez_compressed(
        signal_path,
        index=sample_index,
        active_conflict=active_conflict,
        baseline_target=target["baseline_probability"].astype(np.float32),
        candidate_target=target["candidate_probability"].astype(np.float32),
        candidate_support=target["support"],
        support_size=target["support_size"],
        retained_mass=target["retained_mass"].astype(np.float32),
        baseline_tmi_descent=baseline_replay["mean_descent"].astype(np.float32),
        candidate_tmi_descent=candidate_replay["mean_descent"].astype(np.float32),
        baseline_tmi_descent_by_replay=(
            baseline_replay["descent_by_replay"].astype(np.float32)
        ),
        candidate_tmi_descent_by_replay=(
            candidate_replay["descent_by_replay"].astype(np.float32)
        ),
        permutation_seeds=baseline_replay["permutation_seeds"],
        final_q=baseline_replay["final_q"],
    )
    lock = {
        "phase": "LABEL_FREE_DVO_CANDIDATE_TARGET_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "target_list_not_parsed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "scope": "cycle1 task/CLIP top1 conflicts only",
            "training_interface": "CLIP visual DVO/TMI soft target only",
            "baseline_target": "arithmetic mean of task and CLIP probabilities",
            "candidate_support": (
                "task top2 union CLIP top2 union original mixed top1"
            ),
            "within_support_values": "renormalized original arithmetic target",
            "mixed_target_top1_changed": False,
            "task_hard_ce_changed": False,
            "task_clip_kl_changed": False,
            "task_consistency_changed": False,
            "agreement_memory_changed": False,
            "loss_weights_changed": False,
            "fitted_parameters": False,
            "target_label_thresholds": False,
            "training_change_in_this_audit": False,
        },
        "tmi_replay_contract": {
            "batch_size": BATCH_SIZE,
            "initial_q": INITIAL_Q,
            "beta": BETA,
            "permutation_seeds": list(PERMUTATION_SEEDS),
            "scope_limit": (
                "output-level replay of locked weak-view probabilities; no image "
                "or parameter-space gradient"
            ),
        },
        "predeclared_gate": {
            "min_mean_retained_mass": 0.90,
            "max_mean_support_size": 4.0,
            "min_oracle_candidate_coverage_pct": 90.0,
            "paired_cosine_projection_first_order_ci_lower": "> 0",
            "every_replay_first_order_delta": "> 0",
            "class_macro_first_order_delta": "> 0",
            "car_person_truck_other9_first_order_delta": ">= 0",
            "negative_burden": "not worse",
            "max_class_mass_shift_pp": 1.0,
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": {
            "samples": EXPECTED_SAMPLES,
            "agreements": int(agreement.sum()),
            "active_conflicts": int(active_conflict.sum()),
            "mean_support_size": float(target["support_size"][active_conflict].mean()),
            "mean_retained_mass": float(target["retained_mass"].mean()),
            "class_mass_shift_pp": {
                name: float(class_mass_shift_pp[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "max_class_mass_shift_pp": float(np.abs(class_mass_shift_pp).max()),
            "baseline_final_q_by_replay": baseline_replay["final_q"].tolist(),
        },
        "inputs": {
            "pre_cycle1_snapshot": {
                "path": str(args.snapshot),
                "sha256": snapshot_sha256,
            },
            "cycle2_signal_lock": {
                "path": str(args.source_lock),
                "sha256": _sha256(args.source_lock),
            },
            "target_list_opaque_sha256": _sha256(args.target_list),
            "class_names_sha256": _sha256(args.class_names),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "src/utils/dvo_candidate_target_audit.py": _sha256(
                REPO_ROOT / "src/utils/dvo_candidate_target_audit.py"
            ),
            "tools/audit_visda_dvo_candidate_target.py": _sha256(
                Path(__file__).resolve()
            ),
            "src/utils/IID_losses.py": _sha256(REPO_ROOT / "src/utils/IID_losses.py"),
            "src/methods/oh/plmatch.py": _sha256(
                REPO_ROOT / "src/methods/oh/plmatch.py"
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: oracle diagnostic, strictly after the signal lock is written.
    target_hash_matches = (
        _sha256(args.target_list) == lock["inputs"]["target_list_opaque_sha256"]
    )
    target_labels = _parse_labels_after_lock(args.target_list)
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        embedded_labels = np.asarray(snapshot["target_label"], dtype=np.int64).copy()
    labels = target_labels[sample_index]
    labels_match_snapshot = np.array_equal(labels, embedded_labels)

    oracle_descent = oracle_ce_logit_descent(clip_probability, labels)
    baseline_alignment = rowwise_oracle_alignment(
        baseline_replay["mean_descent"], oracle_descent
    )
    candidate_alignment = rowwise_oracle_alignment(
        candidate_replay["mean_descent"], oracle_descent
    )
    comparisons = _comparison(candidate_alignment, baseline_alignment, active_conflict)
    delta = candidate_alignment["first_order"] - baseline_alignment["first_order"]
    replay_deltas = []
    for replay_index, seed in enumerate(PERMUTATION_SEEDS):
        baseline_one = rowwise_oracle_alignment(
            baseline_replay["descent_by_replay"][replay_index], oracle_descent
        )
        candidate_one = rowwise_oracle_alignment(
            candidate_replay["descent_by_replay"][replay_index], oracle_descent
        )
        replay_deltas.append(
            {
                "seed": seed,
                "mean_conflict_first_order_delta": float(
                    (
                        candidate_one["first_order"][active_conflict]
                        - baseline_one["first_order"][active_conflict]
                    ).mean()
                ),
            }
        )

    coverage = active_support[
        np.arange(active_support.shape[0]), labels[active_conflict]
    ]
    retained_mass_full = np.ones(EXPECTED_SAMPLES, dtype=np.float64)
    retained_mass_full[active_conflict] = target["retained_mass"]
    class_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = active_conflict & (labels == class_index)
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "conflict_samples": int(mask.sum()),
                "candidate_set_oracle_coverage_pct": float(
                    target["support"][mask, class_index].mean() * 100.0
                ),
                "baseline_tmi_first_order": float(
                    baseline_alignment["first_order"][mask].mean()
                ),
                "candidate_tmi_first_order": float(
                    candidate_alignment["first_order"][mask].mean()
                ),
                "candidate_minus_baseline_first_order": float(delta[mask].mean()),
            }
        )
    _write_csv(class_path, class_rows)

    oracle_rows = []
    for index in np.flatnonzero(active_conflict):
        oracle_rows.append(
            {
                "index": int(sample_index[index]),
                "label": int(labels[index]),
                "label_name": CLASS_NAMES[int(labels[index])],
                "task_top1": int(task_label[index]),
                "clip_top1": int(clip_label[index]),
                "mixed_top1": int(target["mixed_top1"][index]),
                "candidate_set_covers_label": bool(
                    target["support"][index, labels[index]]
                ),
                "support_size": int(target["support_size"][index]),
                "retained_mixed_mass": float(retained_mass_full[index]),
                "baseline_tmi_first_order": float(
                    baseline_alignment["first_order"][index]
                ),
                "candidate_tmi_first_order": float(
                    candidate_alignment["first_order"][index]
                ),
                "candidate_minus_baseline_first_order": float(delta[index]),
            }
        )
    _write_csv(oracle_path, oracle_rows)

    class_delta = {
        row["class"]: row["candidate_minus_baseline_first_order"] for row in class_rows
    }
    macro_delta = float(np.mean(list(class_delta.values())))
    other_nine = float(
        np.mean(
            [value for name, value in class_delta.items() if name not in HARD_CLASSES]
        )
    )
    hard_delta = {name: class_delta[name] for name in HARD_CLASSES}
    baseline_burden = _negative_burden(
        baseline_alignment["first_order"], active_conflict
    )
    candidate_burden = _negative_burden(
        candidate_alignment["first_order"], active_conflict
    )
    input_contract_valid = (
        all(input_checks.values()) and target_hash_matches and labels_match_snapshot
    )
    gate = evaluate_dvo_candidate_target_gate(
        input_contract_valid=input_contract_valid,
        target_top1_unchanged=input_checks["candidate_mixed_top1_unchanged"],
        mean_retained_mass=float(target["retained_mass"].mean()),
        mean_support_size=float(target["support_size"][active_conflict].mean()),
        oracle_candidate_coverage_pct=float(coverage.mean() * 100.0),
        comparisons=comparisons,
        minimum_replay_first_order_delta=min(
            row["mean_conflict_first_order_delta"] for row in replay_deltas
        ),
        macro_first_order_delta=macro_delta,
        hard_class_first_order_delta=hard_delta,
        other_nine_first_order_delta=other_nine,
        candidate_negative_burden=candidate_burden,
        baseline_negative_burden=baseline_burden,
        max_class_mass_shift_pp=float(np.abs(class_mass_shift_pp).max()),
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
        "trajectory_provenance": {
            "source": "support-conditioned CLIP cycle2 failure-audit pre-cycle1 snapshot",
            "limitation": (
                "locked weak-view probabilities approximate the DVO image-test "
                "forward; this is an output-level gate, not a parameter update replay"
            ),
        },
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
            "candidate_set_oracle_coverage_pct": float(coverage.mean() * 100.0),
            "comparisons": comparisons,
            "replay_first_order_deltas": replay_deltas,
            "minimum_replay_first_order_delta": min(
                row["mean_conflict_first_order_delta"] for row in replay_deltas
            ),
            "class_macro_first_order_delta": macro_delta,
            "hard_class_first_order_delta": hard_delta,
            "other_nine_first_order_delta": other_nine,
            "baseline_negative_burden": baseline_burden,
            "candidate_negative_burden": candidate_burden,
            "classwise": class_rows,
        },
        "gate": gate,
        "scope_limit": (
            "PASS authorizes design of one matched proxy only. This audit never "
            "authorizes or starts proxy/full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(
        json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2)
    )
    print(f"Wrote label-free DVO target and TMI descents: {signal_path}")
    print(f"Locked DVO target before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
