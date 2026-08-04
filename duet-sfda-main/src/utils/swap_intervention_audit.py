"""Cycle 2/3 swap-intervention audit (pure diagnostic, read-only).

This module answers, without touching any training signal:

  - how many of the swap candidates actually changed the fused label;
  - how many were newly admitted into the hard-CE mask;
  - the net correction (W2R - R2W) and which class pairs benefit / suffer;
  - full confusion matrices (raw + row-normalized) for the Task / CLIP /
    base-mix / final-mem label streams.

Ground truth is used exclusively for offline statistics; it never feeds the
selection or training path.  All tensors are detached and moved to CPU before
any computation.  No extra forward pass, no BatchNorm state change, and no RNG
consumption happens here.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


ABSTAIN_REASONS = (
    "not_conflict",
    "conflict_non_swap",
    "inactive_cycle",
    "direction_filter_failed",
    "gate_failed",
    "selected_task",
    "selected_clip",
)

CM_SPECS = (
    ("cm01_conflict_task", "Conflict samples: GT -> Task top1"),
    ("cm02_conflict_clip", "Conflict samples: GT -> CLIP top1"),
    ("cm03_swap_candidate_base_mix", "Swap candidates: GT -> base mix label"),
    ("cm04_swap_selected_final", "Swap selected: GT -> final mem label"),
    ("cm05_newly_admitted_final", "Newly admitted: GT -> final mem label"),
    ("cm06_label_changed_direction", "Label changed: base -> final"),
    ("cm07_w2r_correction", "W2R: base(wrong) -> real(corrected)"),
    ("cm08_r2w_damage", "R2W: real -> final(new wrong)"),
    ("cm09_final_mask_base", "Final mask: GT -> base mix label"),
    ("cm10_final_mask_final", "Final mask: GT -> final mem label"),
)


def _cpu(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value


def _numpy(value: Any, dtype: Any = None) -> np.ndarray:
    return np.asarray(_cpu(value), dtype=dtype)


def class_name(class_names: list[str], class_id: int) -> str:
    return class_names[int(class_id)] if 0 <= int(class_id) < len(class_names) else "?"


def unordered_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smaller class id first so car<->truck and truck<->car share a pair."""
    pair = np.stack([a, b], axis=1)
    return np.min(pair, axis=1) * 100 + np.max(pair, axis=1)


def build_swap_audit_payload(
    *,
    cycle: int,
    task_prob: Any,
    clip_prob: Any,
    base_mix_label: Any,
    final_mem_label: Any,
    base_label_mask: Any,
    final_label_mask: Any,
    prev_label_mask: Any,
    current_agreement: Any,
    swap_selected: Any,
    swap_diagnostics: dict[str, np.ndarray],
    real_label: Any,
    sample_index: Any,
    image_paths: list[str],
    class_names: list[str],
    gate_D: float,
    min_direction_accuracy: float,
) -> dict[str, Any]:
    """Assemble the per-sample audit table for one cycle (CPU numpy)."""
    task = _numpy(task_prob, np.float64)
    clip = _numpy(clip_prob, np.float64)
    base_mix = _numpy(base_mix_label, np.int64)
    final_mem = _numpy(final_mem_label, np.int64)
    base_mask = _numpy(base_label_mask, bool)
    final_mask = _numpy(final_label_mask, bool)
    prev_mask = _numpy(prev_label_mask, bool) if prev_label_mask is not None else np.zeros(task.shape[0], bool)
    agreement = _numpy(current_agreement, bool)
    selected = _numpy(swap_selected, bool)
    real = _numpy(real_label, np.int64)
    sample_idx = _numpy(sample_index, np.int64)
    d = swap_diagnostics
    n = task.shape[0]

    task_top1 = d["task_top1"]
    task_top2 = d["task_top2"]
    clip_top1 = d["clip_top1"]
    clip_top2 = d["clip_top2"]
    pA = d["task_top1_prob"]
    pB = d["task_top2_prob"]
    qB = d["clip_top1_prob"]
    qA = d["clip_top2_prob"]

    task_top1_prob = np.take_along_axis(task, task_top1[:, None], axis=1)[:, 0]
    task_top2_prob = np.take_along_axis(task, task_top2[:, None], axis=1)[:, 0]
    clip_top1_prob = np.take_along_axis(clip, clip_top1[:, None], axis=1)[:, 0]
    clip_top2_prob = np.take_along_axis(clip, clip_top2[:, None], axis=1)[:, 0]
    task_margin = task_top1_prob - task_top2_prob
    clip_margin = clip_top1_prob - clip_top2_prob

    def entropy(prob: np.ndarray) -> np.ndarray:
        clipped = np.clip(prob, 1e-12, 1.0)
        return -np.sum(clipped * np.log(clipped), axis=1)

    task_entropy = entropy(task)
    clip_entropy = entropy(clip)

    is_agreement = task_top1 == clip_top1
    is_conflict = d["is_conflict"]
    is_swap = d["is_swap_candidate"]
    already_admitted = selected & base_mask
    newly_admitted = selected & ~base_mask
    label_changed = selected & (final_mem != base_mix)
    base_correct = base_mix == real
    final_correct = final_mem == real
    assert np.all(newly_admitted <= selected)
    assert np.all(label_changed <= selected)
    assert np.array_equal(final_mask, base_mask | selected)

    # swap proposed label = final mem label on selected samples.
    proposed = np.full(n, -1, dtype=np.int64)
    proposed[selected] = final_mem[selected]

    correction_type = np.full(n, "UNCHANGED", dtype=object)
    changed = label_changed
    correction_type[changed & ~base_correct & final_correct] = "W2R"
    correction_type[changed & base_correct & ~final_correct] = "R2W"
    correction_type[changed & base_correct & final_correct] = "R2R"
    correction_type[changed & ~base_correct & ~final_correct] = "W2W"

    selected_source = np.full(n, "abstain", dtype=object)
    selected_source[selected & d["choose_task"]] = "task"
    selected_source[selected & d["choose_clip"]] = "clip"

    names = [""] * n
    for i in range(n):
        names[i] = image_paths[int(sample_idx[i])] if 0 <= int(sample_idx[i]) < len(image_paths) else ""

    payload: dict[str, Any] = {
        "cycle": int(cycle),
        "total_samples": n,
        "class_names": list(class_names),
        "gate_D": float(gate_D),
        "min_direction_accuracy": float(min_direction_accuracy),
        "sample_index": sample_idx,
        "image_path": np.asarray(names, dtype=object),
        "real_label": real,
        "real_class_name": np.asarray(
            [class_name(class_names, v) for v in real], dtype=object
        ),
        "task_top1": task_top1,
        "task_top1_name": np.asarray(
            [class_name(class_names, v) for v in task_top1], dtype=object
        ),
        "task_top1_prob": task_top1_prob,
        "task_top2": task_top2,
        "task_top2_name": np.asarray(
            [class_name(class_names, v) for v in task_top2], dtype=object
        ),
        "task_top2_prob": task_top2_prob,
        "task_margin": task_margin,
        "task_entropy": task_entropy,
        "task_correct": task_top1 == real,
        "clip_top1": clip_top1,
        "clip_top1_name": np.asarray(
            [class_name(class_names, v) for v in clip_top1], dtype=object
        ),
        "clip_top1_prob": clip_top1_prob,
        "clip_top2": clip_top2,
        "clip_top2_name": np.asarray(
            [class_name(class_names, v) for v in clip_top2], dtype=object
        ),
        "clip_top2_prob": clip_top2_prob,
        "clip_margin": clip_margin,
        "clip_entropy": clip_entropy,
        "clip_correct": clip_top1 == real,
        "is_agreement": is_agreement,
        "is_conflict": is_conflict,
        "is_swap_candidate": is_swap,
        "candidate_A": d["candidate_A"],
        "candidate_A_name": np.asarray(
            [class_name(class_names, v) for v in d["candidate_A"]], dtype=object
        ),
        "candidate_B": d["candidate_B"],
        "candidate_B_name": np.asarray(
            [class_name(class_names, v) for v in d["candidate_B"]], dtype=object
        ),
        "unordered_pair": unordered_pair(task_top1, clip_top1),
        "task_evidence": d["task_evidence"],
        "clip_evidence": d["clip_evidence"],
        "log_task_evidence": d["log_task_evidence"],
        "log_clip_evidence": d["log_clip_evidence"],
        "signed_log_gap": d["signed_log_gap"],
        "absolute_log_gap": d["absolute_log_gap"],
        "passed_gate": d["passed_gate"],
        "passed_direction_filter": d["passed_direction_filter"],
        "swap_selected": selected,
        "selected_source": selected_source,
        "swap_proposed_label": proposed,
        "swap_proposed_name": np.asarray(
            [class_name(class_names, v) for v in proposed], dtype=object
        ),
        "abstain_reason": d["abstain_reason"],
        "prev_label_mask": prev_mask,
        "current_agreement": agreement,
        "base_label_mask": base_mask,
        "already_admitted": already_admitted,
        "newly_admitted": newly_admitted,
        "final_label_mask": final_mask,
        "base_mix_label": base_mix,
        "base_mix_name": np.asarray(
            [class_name(class_names, v) for v in base_mix], dtype=object
        ),
        "final_mem_label": final_mem,
        "final_mem_name": np.asarray(
            [class_name(class_names, v) for v in final_mem], dtype=object
        ),
        "label_changed": label_changed,
        "base_correct": base_correct,
        "final_correct": final_correct,
        "correction_type": correction_type,
    }
    payload["direction_accuracy"] = _direction_accuracy_array(
        task_top1, clip_top1
    )
    return payload


def _direction_accuracy_array(task_top1: np.ndarray, clip_top1: np.ndarray) -> np.ndarray:
    from src.utils.swap_conflict_selection import CYCLE0_DIRECTION_ACCURACY

    return np.asarray(
        [
            CYCLE0_DIRECTION_ACCURACY.get((int(a), int(b)), 0.0)
            for a, b in zip(task_top1, clip_top1)
        ],
        dtype=np.float64,
    )


def correction_stats(payload: dict[str, Any], mask: np.ndarray) -> dict[str, int | float]:
    """W2R/R2W/R2R/W2W counts plus net correction for a given subset."""
    selected = payload["swap_selected"] & mask
    base_correct = payload["base_correct"] & mask
    final_correct = payload["final_correct"] & mask
    w2r = int((selected & ~base_correct & final_correct).sum())
    r2w = int((selected & base_correct & ~final_correct).sum())
    r2r = int((selected & base_correct & final_correct).sum())
    w2w = int((selected & ~base_correct & ~final_correct).sum())
    count = int(selected.sum())
    base_c = int((selected & base_correct).sum())
    final_c = int((selected & final_correct).sum())
    return {
        "sample_count": count,
        "base_correct_count": base_c,
        "final_correct_count": final_c,
        "base_accuracy": 100.0 * base_c / count if count else 0.0,
        "final_accuracy": 100.0 * final_c / count if count else 0.0,
        "accuracy_delta": (
            100.0 * (final_c - base_c) / count if count else 0.0
        ),
        "W2R": w2r,
        "R2W": r2w,
        "R2R": r2r,
        "W2W": w2w,
        "net_correction": w2r - r2w,
        "dataset_level_net_delta": (
            (w2r - r2w) / payload["total_samples"] if payload["total_samples"] else 0.0
        ),
    }


def save_confusion_matrix(
    matrix: np.ndarray,
    *,
    out_dir: Path,
    stem: str,
    title: str,
    class_names: list[str],
) -> None:
    """Save raw + row-normalized confusion matrix as CSV / NPY / PNG."""
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError(
            f"confusion matrix {stem} shape {matrix.shape} does not match "
            f"{len(class_names)} classes"
        )
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(matrix, dtype=np.float64)
    np.divide(
        matrix,
        np.where(row_sum > 0, row_sum, 1),
        out=normalized,
        where=row_sum > 0,
    )
    normalized = normalized * 100.0

    for suffix, data, is_float in (
        ("raw", matrix, False),
        ("row_normalized", normalized, True),
    ):
        csv_path = out_dir / f"{stem}_{suffix}.csv"
        npy_path = out_dir / f"{stem}_{suffix}.npy"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["real_or_row"] + list(class_names))
            for i, row_name in enumerate(class_names):
                if is_float:
                    writer.writerow(
                        [row_name] + [f"{v:.2f}" for v in data[i]]
                    )
                else:
                    writer.writerow([row_name] + [int(v) for v in data[i]])
        np.save(npy_path, data)
        _save_confusion_png(
            matrix if not is_float else normalized,
            out_dir=out_dir,
            stem=f"{stem}_{suffix}",
            title=f"{title} ({suffix})",
            class_names=class_names,
        )
    logging.info(
        "SWAP AUDIT confusion matrix saved: %s (sum=%d)",
        out_dir / stem,
        int(matrix.sum()),
    )


def _save_confusion_png(
    matrix: np.ndarray,
    *,
    out_dir: Path,
    stem: str,
    title: str,
    class_names: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(class_names)), class_names, rotation=90)
    axis.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = matrix[i, j]
            if value == 0:
                continue
            text = f"{value:.0f}" if value == int(value) else f"{value:.1f}"
            axis.text(j, i, text, ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("column label")
    axis.set_ylabel("row label")
    figure.tight_layout()
    figure.savefig(out_dir / f"{stem}.png", dpi=150)
    plt.close(figure)


def build_pair_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Per unordered class-pair statistics over all conflict samples."""
    class_names = payload["class_names"]
    n_classes = len(class_names)
    pair_ids = sorted(
        {
            int(p)
            for p in payload["unordered_pair"]
            if int(p) >= 0
        }
    )
    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        a, b = divmod(int(pair_id), 100)
        mask = payload["unordered_pair"] == pair_id
        conflict = mask & payload["is_conflict"]
        swap = mask & payload["is_swap_candidate"]
        selected = mask & payload["swap_selected"]
        sel_stats = correction_stats(payload, mask)
        rows.append(
            {
                "pair_id": pair_id,
                "class_A": a,
                "class_A_name": class_name(class_names, a),
                "class_B": b,
                "class_B_name": class_name(class_names, b),
                "conflict_count": int(conflict.sum()),
                "swap_candidate_count": int(swap.sum()),
                "selected_count": int(selected.sum()),
                "selected_rate": (
                    100.0 * selected.sum() / swap.sum() if swap.sum() else 0.0
                ),
                "already_admitted_count": int(
                    (mask & payload["already_admitted"]).sum()
                ),
                "newly_admitted_count": int(
                    (mask & payload["newly_admitted"]).sum()
                ),
                "label_changed_count": int(
                    (mask & payload["label_changed"]).sum()
                ),
                "choose_task_count": int(
                    (
                        mask
                        & payload["swap_selected"]
                        & (payload["selected_source"] == "task")
                    ).sum()
                ),
                "choose_clip_count": int(
                    (
                        mask
                        & payload["swap_selected"]
                        & (payload["selected_source"] == "clip")
                    ).sum()
                ),
                "abstain_count": int((mask & ~payload["swap_selected"]).sum()),
                **sel_stats,
                "mean_task_top1_prob": float(
                    np.nanmean(payload["task_top1_prob"][mask])
                    if mask.any()
                    else np.nan
                ),
                "mean_clip_top1_prob": float(
                    np.nanmean(payload["clip_top1_prob"][mask])
                    if mask.any()
                    else np.nan
                ),
                "mean_task_margin": float(
                    np.nanmean(payload["task_margin"][mask])
                    if mask.any()
                    else np.nan
                ),
                "mean_clip_margin": float(
                    np.nanmean(payload["clip_margin"][mask])
                    if mask.any()
                    else np.nan
                ),
                "mean_signed_log_gap": float(
                    np.nanmean(payload["signed_log_gap"][mask])
                    if mask.any()
                    else np.nan
                ),
                "mean_absolute_log_gap": float(
                    np.nanmean(payload["absolute_log_gap"][mask])
                    if mask.any()
                    else np.nan
                ),
            }
        )
    return sorted(rows, key=lambda r: -r["conflict_count"])


def build_class_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Per real-class statistics (rows = ground-truth class)."""
    class_names = payload["class_names"]
    rows: list[dict[str, Any]] = []
    for class_id, name in enumerate(class_names):
        mask = payload["real_label"] == class_id
        selected = mask & payload["swap_selected"]
        base_c = int((selected & payload["base_correct"]).sum())
        final_c = int((selected & payload["final_correct"]).sum())
        w2r = int(
            (selected & ~payload["base_correct"] & payload["final_correct"]).sum()
        )
        r2w = int(
            (selected & payload["base_correct"] & ~payload["final_correct"]).sum()
        )
        rows.append(
            {
                "real_class": class_id,
                "real_class_name": name,
                "total_samples": int(mask.sum()),
                "conflict_count": int((mask & payload["is_conflict"]).sum()),
                "swap_candidate_count": int(
                    (mask & payload["is_swap_candidate"]).sum()
                ),
                "selected_count": int(selected.sum()),
                "newly_admitted_count": int(
                    (mask & payload["newly_admitted"]).sum()
                ),
                "label_changed_count": int(
                    (mask & payload["label_changed"]).sum()
                ),
                "base_accuracy": (
                    100.0 * base_c / selected.sum() if selected.sum() else 0.0
                ),
                "final_accuracy": (
                    100.0 * final_c / selected.sum() if selected.sum() else 0.0
                ),
                "W2R": w2r,
                "R2W": r2w,
                "net_correction": w2r - r2w,
            }
        )
    return rows


def build_cycle_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Aggregate summary matching the requested schema."""
    agreements = int(payload["is_agreement"].sum())
    conflicts = int(payload["is_conflict"].sum())
    swap_candidates = int(payload["is_swap_candidate"].sum())
    selected = int(payload["swap_selected"].sum())
    already = int(payload["already_admitted"].sum())
    newly = int(payload["newly_admitted"].sum())
    changed = int(payload["label_changed"].sum())
    sel_stats = correction_stats(payload, np.ones(payload["total_samples"], bool))
    effective_union = int((payload["newly_admitted"] | payload["label_changed"]).sum())
    return {
        "cycle": payload["cycle"],
        "total_samples": payload["total_samples"],
        "agreements": agreements,
        "conflicts": conflicts,
        "swap_candidates": swap_candidates,
        "swap_over_conflicts": 100.0 * swap_candidates / conflicts if conflicts else 0.0,
        "swap_over_all": 100.0 * swap_candidates / payload["total_samples"],
        "selected": selected,
        "selected_over_candidates": (
            100.0 * selected / swap_candidates if swap_candidates else 0.0
        ),
        "already_admitted": already,
        "newly_admitted": newly,
        "label_changed": changed,
        "selected_base_correct": sel_stats["base_correct_count"],
        "selected_final_correct": sel_stats["final_correct_count"],
        "selected_base_accuracy": sel_stats["base_accuracy"],
        "selected_final_accuracy": sel_stats["final_accuracy"],
        "wrong_to_right": sel_stats["W2R"],
        "right_to_wrong": sel_stats["R2W"],
        "right_to_right": sel_stats["R2R"],
        "wrong_to_wrong": sel_stats["W2W"],
        "net_correction": sel_stats["net_correction"],
        "dataset_level_net_delta": sel_stats["dataset_level_net_delta"],
        "effective_intervention_union": effective_union,
    }


def _confusion_matrix(
    row_index: np.ndarray, col_index: np.ndarray, n_classes: int
) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (row_index, col_index), 1)
    return matrix


def build_cycle_confusion_matrices(
    payload: dict[str, Any],
) -> dict[str, np.ndarray]:
    """All ten requested confusion matrices for one cycle."""
    n_classes = len(payload["class_names"])
    real = payload["real_label"]
    task_top1 = payload["task_top1"]
    clip_top1 = payload["clip_top1"]
    base_mix = payload["base_mix_label"]
    final_mem = payload["final_mem_label"]
    conflict = payload["is_conflict"]
    swap_candidate = payload["is_swap_candidate"]
    selected = payload["swap_selected"]
    newly = payload["newly_admitted"]
    changed = payload["label_changed"]
    base_correct = payload["base_correct"]
    final_correct = payload["final_correct"]
    final_mask = payload["final_label_mask"]
    w2r = selected & ~base_correct & final_correct
    r2w = selected & base_correct & ~final_correct
    return {
        "cm01_conflict_task": _confusion_matrix(real[conflict], task_top1[conflict], n_classes),
        "cm02_conflict_clip": _confusion_matrix(real[conflict], clip_top1[conflict], n_classes),
        "cm03_swap_candidate_base_mix": _confusion_matrix(real[swap_candidate], base_mix[swap_candidate], n_classes),
        "cm04_swap_selected_final": _confusion_matrix(real[selected], final_mem[selected], n_classes),
        "cm05_newly_admitted_final": _confusion_matrix(real[newly], final_mem[newly], n_classes),
        "cm06_label_changed_direction": _confusion_matrix(base_mix[changed], final_mem[changed], n_classes),
        "cm07_w2r_correction": _confusion_matrix(base_mix[w2r], real[w2r], n_classes),
        "cm08_r2w_damage": _confusion_matrix(real[r2w], final_mem[r2w], n_classes),
        "cm09_final_mask_base": _confusion_matrix(real[final_mask], base_mix[final_mask], n_classes),
        "cm10_final_mask_final": _confusion_matrix(real[final_mask], final_mem[final_mask], n_classes),
    }


SAMPLE_TABLE_COLUMNS = [
    "cycle",
    "sample_index",
    "image_path",
    "real_label",
    "real_class_name",
    "task_top1",
    "task_top1_name",
    "task_top1_prob",
    "task_top2",
    "task_top2_name",
    "task_top2_prob",
    "task_margin",
    "task_entropy",
    "task_correct",
    "clip_top1",
    "clip_top1_name",
    "clip_top1_prob",
    "clip_top2",
    "clip_top2_name",
    "clip_top2_prob",
    "clip_margin",
    "clip_entropy",
    "clip_correct",
    "is_agreement",
    "is_conflict",
    "is_swap_candidate",
    "candidate_A",
    "candidate_A_name",
    "candidate_B",
    "candidate_B_name",
    "unordered_pair",
    "task_evidence",
    "clip_evidence",
    "log_task_evidence",
    "log_clip_evidence",
    "signed_log_gap",
    "absolute_log_gap",
    "gate_D",
    "direction_accuracy",
    "min_direction_accuracy",
    "passed_gate",
    "passed_direction_filter",
    "swap_selected",
    "selected_source",
    "swap_proposed_label",
    "swap_proposed_name",
    "abstain_reason",
    "prev_label_mask",
    "current_agreement",
    "base_label_mask",
    "already_admitted",
    "newly_admitted",
    "final_label_mask",
    "base_mix_label",
    "base_mix_name",
    "final_mem_label",
    "final_mem_name",
    "label_changed",
    "base_correct",
    "final_correct",
    "correction_type",
]


def save_sample_table(payload: dict[str, Any], out_dir: Path) -> None:
    """One row per target sample (not only selected ones)."""
    frame = {key: payload[key] for key in SAMPLE_TABLE_COLUMNS}
    frame["cycle"] = np.full(payload["total_samples"], payload["cycle"], dtype=np.int64)
    frame["gate_D"] = np.full(payload["total_samples"], payload["gate_D"], dtype=np.float64)
    frame["min_direction_accuracy"] = np.full(
        payload["total_samples"], payload["min_direction_accuracy"], dtype=np.float64
    )
    path = out_dir / f"cycle{payload['cycle']:02d}_all_samples.csv"
    try:
        import pandas as pd

        pd.DataFrame(frame).to_csv(path, index=False)
    except (ImportError, ModuleNotFoundError):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SAMPLE_TABLE_COLUMNS)
            for i in range(payload["total_samples"]):
                writer.writerow([frame[k][i] for k in SAMPLE_TABLE_COLUMNS])


class SwapInterventionAuditor:
    """Collects Cycle 2/3 audit payloads and writes all artifacts."""

    def __init__(self, output_root: str | Path, class_names: list[str]) -> None:
        self.root = Path(output_root) / "swap_intervention_audit"
        self.class_names = list(class_names)
        self._payloads: dict[int, dict[str, Any]] = {}

    def record_cycle(self, curr_cycle: int, payload: dict[str, Any]) -> None:
        """Record one audited cycle (curr_cycle must be 1 or 2)."""
        if curr_cycle not in (1, 2):
            return
        cycle_dir = self.root / f"cycle{curr_cycle + 1:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        save_sample_table(payload, cycle_dir)
        for stem, title in CM_SPECS:
            matrices = build_cycle_confusion_matrices(payload)
            save_confusion_matrix(
                matrices[stem],
                out_dir=cycle_dir,
                stem=stem,
                title=title,
                class_names=self.class_names,
            )
        pair_rows = build_pair_summary(payload)
        _write_csv_rows(
            cycle_dir / f"cycle{curr_cycle + 1:02d}_pair_summary.csv",
            pair_rows,
        )
        _write_csv_rows(
            cycle_dir / "pair_summary_sorted_best.csv",
            sorted(pair_rows, key=lambda r: -r["net_correction"]),
        )
        _write_csv_rows(
            cycle_dir / "pair_summary_sorted_worst.csv",
            sorted(pair_rows, key=lambda r: r["net_correction"]),
        )
        class_rows = build_class_summary(payload)
        _write_csv_rows(
            cycle_dir / f"cycle{curr_cycle + 1:02d}_class_summary.csv",
            class_rows,
        )
        summary = build_cycle_summary(payload)
        (cycle_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        self._log_terminal(payload, summary, pair_rows)
        self._payloads[curr_cycle] = payload
        if 1 in self._payloads and 2 in self._payloads:
            self._save_transition(self._payloads[1], self._payloads[2])

    def _log_terminal(
        self,
        payload: dict[str, Any],
        summary: dict[str, Any],
        pair_rows: list[dict[str, Any]],
    ) -> None:
        logging.info(
            "SWAP AUDIT cycle=%d: total=%d, conflicts=%d, swap_candidates=%d, "
            "selected=%d, already_admitted=%d, newly_admitted=%d, "
            "label_changed=%d, W2R=%d, R2W=%d, net_correction=%d, "
            "base_acc_selected=%.2f, final_acc_selected=%.2f",
            summary["cycle"],
            summary["total_samples"],
            summary["conflicts"],
            summary["swap_candidates"],
            summary["selected"],
            summary["already_admitted"],
            summary["newly_admitted"],
            summary["label_changed"],
            summary["wrong_to_right"],
            summary["right_to_wrong"],
            summary["net_correction"],
            summary["selected_base_accuracy"],
            summary["selected_final_accuracy"],
        )
        best = sorted(pair_rows, key=lambda r: -r["net_correction"])[:5]
        worst = sorted(pair_rows, key=lambda r: r["net_correction"])[:5]
        logging.info(
            "SWAP AUDIT cycle=%d best pairs: %s",
            summary["cycle"],
            "; ".join(
                f"{r['class_A_name']}<->{r['class_B_name']}:{r['net_correction']}"
                for r in best
                if r["selected_count"] > 0
            ),
        )
        logging.info(
            "SWAP AUDIT cycle=%d worst pairs: %s",
            summary["cycle"],
            "; ".join(
                f"{r['class_A_name']}<->{r['class_B_name']}:{r['net_correction']}"
                for r in worst
                if r["selected_count"] > 0
            ),
        )

    def _save_transition(
        self, payload2: dict[str, Any], payload3: dict[str, Any]
    ) -> None:
        transition = build_cross_cycle_transition(payload2, payload3)
        out_dir = self.root
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "cycle02_cycle03_transition.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(list(transition["columns"]))
            for row in transition["rows"]:
                writer.writerow(row)
        (out_dir / "cycle_transition_summary.json").write_text(
            json.dumps(transition["summary"], indent=2, ensure_ascii=False)
            + "\n"
        )


def build_cross_cycle_transition(
    payload2: dict[str, Any], payload3: dict[str, Any]
) -> dict[str, Any]:
    """Align cycle 2/3 by sample_index and summarize label-stream transitions."""
    order2 = np.argsort(payload2["sample_index"], kind="stable")
    order3 = np.argsort(payload3["sample_index"], kind="stable")

    def pick(payload: dict[str, Any], order: np.ndarray, key: str) -> np.ndarray:
        return np.asarray(payload[key])[order]

    columns = [
        "sample_index",
        "image_path",
        "real_label",
        "real_class_name",
        "cycle2_is_conflict",
        "cycle3_is_conflict",
        "cycle2_is_swap_candidate",
        "cycle3_is_swap_candidate",
        "cycle2_swap_selected",
        "cycle3_swap_selected",
        "cycle2_selected_source",
        "cycle3_selected_source",
        "cycle2_base_mix_label",
        "cycle3_base_mix_label",
        "cycle2_final_mem_label",
        "cycle3_final_mem_label",
        "cycle2_base_label_mask",
        "cycle3_base_label_mask",
        "cycle2_final_label_mask",
        "cycle3_final_label_mask",
        "cycle2_correction_type",
        "cycle3_correction_type",
    ]
    aligned = {
        "sample_index": pick(payload2, order2, "sample_index"),
        "image_path": pick(payload2, order2, "image_path"),
        "real_label": pick(payload2, order2, "real_label"),
        "real_class_name": pick(payload2, order2, "real_class_name"),
        "cycle2_is_conflict": pick(payload2, order2, "is_conflict"),
        "cycle3_is_conflict": pick(payload3, order3, "is_conflict"),
        "cycle2_is_swap_candidate": pick(payload2, order2, "is_swap_candidate"),
        "cycle3_is_swap_candidate": pick(payload3, order3, "is_swap_candidate"),
        "cycle2_swap_selected": pick(payload2, order2, "swap_selected"),
        "cycle3_swap_selected": pick(payload3, order3, "swap_selected"),
        "cycle2_selected_source": pick(payload2, order2, "selected_source"),
        "cycle3_selected_source": pick(payload3, order3, "selected_source"),
        "cycle2_base_mix_label": pick(payload2, order2, "base_mix_label"),
        "cycle3_base_mix_label": pick(payload3, order3, "base_mix_label"),
        "cycle2_final_mem_label": pick(payload2, order2, "final_mem_label"),
        "cycle3_final_mem_label": pick(payload3, order3, "final_mem_label"),
        "cycle2_base_label_mask": pick(payload2, order2, "base_label_mask"),
        "cycle3_base_label_mask": pick(payload3, order3, "base_label_mask"),
        "cycle2_final_label_mask": pick(payload2, order2, "final_label_mask"),
        "cycle3_final_label_mask": pick(payload3, order3, "final_label_mask"),
        "cycle2_correction_type": pick(payload2, order2, "correction_type"),
        "cycle3_correction_type": pick(payload3, order3, "correction_type"),
        "cycle2_newly_admitted": pick(payload2, order2, "newly_admitted"),
    }
    rows = [[aligned[k][i] for k in columns] for i in range(len(aligned["sample_index"]))]

    c2 = aligned
    c2_sel = c2["cycle2_swap_selected"]
    c3_sel = aligned["cycle3_swap_selected"]
    c3 = aligned
    c3_final_correct = c3["cycle3_final_mem_label"] == c3["real_label"]
    c2_newly = c2["cycle2_newly_admitted"]
    c2_final_correct = c2["cycle2_final_mem_label"] == c2["real_label"]
    summary = {
        "cycle2_swap_to_cycle3_agreement": int(
            (c2_sel & ~c3["cycle3_is_conflict"]).sum()
        ),
        "cycle2_swap_to_cycle3_swap": int(
            (c2_sel & c3["cycle3_is_swap_candidate"]).sum()
        ),
        "cycle2_swap_to_cycle3_non_swap_conflict": int(
            (c2_sel & c3["cycle3_is_conflict"] & ~c3["cycle3_is_swap_candidate"]).sum()
        ),
        "cycle2_selected_to_cycle3_selected_same_side": int(
            (c2_sel & c3_sel & (c2["cycle2_final_mem_label"] == c3["cycle3_final_mem_label"])).sum()
        ),
        "cycle2_selected_to_cycle3_selected_opposite_side": int(
            (c2_sel & c3_sel & (c2["cycle2_final_mem_label"] != c3["cycle3_final_mem_label"])).sum()
        ),
        "cycle2_selected_to_cycle3_abstain": int((c2_sel & ~c3_sel).sum()),
        "cycle2_w2r_to_cycle3_remains_correct": int(
            ((c2["cycle2_correction_type"] == "W2R") & c3_final_correct).sum()
        ),
        "cycle2_r2w_to_cycle3_remains_wrong": int(
            ((c2["cycle2_correction_type"] == "R2W") & ~c3_final_correct).sum()
        ),
        "cycle2_newly_admitted_wrong_to_cycle3_prediction_status": int(
            (c2_newly & ~c2_final_correct & c3_final_correct).sum()
        ),
    }
    return {"columns": columns, "rows": rows, "summary": summary}


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
