"""Oracle-only Task/CLIP top-k conflict coverage probe.

The candidate sets are constructed from detached Task and CLIP probabilities.
Ground-truth labels are used only to annotate and summarize the already fixed
sets; this module returns no tensor or decision to the training path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VISDA_CYCLE0_REGRESSION = {
    "total_samples": 55_388,
    "conflict_samples": 28_255,
    "top1_union_gt_count": 23_321,
    "top1_both_wrong_count": 4_934,
    "top2_union_gt_count": 26_387,
    "top2_recovered_count": 3_066,
    "top2_union_missed_count": 1_868,
}


def _numpy(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _ordered_unique(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_probability(name: str, value: Any) -> np.ndarray:
    probability = _numpy(value, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] == 0 or probability.shape[1] < 2:
        raise ValueError(f"{name} must have shape [sample, class>=2]")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} rows must sum to one")
    return probability


def _group_metrics(
    mask: np.ndarray,
    *,
    task_top1_correct: np.ndarray,
    clip_top1_correct: np.ndarray,
    gt_in_top1_union: np.ndarray,
    gt_in_top2_union: np.ndarray,
) -> dict[str, Any]:
    samples = int(mask.sum())
    top1_missed = mask & ~gt_in_top1_union
    recovered = top1_missed & gt_in_top2_union
    return {
        "samples": samples,
        "task_top1_accuracy": _pct((mask & task_top1_correct).sum(), samples),
        "clip_top1_accuracy": _pct((mask & clip_top1_correct).sum(), samples),
        "top1_union_coverage": _pct((mask & gt_in_top1_union).sum(), samples),
        "top2_union_coverage": _pct((mask & gt_in_top2_union).sum(), samples),
        "top2_conditional_recovery_rate": _pct(recovered.sum(), top1_missed.sum()),
    }


def build_topk_conflict_probe(
    *,
    task_probability: Any,
    clip_probability: Any,
    labels: Any,
    sample_indices: Any,
    dataset_items: list[tuple[Any, Any]],
    class_names: list[str],
    task_name: str,
    source_domain: str,
    target_domain: str,
    seed: int,
    cycle: int,
    probability_stage: str = "pre_first_cycle_prior",
) -> dict[str, Any]:
    """Build sample rows and summaries without affecting model state."""
    task = _validate_probability("task_probability", task_probability)
    clip = _validate_probability("clip_probability", clip_probability)
    if task.shape != clip.shape:
        raise ValueError("Task and CLIP probability shapes must match")
    label = _numpy(labels, dtype=np.int64)
    sample_index = _numpy(sample_indices, dtype=np.int64)
    sample_count, class_count = task.shape
    if label.shape != (sample_count,) or sample_index.shape != (sample_count,):
        raise ValueError("labels and sample_indices must contain one value per row")
    if len(class_names) != class_count:
        raise ValueError("class_names length must match the probability columns")
    if np.any(label < 0) or np.any(label >= class_count):
        raise ValueError("ground-truth label is outside the class range")
    if np.unique(sample_index).size != sample_count:
        raise ValueError("sample_indices must be unique")
    if np.any(sample_index < 0) or np.any(sample_index >= len(dataset_items)):
        raise ValueError("sample index is outside loader.dataset.imgs")

    order = np.argsort(sample_index, kind="stable")
    task = task[order]
    clip = clip[order]
    label = label[order]
    sample_index = sample_index[order]

    task_rank = np.argsort(-task, axis=1, kind="stable")[:, :2]
    clip_rank = np.argsort(-clip, axis=1, kind="stable")[:, :2]
    task_top1 = task_rank[:, 0]
    task_top2 = task_rank[:, 1]
    clip_top1 = clip_rank[:, 0]
    clip_top2 = clip_rank[:, 1]
    conflict = task_top1 != clip_top1

    task_top1_correct = task_top1 == label
    clip_top1_correct = clip_top1 == label
    task_top2_correct = task_top2 == label
    clip_top2_correct = clip_top2 == label
    gt_in_top1_union = task_top1_correct | clip_top1_correct
    gt_in_top2_union = gt_in_top1_union | task_top2_correct | clip_top2_correct
    top1_both_wrong = ~gt_in_top1_union

    recovery_source = np.full(sample_count, "not_applicable", dtype=object)
    recovery_source[top1_both_wrong] = "not_recovered"
    recovery_source[top1_both_wrong & task_top2_correct] = "task_top2_only"
    recovery_source[top1_both_wrong & clip_top2_correct] = "clip_top2_only"
    recovery_source[top1_both_wrong & task_top2_correct & clip_top2_correct] = "both_top2"

    probe_group = np.full(sample_count, "", dtype=object)
    probe_group[task_top1_correct] = "task_top1_correct"
    probe_group[clip_top1_correct] = "clip_top1_correct"
    probe_group[top1_both_wrong & task_top2_correct & ~clip_top2_correct] = "task_top2_only_recovery"
    probe_group[top1_both_wrong & ~task_top2_correct & clip_top2_correct] = "clip_top2_only_recovery"
    probe_group[top1_both_wrong & task_top2_correct & clip_top2_correct] = "both_top2_recovery"
    probe_group[top1_both_wrong & ~task_top2_correct & ~clip_top2_correct] = "top2_union_missed"

    task_top1_eq_clip_top2 = task_top1 == clip_top2
    clip_top1_eq_task_top2 = clip_top1 == task_top2
    task_top2_eq_clip_top2 = task_top2 == clip_top2
    bidirectional = task_top1_eq_clip_top2 & clip_top1_eq_task_top2
    cross_group = np.full(sample_count, "no_cross_support", dtype=object)
    cross_group[task_top1_eq_clip_top2] = "task_top1_supported_by_clip_top2"
    cross_group[clip_top1_eq_task_top2] = "clip_top1_supported_by_task_top2"
    cross_group[bidirectional] = "bidirectional_cross_support"

    conflict_positions = np.flatnonzero(conflict)
    if conflict_positions.size == 0:
        raise ValueError("Top-k conflict probe found no task/CLIP top-1 conflicts")
    if np.any(probe_group[conflict_positions] == ""):
        raise AssertionError("Every conflict must belong to exactly one probe group")

    rows: list[dict[str, Any]] = []
    union_size = np.zeros(sample_count, dtype=np.int64)
    new_candidate_count = np.zeros(sample_count, dtype=np.int64)
    for position in conflict_positions:
        task1 = int(task_top1[position])
        task2 = int(task_top2[position])
        clip1 = int(clip_top1[position])
        clip2 = int(clip_top2[position])
        top1_union = _ordered_unique([task1, clip1])
        top2_union = _ordered_unique([task1, task2, clip1, clip2])
        union_size[position] = len(top2_union)
        new_candidate_count[position] = len(top2_union) - len(top1_union)
        item = dataset_items[int(sample_index[position])]
        item_target = int(np.asarray(item[1]).item())
        if item_target != int(label[position]):
            raise RuntimeError("Loader label and dataset item target are misaligned")
        rows.append(
            {
                "task": task_name,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "seed": int(seed),
                "cycle": int(cycle),
                "probability_stage": probability_stage,
                "sample_id": int(sample_index[position]),
                "sample_path": str(item[0]),
                "gt_label_probe": int(label[position]),
                "task_top1_id": task1,
                "task_top1_prob": float(task[position, task1]),
                "task_top2_id": task2,
                "task_top2_prob": float(task[position, task2]),
                "clip_top1_id": clip1,
                "clip_top1_score": float(clip[position, clip1]),
                "clip_top2_id": clip2,
                "clip_top2_score": float(clip[position, clip2]),
                "clip_score_type": "softmax_probability",
                "task_margin_12": float(task[position, task1] - task[position, task2]),
                "clip_margin_12": float(clip[position, clip1] - clip[position, clip2]),
                "task_top1_correct": bool(task_top1_correct[position]),
                "clip_top1_correct": bool(clip_top1_correct[position]),
                "task_top2_correct": bool(task_top2_correct[position]),
                "clip_top2_correct": bool(clip_top2_correct[position]),
                "gt_in_top1_union": bool(gt_in_top1_union[position]),
                "gt_in_top2_union": bool(gt_in_top2_union[position]),
                "task_top1_eq_clip_top2": bool(task_top1_eq_clip_top2[position]),
                "clip_top1_eq_task_top2": bool(clip_top1_eq_task_top2[position]),
                "task_top2_eq_clip_top2": bool(task_top2_eq_clip_top2[position]),
                "bidirectional_cross_support": bool(bidirectional[position]),
                "cross_support_group": str(cross_group[position]),
                "top1_union": json.dumps(top1_union),
                "top2_union": json.dumps(top2_union),
                "top1_union_size": len(top1_union),
                "top2_union_size": len(top2_union),
                "new_candidate_count": int(new_candidate_count[position]),
                "recovery_source": str(recovery_source[position]),
                "probe_group": str(probe_group[position]),
                "oracle_usage": "diagnostic_only_never_training",
            }
        )

    selected = conflict
    top1_union_gt_count = int((selected & gt_in_top1_union).sum())
    top2_union_gt_count = int((selected & gt_in_top2_union).sum())
    top1_both_wrong_count = int((selected & top1_both_wrong).sum())
    top2_recovered_count = top2_union_gt_count - top1_union_gt_count
    top2_union_missed_count = int((selected & ~gt_in_top2_union).sum())

    group_counts = {
        name: int((selected & (probe_group == name)).sum())
        for name in (
            "task_top1_correct",
            "clip_top1_correct",
            "task_top2_only_recovery",
            "clip_top2_only_recovery",
            "both_top2_recovery",
            "top2_union_missed",
        )
    }
    conflict_count = int(selected.sum())
    assert conflict_count == sum(group_counts.values())
    assert top2_recovered_count == (
        group_counts["task_top2_only_recovery"]
        + group_counts["clip_top2_only_recovery"]
        + group_counts["both_top2_recovery"]
    )
    assert top2_union_gt_count == top1_union_gt_count + top2_recovered_count

    overall = {
        "total_samples": sample_count,
        "conflict_samples": conflict_count,
        "top1_union_gt_count": top1_union_gt_count,
        "top1_union_coverage": _pct(top1_union_gt_count, conflict_count),
        "top1_both_wrong_count": top1_both_wrong_count,
        "top2_union_gt_count": top2_union_gt_count,
        "top2_union_coverage": _pct(top2_union_gt_count, conflict_count),
        "top2_recovered_count": top2_recovered_count,
        "top2_conditional_recovery_rate": _pct(top2_recovered_count, top1_both_wrong_count),
        "top2_union_missed_count": top2_union_missed_count,
        "top2_union_missed_rate": _pct(top2_union_missed_count, conflict_count),
        "task_top1_correct_count": group_counts["task_top1_correct"],
        "clip_top1_correct_count": group_counts["clip_top1_correct"],
        "task_top2_only_recovery_count": group_counts["task_top2_only_recovery"],
        "clip_top2_only_recovery_count": group_counts["clip_top2_only_recovery"],
        "both_top2_recovery_count": group_counts["both_top2_recovery"],
        "not_recovered_count": group_counts["top2_union_missed"],
        "mean_top2_union_size": float(union_size[selected].mean()),
        "mean_new_candidate_count": float(new_candidate_count[selected].mean()),
    }

    by_union_size = []
    for size in (2, 3, 4):
        mask = selected & (union_size == size)
        metrics = _group_metrics(
            mask,
            task_top1_correct=task_top1_correct,
            clip_top1_correct=clip_top1_correct,
            gt_in_top1_union=gt_in_top1_union,
            gt_in_top2_union=gt_in_top2_union,
        )
        metrics.update(
            {
                "top2_union_size": size,
                "mean_new_candidate_count": (
                    float(new_candidate_count[mask].mean()) if mask.any() else 0.0
                ),
            }
        )
        by_union_size.append(metrics)

    by_cross_support = []
    for name in (
        "bidirectional_cross_support",
        "task_top1_supported_by_clip_top2",
        "clip_top1_supported_by_task_top2",
        "no_cross_support",
    ):
        metrics = _group_metrics(
            selected & (cross_group == name),
            task_top1_correct=task_top1_correct,
            clip_top1_correct=clip_top1_correct,
            gt_in_top1_union=gt_in_top1_union,
            gt_in_top2_union=gt_in_top2_union,
        )
        metrics["cross_support_group"] = name
        by_cross_support.append(metrics)

    class_rows = []
    for class_id, class_name in enumerate(class_names):
        mask = selected & (label == class_id)
        metrics = _group_metrics(
            mask,
            task_top1_correct=task_top1_correct,
            clip_top1_correct=clip_top1_correct,
            gt_in_top1_union=gt_in_top1_union,
            gt_in_top2_union=gt_in_top2_union,
        )
        class_rows.append(
            {
                "gt_class_id": class_id,
                "gt_class_name": class_name,
                "conflict_count": metrics["samples"],
                "task_top1_correct_count": int((mask & task_top1_correct).sum()),
                "clip_top1_correct_count": int((mask & clip_top1_correct).sum()),
                "top1_union_coverage": metrics["top1_union_coverage"],
                "top2_union_coverage": metrics["top2_union_coverage"],
                "conditional_recovery_rate": metrics["top2_conditional_recovery_rate"],
                "top2_union_missed_count": int((mask & ~gt_in_top2_union).sum()),
            }
        )

    regression_checks = {
        key: overall[key] == expected
        for key, expected in VISDA_CYCLE0_REGRESSION.items()
    }
    regression_applicable = (
        int(cycle) == 0
        and probability_stage == "pre_first_cycle_prior"
        and sample_count == VISDA_CYCLE0_REGRESSION["total_samples"]
    )
    summary = {
        "task": task_name,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "seed": int(seed),
        "cycle": int(cycle),
        "probability_stage": probability_stage,
        "oracle_diagnostic": True,
        "ground_truth_affects_training": False,
        "clip_score_type": "softmax_probability",
        "rate_scale": "percentage_0_to_100",
        "overall": overall,
        "by_top2_union_size": by_union_size,
        "by_cross_support": by_cross_support,
        "known_visda_cycle0_regression": {
            "applicable": regression_applicable,
            "expected": VISDA_CYCLE0_REGRESSION,
            "checks": regression_checks if regression_applicable else {},
            "passed": all(regression_checks.values()) if regression_applicable else None,
        },
    }
    return {
        "rows": rows,
        "summary": summary,
        "class_rows": class_rows,
        # 排序后（按 sample_index）的全量概率矩阵，供可选的 softmax dump 使用。
        # 仅返回引用、不复制；诊断语义与既有输出完全不变。
        "full_probabilities": {
            "task_probability": task,
            "clip_probability": clip,
            "labels": label,
            "sample_indices": sample_index,
        },
    }


def write_topk_conflict_probe(
    *, output_root: str | Path, softmax_dump_dir: str | Path | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Build and write one cycle of probe artifacts."""
    result = build_topk_conflict_probe(**kwargs)
    summary = result["summary"]
    cycle_dir = (
        Path(output_root)
        / "conflict_probe"
        / f"task_{summary['task']}_seed_{summary['seed']}"
        / f"cycle_{summary['cycle']:03d}"
    )
    cycle_dir.mkdir(parents=True, exist_ok=True)

    sample_csv = cycle_dir / "conflict_samples.csv"
    sample_parquet = cycle_dir / "conflict_samples.parquet"
    sample_format = "csv"
    try:
        import pandas as pd

        pd.DataFrame(result["rows"]).to_parquet(sample_parquet, index=False)
        sample_path = sample_parquet
        sample_format = "parquet"
        if sample_csv.exists():
            sample_csv.unlink()
    except (ImportError, ModuleNotFoundError, OSError, ValueError):
        _write_csv(sample_csv, result["rows"])
        sample_path = sample_csv
        if sample_parquet.exists():
            sample_parquet.unlink()

    overall_row = {"section": "overall", "group": "all", **summary["overall"]}
    summary_rows = [overall_row]
    for row in summary["by_top2_union_size"]:
        summary_rows.append(
            {"section": "top2_union_size", "group": str(row["top2_union_size"]), **row}
        )
    for row in summary["by_cross_support"]:
        summary_rows.append(
            {"section": "cross_support", "group": row["cross_support_group"], **row}
        )

    summary["artifacts"] = {
        "sample_level": str(sample_path),
        "sample_level_format": sample_format,
        "summary_json": str(cycle_dir / "conflict_summary.json"),
        "summary_csv": str(cycle_dir / "conflict_summary.csv"),
        "class_csv": str(cycle_dir / "conflict_summary_by_class.csv"),
    }
    (cycle_dir / "conflict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    _write_csv(cycle_dir / "conflict_summary.csv", summary_rows)
    _write_csv(cycle_dir / "conflict_summary_by_class.csv", result["class_rows"])
    if softmax_dump_dir:
        dump_dir = Path(softmax_dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        full = result["full_probabilities"]
        dump_path = dump_dir / "cycle_{:03d}.npz".format(summary["cycle"])
        np.savez(
            dump_path,
            task_probability=full["task_probability"],
            clip_probability=full["clip_probability"],
            labels=full["labels"],
            sample_indices=full["sample_indices"],
            class_names=np.asarray(kwargs.get("class_names", []), dtype=object),
            task=np.asarray(summary["task"]),
            seed=np.asarray(summary["seed"]),
            cycle=np.asarray(summary["cycle"]),
        )
    return summary
