import json
from pathlib import Path

import numpy as np

from src.utils.topk_conflict_probe import (
    build_topk_conflict_probe,
    write_topk_conflict_probe,
)


def test_topk_probe_groups_recovery_and_cross_support():
    task = np.asarray(
        [
            [0.60, 0.30, 0.10, 0.00],
            [0.10, 0.60, 0.20, 0.10],
            [0.45, 0.05, 0.40, 0.10],
            [0.45, 0.35, 0.15, 0.05],
            [0.45, 0.35, 0.15, 0.05],
            [0.45, 0.35, 0.15, 0.05],
        ]
    )
    clip = np.asarray(
        [
            [0.30, 0.60, 0.10, 0.00],
            [0.60, 0.10, 0.20, 0.10],
            [0.10, 0.45, 0.40, 0.05],
            [0.10, 0.45, 0.40, 0.05],
            [0.10, 0.45, 0.05, 0.40],
            [0.10, 0.45, 0.40, 0.05],
        ]
    )
    labels = np.asarray([0, 0, 2, 1, 3, 3])
    result = build_topk_conflict_probe(
        task_probability=task,
        clip_probability=clip,
        labels=labels,
        sample_indices=np.arange(6),
        dataset_items=[(f"image_{i}.jpg", int(labels[i])) for i in range(6)],
        class_names=["a", "b", "c", "d"],
        task_name="TV",
        source_domain="train",
        target_domain="validation",
        seed=2020,
        cycle=0,
    )
    overall = result["summary"]["overall"]
    assert overall["conflict_samples"] == 6
    assert overall["top1_union_gt_count"] == 3
    assert overall["top1_both_wrong_count"] == 3
    assert overall["top2_union_gt_count"] == 5
    assert overall["top2_recovered_count"] == 2
    assert overall["top2_union_missed_count"] == 1
    assert [row["probe_group"] for row in result["rows"]] == [
        "task_top1_correct",
        "clip_top1_correct",
        "both_top2_recovery",
        "clip_top1_correct",
        "clip_top2_only_recovery",
        "top2_union_missed",
    ]
    assert all(2 <= row["top2_union_size"] <= 4 for row in result["rows"])
    assert all(isinstance(json.loads(row["top2_union"]), list) for row in result["rows"])


def test_topk_probe_writes_all_summary_tables(tmp_path):
    task = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]])
    clip = np.asarray([[0.2, 0.7, 0.1], [0.7, 0.2, 0.1]])
    summary = write_topk_conflict_probe(
        output_root=tmp_path,
        task_probability=task,
        clip_probability=clip,
        labels=np.asarray([0, 2]),
        sample_indices=np.asarray([0, 1]),
        dataset_items=[("a.jpg", 0), ("b.jpg", 2)],
        class_names=["a", "b", "c"],
        task_name="TV",
        source_domain="train",
        target_domain="validation",
        seed=2020,
        cycle=0,
    )
    cycle_dir = tmp_path / "conflict_probe/task_TV_seed_2020/cycle_000"
    assert (cycle_dir / "conflict_summary.json").is_file()
    assert (cycle_dir / "conflict_summary.csv").is_file()
    assert (cycle_dir / "conflict_summary_by_class.csv").is_file()
    assert Path(summary["artifacts"]["sample_level"]).is_file()
