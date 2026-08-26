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


def test_topk_probe_optional_full_softmax_dump(tmp_path):
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
    labels = np.asarray([3, 1, 2, 0, 3, 2])
    # 乱序 sample_indices：dump 必须按 sample_index 重排，与 probe 内部一致。
    shuffled = np.asarray([5, 0, 4, 2, 1, 3])
    class_names = ["a", "b", "c", "d"]
    # dataset_items 按数据集索引给出目标：dataset_items[k].target 必须等于
    # 样本 k 的 GT（probe 的对齐校验），k 对应的行位置 p 满足 shuffled[p]==k。
    inverse = np.argsort(shuffled, kind="stable")
    dataset_items = [
        (f"image_{k}.jpg", int(labels[inverse[k]])) for k in range(len(shuffled))
    ]
    kwargs = dict(
        task_probability=task,
        clip_probability=clip,
        labels=labels,
        sample_indices=shuffled,
        dataset_items=dataset_items,
        class_names=class_names,
        task_name="TV",
        source_domain="train",
        target_domain="validation",
        seed=2020,
        cycle=1,
    )
    dump_dir = tmp_path / "softmax_dump"
    write_topk_conflict_probe(output_root=tmp_path, softmax_dump_dir=dump_dir, **kwargs)
    dump_path = dump_dir / "cycle_001.npz"
    assert dump_path.is_file()
    with np.load(dump_path, allow_pickle=True) as loaded:
        assert loaded["task_probability"].shape == (6, 4)
        assert loaded["clip_probability"].shape == (6, 4)
        # 按 sample_index 排序后应与输入一致（probe 内部对全量矩阵做了 stable 排序）。
        order = np.argsort(shuffled, kind="stable")
        np.testing.assert_allclose(loaded["task_probability"], task[order])
        np.testing.assert_allclose(loaded["clip_probability"], clip[order])
        np.testing.assert_array_equal(loaded["labels"], labels[order])
        np.testing.assert_array_equal(loaded["sample_indices"], shuffled[order])
        np.testing.assert_array_equal(loaded["class_names"], np.asarray(class_names, dtype=object))
        assert int(loaded["cycle"]) == 1
        assert int(loaded["seed"]) == 2020


def test_topk_probe_dump_disabled_by_default(tmp_path):
    task = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]])
    clip = np.asarray([[0.2, 0.7, 0.1], [0.7, 0.2, 0.1]])
    write_topk_conflict_probe(
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
    assert not any(tmp_path.rglob("*.npz"))
    assert (tmp_path / "conflict_probe/task_TV_seed_2020/cycle_000/conflict_summary.json").is_file()
