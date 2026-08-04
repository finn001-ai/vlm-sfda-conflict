#!/usr/bin/env python
"""DCPL 噪声转移矩阵的离线诊断（纯 numpy，无需 GPU/训练依赖）。

输入：tools/run_visda_duet_softmax_dump.sh 导出的 npz（每 cycle 一个）。
每个 npz 含：task_probability [N,12]、clip_probability [N,12]、labels [N]（仅评估用）、
sample_indices [N]、class_names。

做什么：
  1. 基线：task argmax 精度、CLIP 伪标签精度（全量样本，不只冲突子集）。
  2. 按 DCPL（ECCV 2024）的方式估计噪声转移矩阵 CM：
     CM[L] = 伪标签为 L 的样本的 task softmax 均值（行归一化），
     另附 rank 版（softmax 排名分布）对照。
  3. 校正：pred = argmax(task_softmax @ CM)，与基线对比。
GT 只用于评估，不参与 CM 估计，保持 oracle-diagnostic 原则。

用法：
  python tools/analyze_dcpl_confusion.py --dump-dir <dir> [--out out.json]
  python tools/analyze_dcpl_confusion.py --npz <cycle_000.npz> [--out out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float64)
    return matrix / matrix.sum(axis=1, keepdims=True)


def estimate_confusion_matrix(task_prob: np.ndarray, pseudo: np.ndarray) -> np.ndarray:
    """DCPL 式 CM：CM[L][j] = mean(task_prob[j] | pseudo_label == L)，行归一化。"""
    num_classes = task_prob.shape[1]
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for class_id in range(num_classes):
        rows = task_prob[pseudo == class_id]
        if rows.size == 0:
            cm[class_id] = np.ones(num_classes) / num_classes  # 与 DCPL 代码一致
        else:
            cm[class_id] = rows.mean(axis=0)
    return _row_normalize(cm)


def rank_softmax(prob: np.ndarray) -> np.ndarray:
    """把概率转成排名分布的 softmax（DCPL 的 pl_order 变体）。"""
    ranks = np.argsort(np.argsort(-prob, axis=1, kind="stable"), axis=1)
    scores = np.exp(-ranks.astype(np.float64))
    return scores / scores.sum(axis=1, keepdims=True)


def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(pred == labels)) * 100.0 if pred.size else 0.0


def analyze_cycle(data: dict[str, np.ndarray]) -> dict:
    task_prob = data["task_probability"].astype(np.float64)
    clip_prob = data["clip_probability"].astype(np.float64)
    labels = data["labels"].astype(np.int64)
    class_names = [str(name) for name in data.get("class_names", [])]
    num_classes = task_prob.shape[1]
    if not class_names:
        class_names = [str(i) for i in range(num_classes)]

    task_pred = np.argmax(task_prob, axis=1)
    pseudo = np.argmax(clip_prob, axis=1)
    agree = task_pred == pseudo

    cm_mean = estimate_confusion_matrix(task_prob, pseudo)
    cm_rank = estimate_confusion_matrix(rank_softmax(task_prob), pseudo)
    corrected_mean = np.argmax(task_prob @ cm_mean, axis=1)
    corrected_rank = np.argmax(rank_softmax(task_prob) @ cm_rank, axis=1)

    per_class = []
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "count": int(mask.sum()),
                "task_acc": round(accuracy(task_pred[mask], labels[mask]), 2),
                "pseudo_acc": round(accuracy(pseudo[mask], labels[mask]), 2),
                "corrected_mean_acc": round(accuracy(corrected_mean[mask], labels[mask]), 2),
                "corrected_rank_acc": round(accuracy(corrected_rank[mask], labels[mask]), 2),
            }
        )

    result = {
        "cycle": int(data.get("cycle", -1)),
        "seed": int(data.get("seed", -1)),
        "samples": int(labels.size),
        "class_names": class_names,
        "baseline": {
            "task_acc": round(accuracy(task_pred, labels), 2),
            "pseudo_acc": round(accuracy(pseudo, labels), 2),
            "agreement_rate": round(100.0 * float(np.mean(agree)), 2),
            "agree_and_pseudo_correct": round(
                accuracy(pseudo[agree], labels[agree]), 2
            ),
        },
        "corrected": {
            "mean_acc": round(accuracy(corrected_mean, labels), 2),
            "rank_acc": round(accuracy(corrected_rank, labels), 2),
            "gain_vs_best_baseline_mean": round(
                accuracy(corrected_mean, labels)
                - max(accuracy(task_pred, labels), accuracy(pseudo, labels)),
                2,
            ),
            "gain_vs_best_baseline_rank": round(
                accuracy(corrected_rank, labels)
                - max(accuracy(task_pred, labels), accuracy(pseudo, labels)),
                2,
            ),
        },
        "cm_diagonal_mass": round(
            float(np.trace(cm_mean)) / num_classes * 100.0, 2
        ),
        "per_class": per_class,
        "cm_mean_rows": {
            class_names[i]: [round(float(v), 4) for v in row]
            for i, row in enumerate(cm_mean)
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", type=Path, help="单个 cycle 的 npz")
    source.add_argument("--dump-dir", type=Path, help="导出目录（循环处理 cycle_*.npz）")
    parser.add_argument("--out", type=Path, default=None, help="可选 JSON 输出路径")
    args = parser.parse_args()

    paths: list[Path] = []
    if args.npz:
        paths = [args.npz]
    else:
        paths = sorted(args.dump_dir.glob("cycle_*.npz"))
    if not paths:
        raise SystemExit("No cycle_*.npz found")

    results = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            result = analyze_cycle({key: data[key] for key in data.files})
        print("=" * 78)
        print(
            "cycle={cycle:02d}  samples={samples}  task={baseline[task_acc]}%  "
            "CLIP={baseline[pseudo_acc]}%  corrected(mean)={corrected[mean_acc]}%  "
            "corrected(rank)={corrected[rank_acc]}%  gain={corrected[gain_vs_best_baseline_mean]}pp".format(
                **result
            )
        )
        print(
            "  agreement={baseline[agreement_rate]}%  "
            "agree&pseudo-correct={baseline[agree_and_pseudo_correct]}%  "
            "CM 对角线均值={cm_diagonal_mass}%".format(**result)
        )
        print("  per-class acc (task/pseudo/corrected-mean):")
        for row in result["per_class"]:
            print(
                "    {class_name:>12}: {task_acc:5.2f} / {pseudo_acc:5.2f} / "
                "{corrected_mean_acc:5.2f}   (n={count})".format(**row)
            )
        results.append(result)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"==> 结果已写入 {args.out}")


if __name__ == "__main__":
    main()
