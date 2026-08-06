#!/usr/bin/env python
"""Phase-1 read-only diagnostics for duet_first_cycle_prior_context_transformer.

Given an offline npz dump containing post-prior Task / CLIP probabilities and
target labels, this tool prints the masks the Context Transformer would use
(anchor bank, strict conflicts, weak agreements) plus Task/CLIP top-1 and
top-2 union coverage.  Target labels are used only for these offline
diagnostics (``ground_truth_affects_training=False``).

Example:
    python tools/run_duet_context_diagnostics.py \
        --npz output/uda/.../some_dump.npz --num-classes 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.duet_context import (
    ClassBalancedAnchorBank,
    _entropy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="path to diagnostic npz")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--anchors-per-class", type=int, default=8)
    parser.add_argument("--anchor-task-conf", type=float, default=0.90)
    parser.add_argument("--anchor-clip-conf", type=float, default=0.90)
    parser.add_argument("--anchor-task-entropy", type=float, default=0.40)
    parser.add_argument("--anchor-clip-entropy", type=float, default=0.40)
    parser.add_argument("--weak-conf-threshold", type=float, default=0.70)
    parser.add_argument("--weak-entropy-threshold", type=float, default=1.00)
    args = parser.parse_args()

    data = np.load(args.npz)
    task_prob = torch.from_numpy(data["task_prob"]).float()
    clip_prob = torch.from_numpy(data["clip_prob"]).float()
    labels = torch.from_numpy(data["target_label"]).long()
    if task_prob.dim() != 2 or clip_prob.shape != task_prob.shape:
        raise ValueError("npz must contain task_prob/clip_prob as [N, C]")
    if labels.shape != (task_prob.size(0),):
        raise ValueError("target_label must be [N]")

    task_conf, task_top1 = task_prob.max(dim=1)
    clip_conf, clip_top1 = clip_prob.max(dim=1)
    task_entropy = _entropy(task_prob)
    clip_entropy = _entropy(clip_prob)

    matching = task_top1 == clip_top1
    strict_conflicts = ~matching
    weak_agreement = (
        matching
        & (
            (task_conf < args.weak_conf_threshold)
            | (clip_conf < args.weak_conf_threshold)
            | (task_entropy > args.weak_entropy_threshold)
            | (clip_entropy > args.weak_entropy_threshold)
        )
    )
    anchor_mask = (
        matching
        & (task_conf >= args.anchor_task_conf)
        & (clip_conf >= args.anchor_clip_conf)
        & (task_entropy <= args.anchor_task_entropy)
        & (clip_entropy <= args.anchor_clip_entropy)
    )

    n = task_prob.size(0)
    top1_union = int(
        ((task_top1 == labels) | (clip_top1 == labels)).sum().item()
    ) / n
    task_top2 = task_prob.topk(2, dim=1).indices
    clip_top2 = clip_prob.topk(2, dim=1).indices
    labels_col = labels.unsqueeze(1)
    top2_union = int(
        (
            (task_top2 == labels_col).any(dim=1)
            | (clip_top2 == labels_col).any(dim=1)
        )
        .sum()
        .item()
    ) / n

    bank = ClassBalancedAnchorBank(
        num_classes=args.num_classes,
        anchors_per_class=args.anchors_per_class,
        feature_dim=task_prob.size(1),
    )
    if int(anchor_mask.sum().item()) > 0:
        reliability = (
            task_conf[anchor_mask]
            + clip_conf[anchor_mask]
            - 1.0 * (task_entropy[anchor_mask] + clip_entropy[anchor_mask])
        )
        bank.update(
            features=torch.zeros(anchor_mask.sum(), task_prob.size(1)),
            labels=task_top1[anchor_mask],
            scores=reliability,
        )
        anchor_precision = float(
            (task_top1[anchor_mask] == labels[anchor_mask]).float().mean().item()
        )
    else:
        anchor_precision = float("nan")
    print("DUET context diagnostics (offline, ground_truth_affects_training=False)")
    print("samples={} classes={}".format(n, args.num_classes))
    print("post-prior Task/CLIP agreement={}".format(int(matching.sum().item())))
    print("strict_conflicts={}".format(int(strict_conflicts.sum().item())))
    print("weak_agreement={}".format(int(weak_agreement.sum().item())))
    print("anchor_candidates={} anchor_precision={:.2f}%".format(
        int(anchor_mask.sum().item()), anchor_precision * 100.0
    ))
    print("anchors_per_class={}".format(bank.per_class_counts().tolist()))
    print("Task/CLIP top-1 union coverage={:.4f}%".format(top1_union * 100.0))
    print("Task/CLIP top-2 union coverage={:.4f}%".format(top2_union * 100.0))


if __name__ == "__main__":
    main()
