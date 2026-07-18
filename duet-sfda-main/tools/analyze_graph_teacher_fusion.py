#!/usr/bin/env python
"""Probe continuous fusion of both_prior teacher and graph diffusion posterior.

This evaluates whether graph diffusion can improve the full soft teacher before
training, without per-sample source/CLIP hard selection.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.conflict_diffusion import dual_space_diffusion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/plmatch/diagnostics/*C_conflicts.npz",
    )
    parser.add_argument(
        "--out",
        default="output/uda/office-home/graph_teacher_fusion_probe.json",
    )
    parser.add_argument("--graph-k", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--anchor-ratio", type=float, default=0.5)
    parser.add_argument("--anchor-min-per-class", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--calib-power", type=float, default=0.5)
    parser.add_argument("--fusion-strength", type=float, default=0.5)
    parser.add_argument("--min-task-delta", type=float, default=0.05)
    parser.add_argument("--min-pass-tasks", type=int, default=2)
    return parser.parse_args()


def prior_calibrate(prob: torch.Tensor, power: float, eps: float = 1e-6) -> torch.Tensor:
    prior = prob.mean(dim=0).clamp_min(eps)
    calibrated = prob / prior.pow(power)
    return calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(eps)


def accuracy(prob: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mask = torch.ones(labels.numel(), dtype=torch.bool)
    pred = prob.argmax(dim=1)
    denom = int(mask.sum().item())
    return round(100.0 * float((pred[mask] == labels[mask]).sum().item()) / denom, 4) if denom else 0.0


def entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.clamp_min(eps)
    return -(prob * prob.log()).sum(dim=1)


def adaptive_product_fusion(
    teacher_prob: torch.Tensor,
    graph_prob: torch.Tensor,
    strength: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("fusion strength must be in [0, 1]")
    num_classes = teacher_prob.size(1)
    confidence = 1.0 - entropy(graph_prob, eps) / math.log(num_classes)
    weight = (strength * confidence.clamp(0.0, 1.0)).unsqueeze(1)
    fused_log = (1.0 - weight) * teacher_prob.clamp_min(eps).log() + weight * graph_prob.clamp_min(eps).log()
    fused = torch.softmax(fused_log, dim=1)
    return fused, weight.squeeze(1)


def analyze(path: Path, args: argparse.Namespace) -> dict[str, object]:
    data = np.load(path)
    required = {"feature", "clip_feature", "source_probs", "clip_probs", "label"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing keys: {', '.join(missing)}")

    labels = torch.from_numpy(data["label"]).long()
    task_features = torch.from_numpy(data["feature"])
    clip_features = torch.from_numpy(data["clip_feature"])
    source_prob = torch.from_numpy(data["source_probs"]).float()
    clip_prob = torch.from_numpy(data["clip_probs"]).float()
    both_source = prior_calibrate(source_prob, args.calib_power)
    both_clip = prior_calibrate(clip_prob, args.calib_power)
    both_teacher = (both_source + both_clip) / 2
    source_label = both_source.argmax(dim=1)
    clip_label = both_clip.argmax(dim=1)

    _, _, graph_post, anchors = dual_space_diffusion(
        task_features,
        clip_features,
        both_source,
        both_clip,
        source_label,
        clip_label,
        anchor_ratio=args.anchor_ratio,
        anchor_min_per_class=args.anchor_min_per_class,
        k=args.graph_k,
        temperature=args.temperature,
        alpha=args.alpha,
        steps=args.steps,
        chunk_size=args.chunk_size,
    )
    fused_teacher, graph_weight = adaptive_product_fusion(
        both_teacher, graph_post, args.fusion_strength
    )

    conflict = source_label != clip_label
    both_acc = accuracy(both_teacher, labels)
    fused_acc = accuracy(fused_teacher, labels)
    delta = round(fused_acc - both_acc, 4)
    graph_acc = accuracy(graph_post, labels)

    return {
        "task": path.stem.replace("_conflicts", ""),
        "samples": int(labels.numel()),
        "anchors": int(anchors.sum().item()),
        "anchor_accuracy": accuracy(both_teacher, labels, anchors),
        "conflicts": int(conflict.sum().item()),
        "conflict_rate": round(100.0 * float(conflict.float().mean().item()), 4),
        "both_prior_teacher_accuracy": both_acc,
        "graph_posterior_accuracy": graph_acc,
        "fused_teacher_accuracy": fused_acc,
        "fused_minus_both_prior": delta,
        "both_prior_conflict_accuracy": accuracy(both_teacher, labels, conflict),
        "graph_conflict_accuracy": accuracy(graph_post, labels, conflict),
        "fused_conflict_accuracy": accuracy(fused_teacher, labels, conflict),
        "mean_graph_weight": round(float(graph_weight.mean().item()), 6),
        "max_graph_weight": round(float(graph_weight.max().item()), 6),
        "graph_entropy_mean": round(float(entropy(graph_post).mean().item()), 6),
        "changed_teacher_top1": int((both_teacher.argmax(dim=1) != fused_teacher.argmax(dim=1)).sum().item()),
        "pass_task_gate": bool(delta >= args.min_task_delta),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.glob}")
    tasks = [analyze(path, args) for path in paths]
    pass_tasks = sum(1 for task in tasks if task["pass_task_gate"])
    mean_delta = round(float(np.mean([task["fused_minus_both_prior"] for task in tasks])), 4)
    output = {
        "config": vars(args),
        "decision": "pass_training_gate"
        if pass_tasks >= args.min_pass_tasks and mean_delta > 0
        else "fail_training_gate",
        "reason": (
            "Graph-teacher fusion improves both_prior on enough probe tasks."
            if pass_tasks >= args.min_pass_tasks and mean_delta > 0
            else "Graph-teacher fusion does not improve both_prior reliably enough."
        ),
        "pass_tasks": int(pass_tasks),
        "mean_fused_minus_both_prior": mean_delta,
        "tasks": tasks,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
