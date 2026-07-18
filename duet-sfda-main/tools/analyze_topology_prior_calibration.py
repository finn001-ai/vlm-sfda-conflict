#!/usr/bin/env python
"""Probe topology-prior calibration before adaptation training.

Ground-truth labels are used only for reporting the probe metrics. The method
itself estimates a class-level prior from dual-space graph diffusion and never
selects or relabels individual conflict samples.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.conflict_diffusion import topology_prior_calibrate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/plmatch/diagnostics/*C_conflicts.npz",
    )
    parser.add_argument(
        "--out",
        default="output/uda/office-home/topology_prior_probe.json",
    )
    parser.add_argument("--graph-k", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--anchor-ratio", type=float, default=0.5)
    parser.add_argument("--anchor-min-per-class", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--calib-power", type=float, default=0.5)
    parser.add_argument("--min-task-delta", type=float, default=0.05)
    parser.add_argument("--min-pass-tasks", type=int, default=2)
    return parser.parse_args()


def prior_calibrate(prob: torch.Tensor, power: float, eps: float = 1e-6) -> torch.Tensor:
    prior = prob.mean(dim=0).clamp_min(eps)
    calibrated = prob / prior.pow(power)
    return calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(eps)


def accuracy(prob: torch.Tensor, labels: torch.Tensor) -> float:
    pred = prob.argmax(dim=1)
    return round(100.0 * float((pred == labels).float().mean().item()), 4)


def agreement_rate(source_prob: torch.Tensor, clip_prob: torch.Tensor) -> float:
    agree = source_prob.argmax(dim=1) == clip_prob.argmax(dim=1)
    return round(100.0 * float(agree.float().mean().item()), 4)


def entropy(prob: torch.Tensor, eps: float = 1e-6) -> float:
    prob = prob.clamp_min(eps)
    return round(float(-(prob * prob.log()).sum().item()), 6)


def analyze(path: Path, args: argparse.Namespace) -> dict[str, int | float | str | bool]:
    data = np.load(path)
    required = {"feature", "clip_feature", "source_probs", "clip_probs", "label"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing keys: {', '.join(missing)}")

    labels = torch.from_numpy(data["label"]).long()
    source_prob = torch.from_numpy(data["source_probs"]).float()
    clip_prob = torch.from_numpy(data["clip_probs"]).float()
    raw_mix = (source_prob + clip_prob) / 2

    both_source = prior_calibrate(source_prob, args.calib_power)
    both_clip = prior_calibrate(clip_prob, args.calib_power)
    both_mix = (both_source + both_clip) / 2

    topo_source, topo_clip, topo_mix, graph_prior, anchors = topology_prior_calibrate(
        torch.from_numpy(data["feature"]),
        torch.from_numpy(data["clip_feature"]),
        source_prob,
        clip_prob,
        source_prob.argmax(dim=1),
        clip_prob.argmax(dim=1),
        power=args.calib_power,
        anchor_ratio=args.anchor_ratio,
        anchor_min_per_class=args.anchor_min_per_class,
        k=args.graph_k,
        temperature=args.temperature,
        alpha=args.alpha,
        steps=args.steps,
        chunk_size=args.chunk_size,
    )

    topo_acc = accuracy(topo_mix, labels)
    both_acc = accuracy(both_mix, labels)
    delta = round(topo_acc - both_acc, 4)
    anchor_correct = source_prob.argmax(dim=1)[anchors] == labels[anchors]

    return {
        "task": path.stem.replace("_conflicts", ""),
        "samples": int(labels.numel()),
        "anchors": int(anchors.sum().item()),
        "anchor_accuracy": round(100.0 * float(anchor_correct.float().mean().item()), 4)
        if anchors.any()
        else 0.0,
        "raw_source_accuracy": accuracy(source_prob, labels),
        "raw_clip_accuracy": accuracy(clip_prob, labels),
        "raw_mix_accuracy": accuracy(raw_mix, labels),
        "both_prior_source_accuracy": accuracy(both_source, labels),
        "both_prior_clip_accuracy": accuracy(both_clip, labels),
        "both_prior_mix_accuracy": both_acc,
        "topo_prior_source_accuracy": accuracy(topo_source, labels),
        "topo_prior_clip_accuracy": accuracy(topo_clip, labels),
        "topo_prior_mix_accuracy": topo_acc,
        "topo_minus_both_prior": delta,
        "raw_agreement_rate": agreement_rate(source_prob, clip_prob),
        "both_prior_agreement_rate": agreement_rate(both_source, both_clip),
        "topo_prior_agreement_rate": agreement_rate(topo_source, topo_clip),
        "graph_prior_min": round(float(graph_prior.min().item()), 6),
        "graph_prior_max": round(float(graph_prior.max().item()), 6),
        "graph_prior_entropy": entropy(graph_prior),
        "pass_task_gate": bool(delta >= args.min_task_delta),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    tasks = [analyze(path, args) for path in paths]
    pass_tasks = sum(1 for task in tasks if task["pass_task_gate"])
    mean_delta = round(float(np.mean([task["topo_minus_both_prior"] for task in tasks])), 4)
    output = {
        "config": vars(args),
        "decision": "pass_training_gate"
        if pass_tasks >= args.min_pass_tasks and mean_delta > 0
        else "fail_training_gate",
        "reason": (
            "Topology-prior mixed top-1 beats both_prior on enough probe tasks."
            if pass_tasks >= args.min_pass_tasks and mean_delta > 0
            else "Topology-prior mixed top-1 does not beat both_prior strongly enough."
        ),
        "pass_tasks": int(pass_tasks),
        "mean_topo_minus_both_prior": mean_delta,
        "tasks": tasks,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
