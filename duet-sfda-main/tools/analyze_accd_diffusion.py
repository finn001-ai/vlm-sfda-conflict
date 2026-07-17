#!/usr/bin/env python
"""Oracle-free ACCD inference with labels used only for diagnostic metrics."""

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

from src.utils.conflict_diffusion import (  # noqa: E402
    conflict_diffusion_evidence,
    dual_space_diffusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/plmatch/diagnostics/*_conflicts.npz",
    )
    parser.add_argument("--out", default="output/uda/office-home/accd_diffusion_probe.json")
    parser.add_argument("--graph-k", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--anchor-ratio", type=float, default=0.5)
    parser.add_argument("--anchor-min-per-class", type=int, default=5)
    parser.add_argument("--candidate-mass", type=float, default=0.6)
    parser.add_argument("--candidate-margin", type=float, default=0.1)
    parser.add_argument("--calib-power", type=float, default=0.5)
    return parser.parse_args()


def pct(numerator: int | float, denominator: int) -> float:
    return round(100.0 * float(numerator) / denominator, 4) if denominator else 0.0


def prior_calibrate(prob: torch.Tensor, power: float) -> torch.Tensor:
    prior = prob.mean(dim=0).clamp_min(1e-6)
    calibrated = prob / prior.pow(power)
    return calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(1e-6)


def analyze(path: Path, args: argparse.Namespace) -> dict[str, int | float | str | bool]:
    data = np.load(path)
    if "clip_feature" not in data:
        raise KeyError(
            f"{path} has no clip_feature. Rerun tools/export_conflict_diagnostics.py."
        )

    task_features = torch.from_numpy(data["feature"])
    clip_features = torch.from_numpy(data["clip_feature"])
    source_prob = prior_calibrate(torch.from_numpy(data["source_probs"]).float(), args.calib_power)
    clip_prob = prior_calibrate(torch.from_numpy(data["clip_probs"]).float(), args.calib_power)
    labels = torch.from_numpy(data["label"]).long()
    source_label = source_prob.argmax(dim=1)
    clip_label = clip_prob.argmax(dim=1)

    task_post, clip_post, fused, anchors = dual_space_diffusion(
        task_features,
        clip_features,
        source_prob,
        clip_prob,
        source_label,
        clip_label,
        anchor_ratio=args.anchor_ratio,
        anchor_min_per_class=args.anchor_min_per_class,
        k=args.graph_k,
        temperature=args.temperature,
        alpha=args.alpha,
        steps=args.steps,
    )
    evidence = conflict_diffusion_evidence(
        task_post,
        clip_post,
        fused,
        source_label,
        clip_label,
        candidate_mass_threshold=args.candidate_mass,
        candidate_margin_threshold=args.candidate_margin,
    )

    conflict = evidence["conflict"]
    eligible = evidence["eligible"]
    outside = evidence["outside_candidate"]
    graph_label = evidence["graph_label"]
    candidate_contains_truth = (labels == source_label) | (labels == clip_label)

    graph_correct = graph_label == labels
    clip_correct = clip_label == labels
    source_correct = source_label == labels
    resolved_to_source = eligible & (graph_label == source_label)
    resolved_to_clip = eligible & (graph_label == clip_label)
    clip_to_graph_corrections = eligible & (~clip_correct) & graph_correct
    clip_to_graph_degradations = eligible & clip_correct & (~graph_correct)
    net_gain = int(graph_correct[eligible].sum() - clip_correct[eligible].sum())
    outside_true = ~candidate_contains_truth[outside]
    eligible_accuracy = pct(graph_correct[eligible].sum(), int(eligible.sum()))
    eligible_clip_accuracy = pct(clip_correct[eligible].sum(), int(eligible.sum()))
    conflict_count = int(conflict.sum())
    eligible_count = int(eligible.sum())

    return {
        "task": path.stem.replace("_conflicts", ""),
        "samples": int(labels.numel()),
        "anchors": int(anchors.sum()),
        "anchor_accuracy": pct((source_label[anchors] == labels[anchors]).sum(), int(anchors.sum())),
        "conflicts": conflict_count,
        "conflict_rate": pct(conflict_count, int(labels.numel())),
        "candidate_recall_on_conflicts": pct(candidate_contains_truth[conflict].sum(), conflict_count),
        "eligible": eligible_count,
        "eligible_conflict_coverage": pct(eligible_count, conflict_count),
        "eligible_accuracy": eligible_accuracy,
        "eligible_clip_accuracy": eligible_clip_accuracy,
        "eligible_source_accuracy": pct(source_correct[eligible].sum(), eligible_count),
        "resolved_to_source": int(resolved_to_source.sum()),
        "resolved_to_clip": int(resolved_to_clip.sum()),
        "clip_to_graph_corrections": int(clip_to_graph_corrections.sum()),
        "clip_to_graph_degradations": int(clip_to_graph_degradations.sum()),
        "net_correct_gain_over_clip": net_gain,
        "projected_full_accuracy_gain": round(100.0 * net_gain / labels.numel(), 4),
        "outside_candidate": int(outside.sum()),
        "outside_candidate_precision": pct(outside_true.sum(), int(outside.sum())),
        "pass_training_gate": bool(
            eligible_count >= max(20, int(0.05 * conflict_count))
            and net_gain > 0
            and eligible_accuracy > eligible_clip_accuracy
        ),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    results = [analyze(path, args) for path in paths]
    output = {"config": vars(args), "tasks": results}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
