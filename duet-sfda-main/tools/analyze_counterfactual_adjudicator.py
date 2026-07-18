#!/usr/bin/env python
"""Probe a label-free conflict adjudicator trained on synthetic anchor conflicts."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.conflict_diffusion import (  # noqa: E402
    dual_space_diffusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/plmatch/diagnostics/*C_conflicts.npz",
    )
    parser.add_argument(
        "--out",
        default="output/uda/office-home/counterfactual_adjudicator_probe.json",
    )
    parser.add_argument("--graph-k", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--anchor-ratio", type=float, default=0.5)
    parser.add_argument("--anchor-min-per-class", type=int, default=5)
    parser.add_argument("--calib-power", type=float, default=0.5)
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def prior_calibrate(prob: torch.Tensor, power: float) -> torch.Tensor:
    prior = prob.mean(dim=0).clamp_min(1e-6)
    calibrated = prob / prior.pow(power)
    return calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(1e-6)


def class_prototype_scores(
    features: np.ndarray,
    anchors: np.ndarray,
    anchor_labels: np.ndarray,
    class_num: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = normalize_rows(features.astype(np.float32))
    prototypes = np.zeros((class_num, features.shape[1]), dtype=np.float32)
    counts = np.zeros(class_num, dtype=np.float32)
    for index in np.flatnonzero(anchors):
        label = anchor_labels[index]
        prototypes[label] += features[index]
        counts[label] += 1.0
    valid = counts > 0
    prototypes[valid] /= counts[valid, None]
    prototypes[valid] = normalize_rows(prototypes[valid])
    scores = features @ prototypes.T
    scores[:, ~valid] = 0.0
    return scores, counts


def top_alternative(prob: np.ndarray, labels: np.ndarray) -> np.ndarray:
    masked = prob.copy()
    masked[np.arange(labels.size), labels] = -np.inf
    return masked.argmax(axis=1)


def pair_features(
    source_prob: np.ndarray,
    clip_prob: np.ndarray,
    task_graph: np.ndarray,
    clip_graph: np.ndarray,
    fused_graph: np.ndarray,
    task_proto: np.ndarray,
    clip_proto: np.ndarray,
    anchor_counts: np.ndarray,
    rows: np.ndarray,
    source_candidate: np.ndarray,
    clip_candidate: np.ndarray,
    *,
    corrupt: str | None = None,
) -> np.ndarray:
    eps = 1e-8
    s = source_candidate
    c = clip_candidate
    ps_s = source_prob[rows, s].copy()
    ps_c = source_prob[rows, c].copy()
    pc_s = clip_prob[rows, s].copy()
    pc_c = clip_prob[rows, c].copy()
    if corrupt == "source":
        ps_s, ps_c = ps_c.copy(), ps_s.copy()
    elif corrupt == "clip":
        pc_s, pc_c = pc_c.copy(), pc_s.copy()
    elif corrupt is not None:
        raise ValueError(f"Unknown corruption side: {corrupt}")

    source_entropy = -(source_prob[rows] * np.log(source_prob[rows] + eps)).sum(axis=1)
    clip_entropy = -(clip_prob[rows] * np.log(clip_prob[rows] + eps)).sum(axis=1)
    return np.column_stack(
        [
            np.log(ps_s + eps),
            np.log(ps_c + eps),
            np.log(pc_s + eps),
            np.log(pc_c + eps),
            task_graph[rows, s],
            task_graph[rows, c],
            clip_graph[rows, s],
            clip_graph[rows, c],
            fused_graph[rows, s],
            fused_graph[rows, c],
            task_proto[rows, s],
            task_proto[rows, c],
            clip_proto[rows, s],
            clip_proto[rows, c],
            np.log1p(anchor_counts[s]),
            np.log1p(anchor_counts[c]),
            source_entropy,
            clip_entropy,
        ]
    ).astype(np.float32)


def build_counterfactual_training(
    source_prob: np.ndarray,
    clip_prob: np.ndarray,
    task_graph: np.ndarray,
    clip_graph: np.ndarray,
    fused_graph: np.ndarray,
    task_proto: np.ndarray,
    clip_proto: np.ndarray,
    anchor_counts: np.ndarray,
    anchors: np.ndarray,
    anchor_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(anchors)
    labels = anchor_labels[rows]
    source_alt = top_alternative(source_prob[rows], labels)
    clip_alt = top_alternative(clip_prob[rows], labels)

    source_correct = pair_features(
        source_prob,
        clip_prob,
        task_graph,
        clip_graph,
        fused_graph,
        task_proto,
        clip_proto,
        anchor_counts,
        rows,
        labels,
        clip_alt,
        corrupt="clip",
    )
    clip_correct = pair_features(
        source_prob,
        clip_prob,
        task_graph,
        clip_graph,
        fused_graph,
        task_proto,
        clip_proto,
        anchor_counts,
        rows,
        source_alt,
        labels,
        corrupt="source",
    )
    x_train = np.concatenate([source_correct, clip_correct], axis=0)
    y_train = np.concatenate(
        [np.ones(rows.size, dtype=np.int64), np.zeros(rows.size, dtype=np.int64)]
    )
    groups = np.concatenate([rows, rows])
    return x_train, y_train, groups


def pct(numerator: int | float, denominator: int) -> float:
    return round(100.0 * float(numerator) / denominator, 4) if denominator else 0.0


def analyze(path: Path, args: argparse.Namespace) -> dict[str, int | float | str | bool]:
    data = np.load(path)
    source_prob_t = prior_calibrate(
        torch.from_numpy(data["source_probs"]).float(), args.calib_power
    )
    clip_prob_t = prior_calibrate(
        torch.from_numpy(data["clip_probs"]).float(), args.calib_power
    )
    source_label_t = source_prob_t.argmax(dim=1)
    clip_label_t = clip_prob_t.argmax(dim=1)
    task_graph_t, clip_graph_t, fused_t, anchors_t = dual_space_diffusion(
        torch.from_numpy(data["feature"]),
        torch.from_numpy(data["clip_feature"]),
        source_prob_t,
        clip_prob_t,
        source_label_t,
        clip_label_t,
        anchor_ratio=args.anchor_ratio,
        anchor_min_per_class=args.anchor_min_per_class,
        k=args.graph_k,
        temperature=args.temperature,
        alpha=args.alpha,
        steps=args.steps,
    )

    source_prob = source_prob_t.numpy()
    clip_prob = clip_prob_t.numpy()
    source_label = source_label_t.numpy()
    clip_label = clip_label_t.numpy()
    anchors = anchors_t.numpy()
    task_graph = task_graph_t.numpy()
    clip_graph = clip_graph_t.numpy()
    fused = fused_t.numpy()
    labels = data["label"].astype(np.int64)
    class_num = source_prob.shape[1]
    task_proto, anchor_counts = class_prototype_scores(
        data["feature"], anchors, source_label, class_num
    )
    clip_proto, _ = class_prototype_scores(
        data["clip_feature"], anchors, source_label, class_num
    )

    x_train, y_train, groups = build_counterfactual_training(
        source_prob,
        clip_prob,
        task_graph,
        clip_graph,
        fused,
        task_proto,
        clip_proto,
        anchor_counts,
        anchors,
        source_label,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    folds = min(5, int(anchors.sum()))
    cv_pred = cross_val_predict(
        model,
        x_train,
        y_train,
        groups=groups,
        cv=GroupKFold(n_splits=folds),
        method="predict",
    )
    synthetic_cv_accuracy = accuracy_score(y_train, cv_pred)
    model.fit(x_train, y_train)

    conflict_rows = np.flatnonzero(source_label != clip_label)
    x_conflict = pair_features(
        source_prob,
        clip_prob,
        task_graph,
        clip_graph,
        fused,
        task_proto,
        clip_proto,
        anchor_counts,
        conflict_rows,
        source_label[conflict_rows],
        clip_label[conflict_rows],
    )
    source_probability = model.predict_proba(x_conflict)[:, 1]
    choose_source = source_probability >= 0.5
    selected = np.where(
        choose_source, source_label[conflict_rows], clip_label[conflict_rows]
    )
    truth = labels[conflict_rows]
    source_correct = source_label[conflict_rows] == truth
    clip_correct = clip_label[conflict_rows] == truth
    selected_correct = selected == truth
    corrections = choose_source & source_correct & (~clip_correct)
    degradations = choose_source & (~source_correct) & clip_correct
    net_gain = int(corrections.sum() - degradations.sum())
    candidate_recall = source_correct | clip_correct

    return {
        "task": path.stem.replace("_conflicts", ""),
        "samples": int(labels.size),
        "anchors": int(anchors.sum()),
        "anchor_oracle_accuracy": pct(
            (source_label[anchors] == labels[anchors]).sum(), int(anchors.sum())
        ),
        "synthetic_train_samples": int(y_train.size),
        "synthetic_group_cv_accuracy": round(100.0 * synthetic_cv_accuracy, 4),
        "conflicts": int(conflict_rows.size),
        "candidate_recall": pct(candidate_recall.sum(), int(conflict_rows.size)),
        "always_source_accuracy": pct(source_correct.sum(), int(conflict_rows.size)),
        "always_clip_accuracy": pct(clip_correct.sum(), int(conflict_rows.size)),
        "adjudicator_accuracy": pct(selected_correct.sum(), int(conflict_rows.size)),
        "source_selection_rate": pct(choose_source.sum(), int(conflict_rows.size)),
        "corrections_over_clip": int(corrections.sum()),
        "degradations_from_clip": int(degradations.sum()),
        "net_correct_gain_over_clip": net_gain,
        "projected_full_accuracy_gain": round(100.0 * net_gain / labels.size, 4),
        "pass_training_gate": bool(net_gain > 0 and selected_correct.sum() > clip_correct.sum()),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(value) for value in glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(
            f"No files matched {args.glob}. Run tools/run_office_home_accd_diffusion_probe.sh first."
        )
    results = [analyze(path, args) for path in paths]
    output = {"config": vars(args), "tasks": results}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
