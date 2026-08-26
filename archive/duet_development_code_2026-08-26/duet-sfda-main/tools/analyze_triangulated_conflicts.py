#!/usr/bin/env python
"""Analyze conflict samples with candidate-set and target-structure evidence.

This is an offline diagnostic tool. It reads the CSV and NPZ files exported by
`export_conflict_diagnostics.py` and evaluates whether feature-structure
signals can improve over naive confidence or always trusting CLIP.

Ground-truth labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/plmatch/diagnostics/*_conflicts.csv",
        help="Glob for per-sample diagnostic CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="output/uda/office-home/triangulated_conflicts",
        help="Directory for outputs.",
    )
    parser.add_argument("--proto-conf", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=20, help="Number of neighbors for support.")
    parser.add_argument("--w-conf", type=float, default=1.0)
    parser.add_argument("--w-proto", type=float, default=1.0)
    parser.add_argument("--w-neigh", type=float, default=1.0)
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Reject conflicts when triangulated source/CLIP score gap is below this value.",
    )
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"true", "1", "yes"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["index"]))
    return rows


def task_name(path: Path) -> str:
    return path.name.split("_conflicts.csv")[0]


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def pct(num: float, denom: float) -> float:
    return round(100.0 * num / denom, 4) if denom else 0.0


def eval_choices(rows: list[dict[str, str]], conflict_idx: np.ndarray, choices: list[str | None]) -> dict[str, float | int]:
    assert len(conflict_idx) == len(choices)
    selected = [(rows[i], choice) for i, choice in zip(conflict_idx, choices) if choice is not None]
    rejected = [(rows[i], choice) for i, choice in zip(conflict_idx, choices) if choice is None]
    correct = 0
    for row, choice in selected:
        correct += parse_bool(row[f"{choice}_correct"])
    rejected_harmful = sum(
        (not parse_bool(row["source_correct"])) and (not parse_bool(row["clip_correct"]))
        for row, _ in rejected
    )
    return {
        "selected": len(selected),
        "rejected": len(rejected),
        "coverage": pct(len(selected), len(conflict_idx)),
        "accuracy": pct(correct, len(selected)),
        "rejected_harmful_rate": pct(rejected_harmful, len(rejected)),
    }


def build_prototypes(
    rows: list[dict[str, str]],
    features: np.ndarray,
    class_num: int,
    proto_conf: float,
) -> tuple[np.ndarray, np.ndarray]:
    reliable = []
    for i, row in enumerate(rows):
        if not parse_bool(row["agree"]):
            continue
        if float(row["source_conf"]) < proto_conf or float(row["clip_conf"]) < proto_conf:
            continue
        reliable.append(i)

    prototypes = np.zeros((class_num, features.shape[1]), dtype=np.float32)
    counts = np.zeros(class_num, dtype=np.int64)
    for i in reliable:
        cls = int(rows[i]["source_pred"])
        prototypes[cls] += features[i]
        counts[cls] += 1

    valid = counts > 0
    prototypes[valid] /= counts[valid, None]
    prototypes[valid] = normalize(prototypes[valid])
    return prototypes, valid


def prototype_scores(features: np.ndarray, prototypes: np.ndarray, valid: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.zeros(labels.shape[0], dtype=np.float32)
    valid_rows = valid[labels]
    if np.any(valid_rows):
        scores[valid_rows] = np.sum(features[valid_rows] * prototypes[labels[valid_rows]], axis=1)
    return (scores + 1.0) / 2.0


def neighborhood_support(
    rows: list[dict[str, str]],
    features: np.ndarray,
    source_pred: np.ndarray,
    clip_pred: np.ndarray,
    conflict_idx: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    sim = features @ features.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, features.shape[0] - 1)
    nn_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]

    agreement_label = np.full(len(rows), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        if parse_bool(row["agree"]):
            agreement_label[i] = int(row["source_pred"])

    source_support = np.zeros(len(conflict_idx), dtype=np.float32)
    clip_support = np.zeros(len(conflict_idx), dtype=np.float32)
    for out_i, sample_i in enumerate(conflict_idx):
        neigh = nn_idx[sample_i]
        valid = agreement_label[neigh] >= 0
        if not np.any(valid):
            continue
        neigh_labels = agreement_label[neigh[valid]]
        source_support[out_i] = np.mean(neigh_labels == source_pred[sample_i])
        clip_support[out_i] = np.mean(neigh_labels == clip_pred[sample_i])
    return source_support, clip_support


def analyze_pair(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = load_csv(path)
    npz_path = path.with_suffix(".npz")
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing NPZ for {path}. Rerun export_conflict_diagnostics.py after latest update.")

    data = np.load(npz_path)
    features = normalize(data["feature"].astype(np.float32))
    labels = data["label"].astype(np.int64)
    source_pred = data["source_pred"].astype(np.int64)
    clip_pred = data["clip_pred"].astype(np.int64)
    class_num = int(max(source_pred.max(), clip_pred.max(), labels.max()) + 1)

    conflict_idx = np.array([i for i, row in enumerate(rows) if not parse_bool(row["agree"])], dtype=np.int64)
    useful = np.array([
        parse_bool(rows[i]["source_correct"]) != parse_bool(rows[i]["clip_correct"])
        for i in conflict_idx
    ])
    candidate_recall = np.mean((labels[conflict_idx] == source_pred[conflict_idx]) | (labels[conflict_idx] == clip_pred[conflict_idx]))

    prototypes, valid_proto = build_prototypes(rows, features, class_num, args.proto_conf)
    source_proto = prototype_scores(features, prototypes, valid_proto, source_pred)
    clip_proto = prototype_scores(features, prototypes, valid_proto, clip_pred)
    source_neigh, clip_neigh = neighborhood_support(rows, features, source_pred, clip_pred, conflict_idx, args.k)

    source_conf = np.array([float(row["source_conf"]) for row in rows], dtype=np.float32)
    clip_conf = np.array([float(row["clip_conf"]) for row in rows], dtype=np.float32)

    def choices_from_scores(source_score: np.ndarray, clip_score: np.ndarray, reject_gap: float = 0.0) -> list[str | None]:
        choices: list[str | None] = []
        for s, c in zip(source_score, clip_score):
            if abs(float(s - c)) < reject_gap:
                choices.append(None)
            else:
                choices.append("source" if s >= c else "clip")
        return choices

    source_score = (
        args.w_conf * source_conf[conflict_idx]
        + args.w_proto * source_proto[conflict_idx]
        + args.w_neigh * source_neigh
    )
    clip_score = (
        args.w_conf * clip_conf[conflict_idx]
        + args.w_proto * clip_proto[conflict_idx]
        + args.w_neigh * clip_neigh
    )

    always_source = ["source"] * len(conflict_idx)
    always_clip = ["clip"] * len(conflict_idx)
    higher_conf = choices_from_scores(source_conf[conflict_idx], clip_conf[conflict_idx])
    proto_only = choices_from_scores(source_proto[conflict_idx], clip_proto[conflict_idx])
    neigh_only = choices_from_scores(source_neigh, clip_neigh)
    triangulated = choices_from_scores(source_score, clip_score)
    triangulated_reject = choices_from_scores(source_score, clip_score, args.gap)

    always_clip_eval = eval_choices(rows, conflict_idx, always_clip)
    triangulated_eval = eval_choices(rows, conflict_idx, triangulated)
    triangulated_reject_eval = eval_choices(rows, conflict_idx, triangulated_reject)

    return {
        "task": task_name(path),
        "total_samples": len(rows),
        "conflict_samples": int(len(conflict_idx)),
        "conflict_rate": pct(len(conflict_idx), len(rows)),
        "useful_conflict_rate": pct(int(useful.sum()), len(conflict_idx)),
        "candidate_set_recall": round(float(candidate_recall * 100.0), 4),
        "always_source_acc": eval_choices(rows, conflict_idx, always_source)["accuracy"],
        "always_clip_acc": always_clip_eval["accuracy"],
        "higher_conf_acc": eval_choices(rows, conflict_idx, higher_conf)["accuracy"],
        "prototype_only_acc": eval_choices(rows, conflict_idx, proto_only)["accuracy"],
        "neighborhood_only_acc": eval_choices(rows, conflict_idx, neigh_only)["accuracy"],
        "triangulated_acc": triangulated_eval["accuracy"],
        "triangulated_minus_clip": round(float(triangulated_eval["accuracy"] - always_clip_eval["accuracy"]), 4),
        "triangulated_reject_acc": triangulated_reject_eval["accuracy"],
        "triangulated_reject_coverage": triangulated_reject_eval["coverage"],
        "triangulated_reject_harmful_in_rejected": triangulated_reject_eval["rejected_harmful_rate"],
    }


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "task",
        "conflict_rate",
        "candidate_set_recall",
        "always_clip_acc",
        "higher_conf_acc",
        "prototype_only_acc",
        "neighborhood_only_acc",
        "triangulated_acc",
        "triangulated_reject_acc",
        "triangulated_reject_coverage",
    ]
    lines = [
        "# Triangulated Conflict Analysis",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row[col]
            vals.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    paths = sorted(Path(p) for p in glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [analyze_pair(path, args) for path in paths]

    csv_path = out_dir / "triangulated_conflict_analysis.csv"
    json_path = out_dir / "triangulated_conflict_analysis.json"
    md_path = out_dir / "triangulated_conflict_analysis.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    macro = {}
    for key in rows[0]:
        if key == "task":
            continue
        values = [row[key] for row in rows if isinstance(row[key], (int, float))]
        macro[key] = mean(values)
    json_path.write_text(json.dumps({"tasks": rows, "macro": macro}, indent=2))
    write_markdown(rows, md_path)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")
    print(json.dumps({"macro": macro}, indent=2))


if __name__ == "__main__":
    main()
