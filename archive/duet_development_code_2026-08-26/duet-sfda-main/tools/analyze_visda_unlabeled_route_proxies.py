#!/usr/bin/env python
"""Test label-free class reliability proxies against oracle VisDA diagnostics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

try:
    from tools.analyze_temporal_conflict_dynamics import load_cycles, teacher_label
except ModuleNotFoundError:
    from analyze_temporal_conflict_dynamics import load_cycles, teacher_label


PROXY_FIELDS = [
    "teacher_confidence",
    "teacher_margin",
    "temporal_probability_agreement",
    "temporal_top1_stability",
    "candidate_containment_rate",
    "source_support_rate",
    "clip_support_rate",
    "base_teacher_agreement_rate",
    "graph_intervention_rate",
    "agreement_anchor_precision",
    "agreement_anchor_recall",
    "conflict_opportunity_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--classwise-oracle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--min-spearman", type=float, default=0.5)
    parser.add_argument("--min-topk-overlap", type=int, default=3)
    return parser.parse_args()


def mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def rate(mask: np.ndarray, denom_mask: np.ndarray) -> float:
    denom = int(np.sum(denom_mask))
    return float(np.sum(mask & denom_mask) / denom) if denom else 0.0


def rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.shape[0], dtype=np.float64)
    start = 0
    while start < order.shape[0]:
        end = start + 1
        while end < order.shape[0] and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(values: list[float], targets: list[float]) -> float:
    value_ranks = rankdata(values)
    target_ranks = rankdata(targets)
    if np.std(value_ranks) == 0 or np.std(target_ranks) == 0:
        return 0.0
    return float(np.corrcoef(value_ranks, target_ranks)[0, 1])


def build_proxy_rows(
    cycles: list[dict[str, np.ndarray]], class_names: list[str]
) -> list[dict[str, object]]:
    if len(cycles) < 2:
        raise ValueError("Route proxy analysis needs at least two cycles")
    first, previous, final = cycles[0], cycles[-2], cycles[-1]
    source = final["source_label"].astype(np.int64)
    clip = final["clip_label"].astype(np.int64)
    teacher = teacher_label(final).astype(np.int64)
    previous_teacher = teacher_label(previous).astype(np.int64)
    teacher_prob = final["teacher_prob"].astype(np.float64)
    previous_prob = previous["teacher_prob"].astype(np.float64)
    base_prob = (
        final["task_prob"].astype(np.float64)
        + final["clip_prob"].astype(np.float64)
    ) / 2.0
    base_label = base_prob.argmax(axis=1)
    initial_conflict = first["source_label"] != first["clip_label"]
    final_agreement = source == clip
    stable = previous_teacher == teacher
    sorted_prob = np.sort(teacher_prob, axis=1)
    confidence = sorted_prob[:, -1]
    margin = sorted_prob[:, -1] - sorted_prob[:, -2]
    probability_agreement = 1.0 - 0.5 * np.abs(teacher_prob - previous_prob).sum(axis=1)

    rows = []
    for index, name in enumerate(class_names):
        predicted = teacher == index
        conflict_predicted = predicted & initial_conflict
        selected = conflict_predicted & stable
        agreement_predicted = predicted & final_agreement
        agreement_class = final_agreement & (source == index)
        rows.append(
            {
                "class_index": index,
                "class": name,
                "predicted_samples": int(np.sum(predicted)),
                "stable_conflicts": int(np.sum(selected)),
                "teacher_confidence": mean_or_zero(confidence[selected]),
                "teacher_margin": mean_or_zero(margin[selected]),
                "temporal_probability_agreement": mean_or_zero(
                    probability_agreement[conflict_predicted]
                ),
                "temporal_top1_stability": rate(stable, conflict_predicted),
                "candidate_containment_rate": rate(
                    (teacher == source) | (teacher == clip), selected
                ),
                "source_support_rate": rate(teacher == source, selected),
                "clip_support_rate": rate(teacher == clip, selected),
                "base_teacher_agreement_rate": rate(teacher == base_label, selected),
                "graph_intervention_rate": rate(teacher != base_label, selected),
                "agreement_anchor_precision": rate(source == index, agreement_predicted),
                "agreement_anchor_recall": rate(teacher == index, agreement_class),
                "conflict_opportunity_rate": rate(initial_conflict, predicted),
            }
        )
    return rows


def evaluate_proxies(
    rows: list[dict[str, object]],
    oracle_gain: dict[str, float],
    oracle_supported: set[str],
    min_spearman: float = 0.5,
    min_topk_overlap: int = 3,
) -> dict[str, object]:
    class_names = [str(row["class"]) for row in rows]
    targets = [float(oracle_gain[name]) for name in class_names]
    top_k = len(oracle_supported)
    evaluations = []
    for field in PROXY_FIELDS:
        values = [float(row[field]) for row in rows]
        ranked = sorted(
            zip(class_names, values), key=lambda item: item[1], reverse=True
        )
        selected = [name for name, _ in ranked[:top_k]]
        overlap = len(set(selected) & oracle_supported)
        rho = spearman(values, targets)
        passed = rho >= min_spearman and overlap >= min_topk_overlap
        evaluations.append(
            {
                "proxy": field,
                "spearman_vs_oracle_gain": round(rho, 6),
                "top_k": top_k,
                "top_classes": selected,
                "top_k_overlap": overlap,
                "passes_proxy_gate": bool(passed),
            }
        )
    evaluations.sort(
        key=lambda item: (
            item["passes_proxy_gate"],
            item["spearman_vs_oracle_gain"],
            item["top_k_overlap"],
        ),
        reverse=True,
    )
    passing = [item for item in evaluations if item["passes_proxy_gate"]]
    return {
        "decision": (
            "supports_unlabeled_class_router"
            if passing
            else "rejects_unlabeled_class_router"
        ),
        "criteria": {
            "min_spearman": min_spearman,
            "min_top_k_overlap": min_topk_overlap,
            "oracle_top_k": top_k,
        },
        "oracle_supported_classes": sorted(oracle_supported),
        "best_proxy": evaluations[0],
        "passing_proxies": [item["proxy"] for item in passing],
        "proxy_evaluations": evaluations,
        "next": (
            "implement a class-conditional temporal residual using the best predeclared proxy"
            if passing
            else "do not use oracle class identities; test PL/GTR stable cycles 3"
        ),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least two cycle files: {args.glob}")
    oracle = json.loads(Path(args.classwise_oracle).read_text())
    oracle_rows = oracle["predicted_class_routes"]
    class_names = [str(row["class"]) for row in oracle_rows]
    oracle_gain = {
        str(row["class"]): float(row["teacher_minus_clip_pp"])
        for row in oracle_rows
    }
    oracle_supported = set(oracle["supported_predicted_classes"])
    rows = build_proxy_rows(load_cycles(paths), class_names)
    result = evaluate_proxies(
        rows,
        oracle_gain,
        oracle_supported,
        min_spearman=args.min_spearman,
        min_topk_overlap=args.min_topk_overlap,
    )
    result["selection_warning"] = (
        "proxy features are label-free, but this gate compares them with a validation-label "
        "oracle and is diagnostic only"
    )
    result["class_proxies"] = rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
