#!/usr/bin/env python3
"""Audit harmful early-admitted pseudo-labels from DUET refresh snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "mix_label",
    "label_mask",
    "task_prob",
    "target_label",
    "sample_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an early DUET refresh with a later refresh and test whether "
            "a small, stable, high-confidence contradiction set is enriched for "
            "incorrect early pseudo-labels."
        )
    )
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--late", type=Path, required=True)
    parser.add_argument("--data-list", type=Path)
    parser.add_argument("--class-names", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-classes", type=Path, required=True)
    parser.add_argument("--out-samples", type=Path, required=True)
    parser.add_argument("--loss-fraction", type=float, default=0.10)
    parser.add_argument("--confidence-fraction", type=float, default=0.20)
    parser.add_argument("--stability-fraction", type=float, default=0.20)
    parser.add_argument("--min-suspicious-precision", type=float, default=0.70)
    parser.add_argument("--min-error-enrichment", type=float, default=3.0)
    parser.add_argument("--max-cut-ratio", type=float, default=0.01)
    parser.add_argument("--min-after-cut-coverage", type=float, default=0.955)
    return parser.parse_args()


def load_snapshot(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        missing = REQUIRED_KEYS.difference(payload.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        return {key: payload[key] for key in payload.files}


def validate_fraction(name: str, value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = probabilities.astype(np.float64, copy=False)
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def select_fraction(
    values: np.ndarray,
    eligible: np.ndarray,
    fraction: float,
    *,
    largest: bool,
) -> tuple[np.ndarray, float | None]:
    """Select an exact rank fraction, avoiding unstable quantile ties."""
    indices = np.flatnonzero(eligible)
    selected = np.zeros(values.shape[0], dtype=bool)
    if indices.size == 0:
        return selected, None
    count = max(1, int(math.ceil(indices.size * fraction)))
    order = np.argsort(values[indices], kind="stable")
    chosen = indices[order[-count:] if largest else order[:count]]
    selected[chosen] = True
    boundary = float(np.min(values[chosen]) if largest else np.max(values[chosen]))
    return selected, boundary


def jensen_shannon(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = normalize_probabilities(first)
    second = normalize_probabilities(second)
    mean = 0.5 * (first + second)
    first_kl = np.sum(first * (np.log(first) - np.log(mean)), axis=1)
    second_kl = np.sum(second * (np.log(second) - np.log(mean)), axis=1)
    return 0.5 * (first_kl + second_kl)


def percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def read_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Text file does not exist: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def analyze(
    early: dict[str, np.ndarray],
    late: dict[str, np.ndarray],
    *,
    loss_fraction: float,
    confidence_fraction: float,
    stability_fraction: float,
    min_suspicious_precision: float,
    min_error_enrichment: float,
    max_cut_ratio: float,
    min_after_cut_coverage: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    for name, value in (
        ("loss_fraction", loss_fraction),
        ("confidence_fraction", confidence_fraction),
        ("stability_fraction", stability_fraction),
    ):
        validate_fraction(name, value)

    early_index = early["sample_index"].astype(np.int64)
    late_index = late["sample_index"].astype(np.int64)
    if not np.array_equal(early_index, late_index):
        raise ValueError("Early and late snapshots do not have identical sample order")

    sample_count = early_index.size
    for snapshot_name, snapshot in (("early", early), ("late", late)):
        row_counts = {
            value.shape[0]
            for key, value in snapshot.items()
            if key not in {"cycle", "task", "phase"} and value.ndim > 0
        }
        if row_counts != {sample_count}:
            raise ValueError(
                f"{snapshot_name} snapshot has inconsistent rows: {sorted(row_counts)}"
            )

    early_selected = early["label_mask"].astype(bool)
    late_selected = late["label_mask"].astype(bool)
    if np.any(early_selected & ~late_selected):
        raise ValueError("Late snapshot violates DUET monotonic admission")

    early_label = early["mix_label"].astype(np.int64)
    target_label = early["target_label"].astype(np.int64)
    if not np.array_equal(target_label, late["target_label"].astype(np.int64)):
        raise ValueError("Target labels differ between snapshots")

    late_prob = normalize_probabilities(late["task_prob"])
    late_label = np.argmax(late_prob, axis=1)
    late_confidence = np.max(late_prob, axis=1)
    early_label_loss = -np.log(late_prob[np.arange(sample_count), early_label])

    if "strong_task_prob" not in late:
        raise ValueError(
            "Late snapshot lacks strong_task_prob; rerun with the audit-enabled code"
        )
    augmentation_js = jensen_shannon(late_prob, late["strong_task_prob"])
    contradiction = early_selected & (late_label != early_label)

    high_loss, loss_boundary = select_fraction(
        early_label_loss, early_selected, loss_fraction, largest=True
    )
    high_confidence, confidence_boundary = select_fraction(
        late_confidence, early_selected, confidence_fraction, largest=True
    )
    low_sensitivity, sensitivity_boundary = select_fraction(
        augmentation_js, early_selected, stability_fraction, largest=False
    )
    suspicious = (
        contradiction & high_loss & high_confidence & low_sensitivity
    )

    early_wrong = early_selected & (early_label != target_label)
    suspicious_wrong = suspicious & (early_label != target_label)
    suspicious_correct = suspicious & (early_label == target_label)

    early_selected_count = int(early_selected.sum())
    late_selected_count = int(late_selected.sum())
    suspicious_count = int(suspicious.sum())
    suspicious_wrong_count = int(suspicious_wrong.sum())
    suspicious_correct_count = int(suspicious_correct.sum())
    early_error_rate = ratio(int(early_wrong.sum()), early_selected_count)
    suspicious_error_rate = ratio(suspicious_wrong_count, suspicious_count)
    enrichment = (
        suspicious_error_rate / early_error_rate
        if early_error_rate > 0.0
        else 0.0
    )
    cut_ratio = ratio(suspicious_count, sample_count)
    after_cut_coverage = ratio(late_selected_count - suspicious_count, sample_count)

    checks = {
        "nonempty_suspicious_set": suspicious_count > 0,
        "suspicious_precision": (
            suspicious_error_rate >= min_suspicious_precision
        ),
        "error_enrichment": enrichment >= min_error_enrichment,
        "cut_ratio": cut_ratio <= max_cut_ratio,
        "after_cut_coverage": after_cut_coverage >= min_after_cut_coverage,
    }
    verdict = "PASS" if all(checks.values()) else "REJECT"

    report = {
        "verdict": verdict,
        "interpretation": (
            "Implement one proxy training run with a one-time early-admission audit."
            if verdict == "PASS"
            else "Do not implement or train Early Cutting on this evidence."
        ),
        "cycles": {
            "early": int(np.asarray(early.get("cycle", 0)).item()),
            "late": int(np.asarray(late.get("cycle", 0)).item()),
        },
        "counts": {
            "samples": sample_count,
            "early_selected": early_selected_count,
            "late_selected": late_selected_count,
            "early_wrong": int(early_wrong.sum()),
            "contradictions": int(contradiction.sum()),
            "suspicious": suspicious_count,
            "suspicious_wrong": suspicious_wrong_count,
            "suspicious_correct": suspicious_correct_count,
        },
        "metrics": {
            "early_selected_coverage_percent": percent(
                early_selected_count, sample_count
            ),
            "late_selected_coverage_percent": percent(
                late_selected_count, sample_count
            ),
            "early_selected_error_percent": percent(
                int(early_wrong.sum()), early_selected_count
            ),
            "suspicious_precision_percent": percent(
                suspicious_wrong_count, suspicious_count
            ),
            "error_enrichment": enrichment,
            "cut_ratio_percent": percent(suspicious_count, sample_count),
            "after_cut_coverage_percent": percent(
                late_selected_count - suspicious_count, sample_count
            ),
        },
        "selection": {
            "loss_fraction": loss_fraction,
            "confidence_fraction": confidence_fraction,
            "stability_fraction": stability_fraction,
            "loss_boundary": loss_boundary,
            "confidence_boundary": confidence_boundary,
            "augmentation_js_boundary": sensitivity_boundary,
            "requires_late_label_contradiction": True,
        },
        "gates": {
            "min_suspicious_precision": min_suspicious_precision,
            "min_error_enrichment": min_error_enrichment,
            "max_cut_ratio": max_cut_ratio,
            "min_after_cut_coverage": min_after_cut_coverage,
            "checks": checks,
        },
    }
    arrays = {
        "sample_index": early_index,
        "target_label": target_label,
        "early_label": early_label,
        "late_label": late_label,
        "early_selected": early_selected,
        "late_selected": late_selected,
        "early_wrong": early_wrong,
        "contradiction": contradiction,
        "high_loss": high_loss,
        "high_confidence": high_confidence,
        "low_sensitivity": low_sensitivity,
        "suspicious": suspicious,
        "late_confidence": late_confidence,
        "early_label_loss": early_label_loss,
        "augmentation_js": augmentation_js,
    }
    return report, arrays


def write_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_sample_csv(
    path: Path,
    arrays: dict[str, np.ndarray],
    data_rows: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "image_path",
        "target_label",
        "early_label",
        "late_label",
        "early_selected",
        "late_selected",
        "early_wrong",
        "contradiction",
        "high_loss",
        "high_confidence",
        "low_sensitivity",
        "suspicious",
        "late_confidence",
        "early_label_loss",
        "augmentation_js",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in range(arrays["sample_index"].size):
            sample_index = int(arrays["sample_index"][row])
            image_path = ""
            if data_rows:
                if not 0 <= sample_index < len(data_rows):
                    raise ValueError(
                        f"Sample index {sample_index} is outside the data list"
                    )
                image_path = data_rows[sample_index].rsplit(" ", 1)[0]
            writer.writerow(
                {
                    "sample_index": sample_index,
                    "image_path": image_path,
                    "target_label": int(arrays["target_label"][row]),
                    "early_label": int(arrays["early_label"][row]),
                    "late_label": int(arrays["late_label"][row]),
                    "early_selected": int(arrays["early_selected"][row]),
                    "late_selected": int(arrays["late_selected"][row]),
                    "early_wrong": int(arrays["early_wrong"][row]),
                    "contradiction": int(arrays["contradiction"][row]),
                    "high_loss": int(arrays["high_loss"][row]),
                    "high_confidence": int(arrays["high_confidence"][row]),
                    "low_sensitivity": int(arrays["low_sensitivity"][row]),
                    "suspicious": int(arrays["suspicious"][row]),
                    "late_confidence": float(arrays["late_confidence"][row]),
                    "early_label_loss": float(arrays["early_label_loss"][row]),
                    "augmentation_js": float(arrays["augmentation_js"][row]),
                }
            )


def write_class_csv(
    path: Path,
    arrays: dict[str, np.ndarray],
    class_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_count = int(
        max(arrays["target_label"].max(), arrays["early_label"].max()) + 1
    )
    fieldnames = [
        "class_index",
        "class_name",
        "early_selected",
        "early_wrong",
        "suspicious",
        "suspicious_wrong",
        "suspicious_correct",
        "suspicious_precision_percent",
        "class_cut_rate_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_index in range(class_count):
            class_mask = arrays["early_label"] == class_index
            early_selected = class_mask & arrays["early_selected"]
            suspicious = class_mask & arrays["suspicious"]
            suspicious_wrong = suspicious & arrays["early_wrong"]
            suspicious_correct = suspicious & ~arrays["early_wrong"]
            writer.writerow(
                {
                    "class_index": class_index,
                    "class_name": (
                        class_names[class_index]
                        if class_index < len(class_names)
                        else str(class_index)
                    ),
                    "early_selected": int(early_selected.sum()),
                    "early_wrong": int(
                        (early_selected & arrays["early_wrong"]).sum()
                    ),
                    "suspicious": int(suspicious.sum()),
                    "suspicious_wrong": int(suspicious_wrong.sum()),
                    "suspicious_correct": int(suspicious_correct.sum()),
                    "suspicious_precision_percent": percent(
                        int(suspicious_wrong.sum()), int(suspicious.sum())
                    ),
                    "class_cut_rate_percent": percent(
                        int(suspicious.sum()), int(early_selected.sum())
                    ),
                }
            )


def main() -> None:
    args = parse_args()
    report, arrays = analyze(
        load_snapshot(args.early),
        load_snapshot(args.late),
        loss_fraction=args.loss_fraction,
        confidence_fraction=args.confidence_fraction,
        stability_fraction=args.stability_fraction,
        min_suspicious_precision=args.min_suspicious_precision,
        min_error_enrichment=args.min_error_enrichment,
        max_cut_ratio=args.max_cut_ratio,
        min_after_cut_coverage=args.min_after_cut_coverage,
    )
    write_json(args.out_json, report)
    write_class_csv(args.out_classes, arrays, read_lines(args.class_names))
    write_sample_csv(args.out_samples, arrays, read_lines(args.data_list))

    metrics = report["metrics"]
    print(
        "Early false-consensus audit: "
        f"{report['verdict']}; "
        f"suspicious={report['counts']['suspicious']}; "
        f"precision={metrics['suspicious_precision_percent']:.2f}%; "
        f"enrichment={metrics['error_enrichment']:.2f}x; "
        f"cut={metrics['cut_ratio_percent']:.2f}%; "
        f"after_cut_coverage={metrics['after_cut_coverage_percent']:.2f}%"
    )
    print(f"Report: {args.out_json}")


if __name__ == "__main__":
    main()
