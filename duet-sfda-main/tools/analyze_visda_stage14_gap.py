#!/usr/bin/env python
"""Matched DUET-vs-Stage14 gap audit for VisDA-C."""

from __future__ import annotations

import argparse
import csv
import glob
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duet-final", required=True)
    parser.add_argument("--stage14-final", required=True)
    parser.add_argument("--duet-temporal-glob", required=True)
    parser.add_argument("--stage14-temporal-glob", required=True)
    parser.add_argument("--class-names", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--tsne-max-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2020)
    return parser.parse_args()


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as item:
        return {key: item[key] for key in item.files}


def latest_snapshot(pattern: str) -> tuple[Path, dict[str, np.ndarray]]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No temporal snapshots match: {pattern}")
    return paths[-1], load_npz(paths[-1])


def require_keys(item: dict[str, np.ndarray], keys: set[str], name: str) -> None:
    missing = sorted(keys - set(item))
    if missing:
        raise ValueError(f"{name} is missing keys: {missing}")


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (target.astype(np.int64), prediction.astype(np.int64)), 1)
    return matrix


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    support = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        support,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=support > 0,
    )


def class_accuracy(probability: np.ndarray, target: np.ndarray, classes: int) -> np.ndarray:
    prediction = probability.argmax(axis=1)
    matrix = confusion_matrix(target, prediction, classes)
    return np.diag(row_normalize(matrix)) * 100.0


def macro_micro(probability: np.ndarray, target: np.ndarray, classes: int) -> tuple[float, float]:
    accuracy = class_accuracy(probability, target, classes)
    micro = float(np.mean(probability.argmax(axis=1) == target)) * 100.0
    return float(np.mean(accuracy)), micro


def true_margin(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows = np.arange(target.size)
    true_probability = probability[rows, target]
    other = probability.copy()
    other[rows, target] = -np.inf
    return true_probability - other.max(axis=1)


def feature_geometry(
    feature: np.ndarray,
    target: np.ndarray,
    classes: int,
) -> dict[str, object]:
    feature = feature.astype(np.float32)
    norm = np.linalg.norm(feature, axis=1, keepdims=True)
    feature = feature / np.maximum(norm, 1e-12)
    prototypes = np.zeros((classes, feature.shape[1]), dtype=np.float32)
    dispersion = np.zeros(classes, dtype=np.float64)
    for class_index in range(classes):
        rows = target == class_index
        if not np.any(rows):
            raise ValueError(f"No target samples for class {class_index}")
        prototype = feature[rows].mean(axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        prototypes[class_index] = prototype
        dispersion[class_index] = float(
            np.mean(1.0 - feature[rows] @ prototype)
        )

    pairs: dict[tuple[int, int], dict[str, float]] = {}
    for left in range(classes):
        for right in range(left + 1, classes):
            distance = float(1.0 - prototypes[left] @ prototypes[right])
            within = float((dispersion[left] + dispersion[right]) / 2.0)
            pairs[(left, right)] = {
                "prototype_cosine_distance": distance,
                "mean_within_class_dispersion": within,
                "fisher_separability": distance / max(within, 1e-12),
            }
    return {
        "dispersion": dispersion,
        "prototypes": prototypes,
        "pairs": pairs,
    }


def flow_metrics(snapshot: dict[str, np.ndarray], classes: int) -> dict[str, np.ndarray]:
    require_keys(
        snapshot,
        {"target_label", "mix_label", "label_mask", "source_label", "clip_label"},
        "temporal snapshot",
    )
    target = snapshot["target_label"].astype(np.int64)
    mix = snapshot["mix_label"].astype(np.int64)
    selected = snapshot["label_mask"].astype(bool)
    selected_count = np.zeros(classes, dtype=np.int64)
    selected_precision = np.full(classes, np.nan, dtype=np.float64)
    true_coverage = np.zeros(classes, dtype=np.float64)
    conflict_rate = np.zeros(classes, dtype=np.float64)
    for class_index in range(classes):
        predicted_rows = selected & (mix == class_index)
        selected_count[class_index] = int(np.sum(predicted_rows))
        if np.any(predicted_rows):
            selected_precision[class_index] = float(
                np.mean(target[predicted_rows] == class_index)
            ) * 100.0
        true_rows = target == class_index
        true_coverage[class_index] = float(np.mean(selected[true_rows])) * 100.0
        conflict_rate[class_index] = float(
            np.mean(
                snapshot["source_label"][true_rows]
                != snapshot["clip_label"][true_rows]
            )
        ) * 100.0
    distribution = selected_count.astype(np.float64)
    distribution /= max(float(distribution.sum()), 1.0)
    return {
        "selected_count": selected_count,
        "selected_precision": selected_precision,
        "true_coverage": true_coverage,
        "conflict_rate": conflict_rate,
        "selected_distribution": distribution,
    }


def _finite_round(value: float, digits: int = 4) -> float | None:
    return round(float(value), digits) if np.isfinite(value) else None


def analyze(
    duet_final: dict[str, np.ndarray],
    stage14_final: dict[str, np.ndarray],
    duet_temporal: dict[str, np.ndarray],
    stage14_temporal: dict[str, np.ndarray],
    class_names: list[str],
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    final_keys = {"target_label", "task_feature", "base_task_prob", "task_prob"}
    require_keys(duet_final, final_keys, "DUET final snapshot")
    require_keys(stage14_final, final_keys, "Stage14 final snapshot")
    classes = len(class_names)
    duet_target = duet_final["target_label"].astype(np.int64)
    stage14_target = stage14_final["target_label"].astype(np.int64)
    if not np.array_equal(duet_target, stage14_target):
        raise ValueError("DUET and Stage14 final snapshots are not sample-aligned")
    target = duet_target
    if int(target.max()) + 1 != classes:
        raise ValueError("Class-name count does not cover final target labels")

    duet_prob = duet_final["task_prob"].astype(np.float64)
    stage14_prob = stage14_final["task_prob"].astype(np.float64)
    stage14_base_prob = stage14_final["base_task_prob"].astype(np.float64)
    duet_prediction = duet_prob.argmax(axis=1)
    stage14_prediction = stage14_prob.argmax(axis=1)
    duet_matrix = confusion_matrix(target, duet_prediction, classes)
    stage14_matrix = confusion_matrix(target, stage14_prediction, classes)
    duet_normalized = row_normalize(duet_matrix)
    stage14_normalized = row_normalize(stage14_matrix)

    duet_accuracy = np.diag(duet_normalized) * 100.0
    stage14_accuracy = np.diag(stage14_normalized) * 100.0
    stage14_base_accuracy = class_accuracy(stage14_base_prob, target, classes)
    delta_accuracy = stage14_accuracy - duet_accuracy
    support = duet_matrix.sum(axis=1)
    if np.unique(support).size < 2 or np.unique(delta_accuracy).size < 2:
        rank_statistic = float("nan")
        rank_p_value = float("nan")
    else:
        rank = spearmanr(support, delta_accuracy)
        rank_statistic = float(rank.statistic)
        rank_p_value = float(rank.pvalue)

    duet_geometry = feature_geometry(
        duet_final["task_feature"], target, classes
    )
    stage14_geometry = feature_geometry(
        stage14_final["task_feature"], target, classes
    )
    duet_flow = flow_metrics(duet_temporal, classes)
    stage14_flow = flow_metrics(stage14_temporal, classes)
    flow_tv = 0.5 * float(
        np.sum(
            np.abs(
                duet_flow["selected_distribution"]
                - stage14_flow["selected_distribution"]
            )
        )
    )

    duet_margin = true_margin(duet_prob, target)
    stage14_margin = true_margin(stage14_prob, target)
    class_rows: list[dict[str, object]] = []
    for class_index, class_name in enumerate(class_names):
        rows = target == class_index
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "target_samples": int(support[class_index]),
                "duet_accuracy": round(float(duet_accuracy[class_index]), 4),
                "stage14_accuracy": round(float(stage14_accuracy[class_index]), 4),
                "stage14_minus_duet_pp": round(
                    float(delta_accuracy[class_index]), 4
                ),
                "stage14_base_head_accuracy": round(
                    float(stage14_base_accuracy[class_index]), 4
                ),
                "stage14_effective_minus_base_pp": round(
                    float(
                        stage14_accuracy[class_index]
                        - stage14_base_accuracy[class_index]
                    ),
                    4,
                ),
                "duet_true_margin": round(float(np.mean(duet_margin[rows])), 6),
                "stage14_true_margin": round(
                    float(np.mean(stage14_margin[rows])), 6
                ),
                "duet_within_dispersion": round(
                    float(duet_geometry["dispersion"][class_index]), 6
                ),
                "stage14_within_dispersion": round(
                    float(stage14_geometry["dispersion"][class_index]), 6
                ),
                "duet_selected_count": int(
                    duet_flow["selected_count"][class_index]
                ),
                "stage14_selected_count": int(
                    stage14_flow["selected_count"][class_index]
                ),
                "duet_selected_precision": _finite_round(
                    duet_flow["selected_precision"][class_index]
                ),
                "stage14_selected_precision": _finite_round(
                    stage14_flow["selected_precision"][class_index]
                ),
                "duet_true_class_coverage": round(
                    float(duet_flow["true_coverage"][class_index]), 4
                ),
                "stage14_true_class_coverage": round(
                    float(stage14_flow["true_coverage"][class_index]), 4
                ),
                "duet_source_clip_conflict_rate": round(
                    float(duet_flow["conflict_rate"][class_index]), 4
                ),
                "stage14_source_clip_conflict_rate": round(
                    float(stage14_flow["conflict_rate"][class_index]), 4
                ),
            }
        )

    pair_rows: list[dict[str, object]] = []
    positive_excess = 0
    for left in range(classes):
        for right in range(left + 1, classes):
            duet_errors = int(duet_matrix[left, right] + duet_matrix[right, left])
            stage14_errors = int(
                stage14_matrix[left, right] + stage14_matrix[right, left]
            )
            pair_support = int(support[left] + support[right])
            excess = stage14_errors - duet_errors
            positive_excess += max(excess, 0)
            duet_pair_geometry = duet_geometry["pairs"][(left, right)]
            stage14_pair_geometry = stage14_geometry["pairs"][(left, right)]
            duet_fisher = duet_pair_geometry["fisher_separability"]
            stage14_fisher = stage14_pair_geometry["fisher_separability"]
            relative_fisher = (stage14_fisher - duet_fisher) / max(
                abs(duet_fisher), 1e-12
            )
            pair_rows.append(
                {
                    "left_index": left,
                    "left_class": class_names[left],
                    "right_index": right,
                    "right_class": class_names[right],
                    "pair_support": pair_support,
                    "duet_mutual_errors": duet_errors,
                    "stage14_mutual_errors": stage14_errors,
                    "stage14_excess_errors": excess,
                    "duet_pair_confusion_rate": round(
                        100.0 * duet_errors / pair_support, 4
                    ),
                    "stage14_pair_confusion_rate": round(
                        100.0 * stage14_errors / pair_support, 4
                    ),
                    "confusion_delta_pp": round(
                        100.0 * excess / pair_support, 4
                    ),
                    "duet_prototype_distance": round(
                        duet_pair_geometry["prototype_cosine_distance"], 6
                    ),
                    "stage14_prototype_distance": round(
                        stage14_pair_geometry["prototype_cosine_distance"], 6
                    ),
                    "duet_fisher_separability": round(duet_fisher, 6),
                    "stage14_fisher_separability": round(stage14_fisher, 6),
                    "fisher_relative_change": round(relative_fisher, 6),
                }
            )
    pair_rows.sort(
        key=lambda row: (
            row["stage14_excess_errors"],
            row["confusion_delta_pp"],
        ),
        reverse=True,
    )

    directional_rows: list[dict[str, object]] = []
    for true_class in range(classes):
        for predicted_class in range(classes):
            if true_class == predicted_class:
                continue
            directional_rows.append(
                {
                    "true_index": true_class,
                    "true_class": class_names[true_class],
                    "predicted_index": predicted_class,
                    "predicted_class": class_names[predicted_class],
                    "duet_errors": int(duet_matrix[true_class, predicted_class]),
                    "stage14_errors": int(
                        stage14_matrix[true_class, predicted_class]
                    ),
                    "duet_row_rate": round(
                        float(duet_normalized[true_class, predicted_class])
                        * 100.0,
                        4,
                    ),
                    "stage14_row_rate": round(
                        float(stage14_normalized[true_class, predicted_class])
                        * 100.0,
                        4,
                    ),
                    "row_rate_delta_pp": round(
                        float(
                            stage14_normalized[true_class, predicted_class]
                            - duet_normalized[true_class, predicted_class]
                        )
                        * 100.0,
                        4,
                    ),
                }
            )
    directional_rows.sort(
        key=lambda row: row["row_rate_delta_pp"], reverse=True
    )

    fisher_changes = np.array(
        [row["fisher_relative_change"] for row in pair_rows], dtype=np.float64
    )
    top_excess = [row for row in pair_rows if row["stage14_excess_errors"] > 0]
    top1_share = (
        max(row["stage14_excess_errors"] for row in top_excess)
        / positive_excess
        if positive_excess > 0
        else 0.0
    )
    compressed_excess_pairs = [
        f"{row['left_class']}<->{row['right_class']}"
        for row in top_excess[:10]
        if row["fisher_relative_change"] <= -0.05
    ]
    harmed_classes = [
        class_names[index]
        for index, delta in enumerate(delta_accuracy)
        if delta <= -0.5
    ]

    duet_macro, duet_micro = macro_micro(duet_prob, target, classes)
    stage14_macro, stage14_micro = macro_micro(
        stage14_prob, target, classes
    )
    stage14_base_macro, stage14_base_micro = macro_micro(
        stage14_base_prob, target, classes
    )
    head_delta = stage14_macro - stage14_base_macro
    median_fisher_change = float(np.median(fisher_changes))
    distributed_confusion = len(harmed_classes) > 2 or top1_share < 0.5
    imbalance_associated = (
        np.isfinite(rank_statistic)
        and abs(rank_statistic) >= 0.5
        and rank_p_value <= 0.1
    )
    localized_geometry = bool(compressed_excess_pairs)
    global_geometry = median_fisher_change <= -0.05
    head_harm = head_delta <= -0.2
    flow_shift = flow_tv >= 0.05

    stage14_failed = stage14_macro < duet_macro
    if not stage14_failed:
        route = "replicate_before_method_change"
        next_step = (
            "Stage14 did not fail on this matched run. Replicate the result on "
            "the full set and additional seeds before changing the method."
        )
    elif global_geometry:
        route = "isolate_global_alignment_components"
        next_step = (
            "Keep the Office-Home Stage14 positive control frozen. On VisDA, "
            "run component ablations to find which loss compresses geometry."
        )
    elif head_harm:
        route = "remove_or_source_anchor_target_head"
        next_step = (
            "Keep the validated Stage14 pseudo-label path and Office-Home "
            "setting, but remove or source-anchor the VisDA target head."
        )
    elif flow_shift:
        route = "balance_pseudo_label_flow"
        next_step = (
            "On VisDA, test class-balanced supervised pseudo-label exposure "
            "before any new boundary module; keep the source head unchanged."
        )
    elif localized_geometry:
        route = "validate_dynamic_hard_confusion_set"
        next_step = (
            "Replicate the compressed pairs across seeds; only then test a "
            "data-discovered hard-confusion residual, never a fixed car-truck pair."
        )
    else:
        route = "component_ablation_before_new_module"
        next_step = (
            "The mechanism is not yet isolated. Do not add a module; ablate "
            "Stage14 memory, graph correction, calibration, and target head."
        )

    summary: dict[str, object] = {
        "decision": (
            "stage14_visda_gap_diagnosed"
            if stage14_failed
            else "stage14_no_visda_gap_on_matched_run"
        ),
        "warning": (
            "Target labels are used only for post-hoc mechanism diagnosis and "
            "must not become an unsupervised training oracle."
        ),
        "comparison": {
            "duet_macro_accuracy": round(duet_macro, 4),
            "stage14_macro_accuracy": round(stage14_macro, 4),
            "stage14_minus_duet_pp": round(stage14_macro - duet_macro, 4),
            "duet_micro_accuracy": round(duet_micro, 4),
            "stage14_micro_accuracy": round(stage14_micro, 4),
            "stage14_base_head_macro_accuracy": round(
                stage14_base_macro, 4
            ),
            "stage14_effective_minus_base_pp": round(head_delta, 4),
            "stage14_base_head_micro_accuracy": round(
                stage14_base_micro, 4
            ),
        },
        "mentor_checks": {
            "confusion_scope": {
                "result": (
                    "distributed_beyond_one_fixed_pair"
                    if distributed_confusion
                    else "concentrated_in_one_pair"
                ),
                "harmed_classes_at_least_0_5pp": harmed_classes,
                "top1_positive_excess_error_share": round(top1_share, 6),
                "top_excess_pairs": [
                    f"{row['left_class']}<->{row['right_class']}"
                    for row in top_excess[:5]
                ],
            },
            "class_imbalance": {
                "result": (
                    "associated_but_not_causal"
                    if imbalance_associated
                    else "no_clear_rank_association"
                ),
                "max_to_min_class_count_ratio": round(
                    float(support.max() / support.min()), 6
                ),
                "spearman_count_vs_accuracy_delta": _finite_round(
                    rank_statistic, 6
                ),
                "p_value": _finite_round(rank_p_value, 6),
                "note": (
                    "Macro accuracy already balances evaluation classes; "
                    "imbalance can still act through pseudo-label exposure."
                ),
            },
            "feature_space": {
                "result": (
                    "global_geometry_compression"
                    if global_geometry
                    else (
                        "localized_pair_compression"
                        if localized_geometry
                        else "no_supported_compression"
                    )
                ),
                "median_pair_fisher_relative_change": round(
                    median_fisher_change, 6
                ),
                "compressed_top_excess_pairs": compressed_excess_pairs,
            },
            "pseudo_label_flow": {
                "result": (
                    "material_distribution_shift"
                    if flow_shift
                    else "no_material_distribution_shift"
                ),
                "selected_class_distribution_total_variation": round(
                    flow_tv, 6
                ),
            },
            "classifier_head": {
                "result": (
                    "effective_head_harms_macro_accuracy"
                    if head_harm
                    else "no_material_head_harm"
                ),
                "effective_minus_source_head_pp": round(head_delta, 6),
            },
        },
        "recommended_route": route,
        "next_experiment": next_step,
        "criteria": {
            "harmed_class_threshold_pp": -0.5,
            "geometry_compression_relative_threshold": -0.05,
            "global_geometry_median_threshold": -0.05,
            "head_harm_threshold_pp": -0.2,
            "pseudo_label_distribution_tv_threshold": 0.05,
            "imbalance_abs_spearman_threshold": 0.5,
            "imbalance_p_value_threshold": 0.1,
        },
    }
    tables = {
        "per_class": class_rows,
        "pair_confusion_geometry": pair_rows,
        "directional_confusions": directional_rows,
        "duet_confusion": matrix_rows(
            duet_matrix, duet_normalized, class_names
        ),
        "stage14_confusion": matrix_rows(
            stage14_matrix, stage14_normalized, class_names
        ),
    }
    return summary, tables


def matrix_rows(
    matrix: np.ndarray,
    normalized: np.ndarray,
    class_names: list[str],
) -> list[dict[str, object]]:
    rows = []
    for true_index, true_name in enumerate(class_names):
        for predicted_index, predicted_name in enumerate(class_names):
            rows.append(
                {
                    "true_index": true_index,
                    "true_class": true_name,
                    "predicted_index": predicted_index,
                    "predicted_class": predicted_name,
                    "count": int(matrix[true_index, predicted_index]),
                    "row_percent": round(
                        float(normalized[true_index, predicted_index]) * 100.0,
                        6,
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_tsne(
    duet_final: dict[str, np.ndarray],
    stage14_final: dict[str, np.ndarray],
    class_names: list[str],
    out_dir: Path,
    max_per_class: int,
    seed: int,
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    target = duet_final["target_label"].astype(np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    for class_index in range(len(class_names)):
        indices = np.flatnonzero(target == class_index)
        if indices.size > max_per_class:
            indices = rng.choice(indices, size=max_per_class, replace=False)
        selected.extend(indices.tolist())
    selected = np.array(sorted(selected), dtype=np.int64)
    duet_feature = duet_final["task_feature"][selected].astype(np.float32)
    stage14_feature = stage14_final["task_feature"][selected].astype(np.float32)
    combined = np.concatenate([duet_feature, stage14_feature], axis=0)
    combined /= np.maximum(
        np.linalg.norm(combined, axis=1, keepdims=True), 1e-12
    )
    components = min(50, combined.shape[1], combined.shape[0] - 1)
    reduced = PCA(n_components=components, random_state=seed).fit_transform(
        combined
    )
    kwargs = {
        "n_components": 2,
        "perplexity": min(30.0, max(5.0, (combined.shape[0] - 1) / 3.0)),
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000
    embedding = TSNE(**kwargs).fit_transform(reduced)
    sample_count = selected.size
    embedding_rows = []
    for method_index, method in enumerate(("DUET", "Stage14")):
        offset = method_index * sample_count
        for local_index, sample_index in enumerate(selected):
            class_index = int(target[sample_index])
            embedding_rows.append(
                {
                    "method": method,
                    "sample_index": int(sample_index),
                    "class_index": class_index,
                    "class": class_names[class_index],
                    "x": float(embedding[offset + local_index, 0]),
                    "y": float(embedding[offset + local_index, 1]),
                }
            )
    write_csv(out_dir / "stage14_visda_feature_tsne.csv", embedding_rows)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    colors = plt.cm.get_cmap("tab20", len(class_names))
    for method_index, (method, axis) in enumerate(
        zip(("DUET", "Stage14"), axes)
    ):
        offset = method_index * sample_count
        for class_index, class_name in enumerate(class_names):
            mask = target[selected] == class_index
            axis.scatter(
                embedding[offset : offset + sample_count][mask, 0],
                embedding[offset : offset + sample_count][mask, 1],
                s=5,
                alpha=0.55,
                color=colors(class_index),
                label=class_name,
            )
        axis.set_title(method)
        axis.set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        frameon=False,
    )
    figure.suptitle(
        "Matched VisDA features (visual evidence only; see Fisher metrics)"
    )
    figure.tight_layout(rect=(0, 0, 0.88, 0.95))
    figure.savefig(out_dir / "stage14_visda_feature_tsne.png", dpi=180)
    plt.close(figure)
    return {
        "status": "written",
        "samples_per_method": int(sample_count),
        "csv": "stage14_visda_feature_tsne.csv",
        "png": "stage14_visda_feature_tsne.png",
    }


def main() -> None:
    args = parse_args()
    class_names = [
        token.replace("_", " ")
        for token in Path(args.class_names).read_text().split()
    ]
    duet_final = load_npz(args.duet_final)
    stage14_final = load_npz(args.stage14_final)
    duet_temporal_path, duet_temporal = latest_snapshot(
        args.duet_temporal_glob
    )
    stage14_temporal_path, stage14_temporal = latest_snapshot(
        args.stage14_temporal_glob
    )
    summary, tables = analyze(
        duet_final,
        stage14_final,
        duet_temporal,
        stage14_temporal,
        class_names,
    )
    summary["inputs"] = {
        "duet_final": str(Path(args.duet_final)),
        "stage14_final": str(Path(args.stage14_final)),
        "duet_temporal": str(duet_temporal_path),
        "stage14_temporal": str(stage14_temporal_path),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(out_dir / f"{name}.csv", rows)

    if args.skip_tsne:
        summary["tsne"] = {"status": "skipped"}
    else:
        try:
            summary["tsne"] = create_tsne(
                duet_final,
                stage14_final,
                class_names,
                out_dir,
                args.tsne_max_per_class,
                args.seed,
            )
        except Exception as error:
            summary["tsne"] = {
                "status": "failed_without_blocking_numeric_audit",
                "error": f"{type(error).__name__}: {error}",
            }

    summary_path = out_dir / "stage14_visda_gap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
