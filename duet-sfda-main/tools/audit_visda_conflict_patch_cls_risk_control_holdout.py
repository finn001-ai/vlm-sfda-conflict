#!/usr/bin/env python3
"""Disjoint held-out validation of the frozen patch-to-CLS risk control.

The 25% proxy paths used to design the rule are removed before any CLIP image
forward.  Task/CLIP candidates come from the previously locked full-target
feature-gravity NPZ.  The only GPU work is one deterministic, frozen CLIP
center-crop forward over held-out conflicts.  The upper-median and 1% mass-cap
rule is unchanged.  Target labels are parsed only after the new signal lock.

There is no task/source-model load, backward pass, optimizer, parameter update,
proxy training, or full training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import clip  # noqa: E402
from src.utils.conflict_boundary import paired_accuracy_bootstrap_ci  # noqa: E402
from src.utils.patch_cls_contribution_audit import (  # noqa: E402
    candidate_peak_response,
    encode_last_block_patch_cls_contributions,
    unanimous_head_partition_rescue,
)
from src.utils.patch_cls_risk_control_audit import (  # noqa: E402
    evaluate_patch_cls_holdout_gate,
    select_upper_median_mass_capped_rescues,
)


EXPECTED_FULL_SAMPLES = 55_388
EXPECTED_PROXY_SAMPLES = 13_847
EXPECTED_HOLDOUT_SAMPLES = EXPECTED_FULL_SAMPLES - EXPECTED_PROXY_SAMPLES
EXPECTED_FULL_CONFLICTS = 28_223
EXPECTED_CLASSES = 12
MAX_CLASS_MASS_SHIFT_FRACTION = 0.01
CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
)
FEATURE_BASE = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_feature_gravity_audit_seed2020/feature_gravity_audit"
)
EXPLORATORY_BASE = Path(
    "output/uda/VISDA-C/TV/"
    "duet_support_conditioned_clip_cycle2_memory_audit_seed2020/"
    "patch_cls_contribution_audit/risk_control_audit"
)
STEM = "visda_conflict_patch_cls_risk_control_holdout"


class _OpaquePathDataset(Dataset):
    """Load image paths while keeping target-list suffixes opaque."""

    def __init__(self, paths: list[str], indices: np.ndarray, transform: Any):
        if len(paths) != indices.size:
            raise ValueError("paths and indices must align")
        self.paths = paths
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        with open(self.paths[position], "rb") as handle:
            with Image.open(handle) as image:
                tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.indices[position])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-signal",
        type=Path,
        default=FEATURE_BASE / "visda_conflict_feature_gravity_signals.npz",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=FEATURE_BASE / "visda_conflict_feature_gravity_signal_lock.json",
    )
    parser.add_argument(
        "--exploratory-summary",
        type=Path,
        default=EXPLORATORY_BASE / "visda_conflict_patch_cls_risk_control_summary.json",
    )
    parser.add_argument(
        "--full-target-list",
        type=Path,
        default=Path("data/VISDA-C/validation_list.txt"),
    )
    parser.add_argument(
        "--proxy-list",
        type=Path,
        default=Path("data/VISDA-C/validation_proxy25_seed2020_list.txt"),
    )
    parser.add_argument(
        "--class-names", type=Path, default=Path("data/VISDA-C/classname.txt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=FEATURE_BASE / "patch_cls_holdout_audit"
    )
    parser.add_argument("--arch", default="ViT-B/32")
    parser.add_argument("--ctx-init", default="a_photo_of_a")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2_020)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_opaque_paths(path: Path, expected: int) -> list[str]:
    paths: list[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            image_path, _opaque_suffix = stripped.rsplit(maxsplit=1)
        except ValueError as error:
            raise ValueError(
                f"Malformed target row {line_number}: expected path and opaque suffix"
            ) from error
        paths.append(image_path)
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} paths in {path}, found {len(paths)}")
    if len(set(paths)) != expected:
        raise ValueError(f"Target paths are not unique in {path}")
    return paths


def _parse_labels_after_lock(path: Path) -> np.ndarray:
    labels: list[int] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            _image_path, label_text = stripped.rsplit(maxsplit=1)
            labels.append(int(label_text))
        except ValueError as error:
            raise ValueError(
                f"Malformed oracle row {line_number}: {stripped}"
            ) from error
    result = np.asarray(labels, dtype=np.int64)
    if result.shape != (EXPECTED_FULL_SAMPLES,):
        raise ValueError("Full-target oracle label count is incorrect")
    if np.any(result < 0) or np.any(result >= EXPECTED_CLASSES):
        raise ValueError("Oracle label outside class range")
    return result


def _load_class_names(path: Path) -> list[str]:
    names = [line.strip().replace("_", " ") for line in path.read_text().splitlines()]
    names = [name for name in names if name]
    if tuple(names) != CLASS_NAMES:
        raise ValueError(f"Unexpected VisDA class contract: {names}")
    return names


def _comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    candidate_correct = candidate == labels
    baseline_correct = baseline == labels
    interval = paired_accuracy_bootstrap_ci(
        candidate_correct, baseline_correct, repeats=repeats, seed=seed
    )
    candidate_accuracy = float(candidate_correct.mean() * 100.0)
    baseline_accuracy = float(baseline_correct.mean() * 100.0)
    return {
        "samples": int(labels.size),
        "candidate_accuracy_pct": candidate_accuracy,
        "baseline_accuracy_pct": baseline_accuracy,
        "gain_pp": candidate_accuracy - baseline_accuracy,
        "net_corrections": int(candidate_correct.sum() - baseline_correct.sum()),
        "paired_bootstrap_95_ci_pp": list(interval),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["label_free_metrics"]
    oracle = summary["oracle_diagnostic"]
    clip = oracle["comparisons"]["fixed_clip"]
    lines = [
        "# VisDA Patch-to-CLS Risk-Control Held-Out Audit",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Held-out paths | "
            f"`{metrics['heldout_samples']}`; proxy overlap `0` "
            "| Label-free path split |"
        ),
        (
            "| Selected conflict coverage | "
            f"`{metrics['selected_coverage_pct']:.6f}%` "
            "| Label-free signal lock |"
        ),
        (
            "| Maximum held-out class-mass shift | "
            f"`{metrics['max_class_mass_shift_pp']:.6f}` pp "
            "| Label-free signal lock |"
        ),
        (
            "| Conflict gain vs fixed CLIP | "
            f"`{clip['gain_pp']:.6f}` pp; CI "
            f"`{clip['paired_bootstrap_95_ci_pp']}` "
            "| Oracle diagnostic after lock |"
        ),
        (
            "| Held-out macro gain vs fixed CLIP | "
            f"`{oracle['heldout_macro_gain_pp']:.6f}` pp "
            "| Oracle diagnostic after lock |"
        ),
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The proxy25 paths used to design the rule were excluded before the",
        "frozen CLIP forward. The upper-median and 1% pseudo-class mass-cap",
        "rule is unchanged. Target labels were parsed only after the NPZ and",
        "manifest were written and SHA256-locked.",
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend(
        [
            "",
            "PASS authorizes one exact parameter-impact audit only. It never",
            "authorizes proxy or full VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    if args.arch != "ViT-B/32":
        raise ValueError("Held-out audit is locked to DUET CLIP ViT-B/32")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    for required in (
        args.source_signal,
        args.source_lock,
        args.exploratory_summary,
        args.full_target_list,
        args.proxy_list,
        args.class_names,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen CLIP held-out forward")

    source_lock = json.loads(args.source_lock.read_text())
    with np.load(args.source_signal, allow_pickle=False) as source:
        source_arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    full_query_index = np.asarray(source_arrays["index"], dtype=np.int64)
    task_all = np.asarray(source_arrays["task_pred"], dtype=np.int64)
    clip_all = np.asarray(source_arrays["clip_pred"], dtype=np.int64)
    task_conf_all = np.asarray(source_arrays["task_conf"], dtype=np.float64)
    clip_conf_all = np.asarray(source_arrays["clip_conf"], dtype=np.float64)
    full_paths = _read_opaque_paths(args.full_target_list, EXPECTED_FULL_SAMPLES)
    proxy_paths = _read_opaque_paths(args.proxy_list, EXPECTED_PROXY_SAMPLES)
    proxy_path_set = set(proxy_paths)
    full_path_set = set(full_paths)
    holdout_sample_mask = np.asarray(
        [path not in proxy_path_set for path in full_paths], dtype=bool
    )
    heldout_query_mask = holdout_sample_mask[full_query_index]
    heldout_source_rows = np.flatnonzero(heldout_query_mask)
    query_index = full_query_index[heldout_query_mask]
    task_candidate = task_all[heldout_query_mask]
    clip_candidate = clip_all[heldout_query_mask]
    task_conf = task_conf_all[heldout_query_mask]
    clip_conf = clip_conf_all[heldout_query_mask]
    query_paths = [full_paths[index] for index in query_index]

    pre_forward_checks = {
        "source_lock_is_label_free": (
            source_lock.get("phase") == "LABEL_FREE_SIGNAL_LOCK"
            and source_lock.get("contains_target_labels") is False
        ),
        "source_signal_hash_matches_lock": (
            _sha256(args.source_signal)
            == source_lock.get("signal_npz", {}).get("sha256")
        ),
        "source_target_list_hash_matches": (
            _sha256(args.full_target_list) == source_lock.get("target_list_sha256")
        ),
        "expected_full_conflict_rows": (
            full_query_index.shape == (EXPECTED_FULL_CONFLICTS,)
        ),
        "source_arrays_align": (
            task_all.shape
            == clip_all.shape
            == task_conf_all.shape
            == clip_conf_all.shape
            == full_query_index.shape
        ),
        "full_query_sorted_unique_in_range": (
            np.array_equal(full_query_index, np.unique(full_query_index))
            and np.all(full_query_index >= 0)
            and np.all(full_query_index < EXPECTED_FULL_SAMPLES)
        ),
        "source_rows_are_conflicts": bool(np.all(task_all != clip_all)),
        "source_values_finite": bool(
            np.isfinite(task_conf_all).all() and np.isfinite(clip_conf_all).all()
        ),
        "proxy_paths_are_full_target_subset": proxy_path_set.issubset(full_path_set),
        "expected_disjoint_holdout_size": (
            int(holdout_sample_mask.sum()) == EXPECTED_HOLDOUT_SAMPLES
        ),
        "heldout_query_nonempty": query_index.size > 0,
        "heldout_query_disjoint_from_proxy": all(
            path not in proxy_path_set for path in query_paths
        ),
        "every_query_image_exists": all(Path(path).is_file() for path in query_paths),
    }
    pre_forward_checks = {
        name: bool(value) for name, value in pre_forward_checks.items()
    }
    failed = [name for name, passed in pre_forward_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Held-out patch pre-forward contract failed: {failed}")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    clip_model, _embed_dim, preprocess = clip.load(args.arch, device=device, jit=False)
    clip_model.float().eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    class_names = _load_class_names(args.class_names)
    prompt_prefix = args.ctx_init.replace("_", " ")
    prompts = [f"{prompt_prefix} {name}." for name in class_names]
    tokens = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)
    with torch.no_grad():
        text_feature = F.normalize(clip_model.encode_text(tokens).float(), dim=1)

    loader = DataLoader(
        _OpaquePathDataset(query_paths, heldout_source_rows, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "source_row",
            "full_task_peak",
            "full_clip_peak",
            "even_task_peak",
            "even_clip_peak",
            "odd_task_peak",
            "odd_clip_peak",
        )
    }
    partition_errors: list[float] = []
    with torch.no_grad():
        for images, source_row in loader:
            images = images.to(device, non_blocking=True)
            source_row_np = source_row.numpy().astype(np.int64, copy=False)
            task_tensor = torch.from_numpy(task_all[source_row_np]).to(device)
            clip_tensor = torch.from_numpy(clip_all[source_row_np]).to(device)
            contribution = encode_last_block_patch_cls_contributions(clip_model, images)
            responses = {
                partition: candidate_peak_response(
                    contribution[partition], text_feature, task_tensor, clip_tensor
                )
                for partition in ("all", "even", "odd")
            }
            collected["source_row"].append(source_row_np)
            for partition, prefix in (
                ("all", "full"),
                ("even", "even"),
                ("odd", "odd"),
            ):
                collected[f"{prefix}_task_peak"].append(
                    responses[partition]["task_peak"].cpu().numpy()
                )
                collected[f"{prefix}_clip_peak"].append(
                    responses[partition]["clip_peak"].cpu().numpy()
                )
            partition_errors.append(
                float(contribution["head_partition_max_abs_error"].item())
            )
    arrays = {name: np.concatenate(parts) for name, parts in collected.items()}
    if not np.array_equal(arrays["source_row"], heldout_source_rows):
        raise RuntimeError("CLIP loader changed the held-out conflict order")
    unanimous = unanimous_head_partition_rescue(
        task_candidate,
        clip_candidate,
        arrays["full_task_peak"],
        arrays["full_clip_peak"],
        arrays["even_task_peak"],
        arrays["even_clip_peak"],
        arrays["odd_task_peak"],
        arrays["odd_clip_peak"],
    )
    full_margin = arrays["full_task_peak"] - arrays["full_clip_peak"]
    result = select_upper_median_mass_capped_rescues(
        task_candidate,
        clip_candidate,
        full_margin,
        unanimous["rescue_task"],
        full_sample_count=EXPECTED_HOLDOUT_SAMPLES,
        max_class_mass_shift_fraction=MAX_CLASS_MASS_SHIFT_FRACTION,
        class_count=EXPECTED_CLASSES,
    )
    confidence_prediction = np.where(
        task_conf >= clip_conf, task_candidate, clip_candidate
    )
    max_partition_error = max(partition_errors, default=float("inf"))
    class_mass_shift_pp = result["class_mass_shift_fraction"] * 100.0
    input_checks = {
        **pre_forward_checks,
        "head_partition_identity_at_most_1e_5": max_partition_error <= 1e-5,
        "label_free_arrays_finite": all(
            np.isfinite(value).all() for value in arrays.values()
        ),
        "candidate_only_task_or_clip": bool(
            np.all(
                (result["prediction"] == task_candidate)
                | (result["prediction"] == clip_candidate)
            )
        ),
        "clip_model_frozen": not any(
            parameter.requires_grad for parameter in clip_model.parameters()
        ),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Held-out patch input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    signal_path = args.output_dir / f"{STEM}_label_free.npz"
    lock_path = args.output_dir / f"{STEM}_signal_lock.json"
    oracle_path = args.output_dir / f"{STEM}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{STEM}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{STEM}_summary.json"
    markdown_path = args.output_dir / f"{STEM}_summary.md"
    np.savez_compressed(
        signal_path,
        query_index=query_index,
        source_signal_row=heldout_source_rows,
        task_candidate=task_candidate,
        clip_candidate=clip_candidate,
        task_confidence=task_conf.astype(np.float32),
        clip_confidence=clip_conf.astype(np.float32),
        full_task_margin=full_margin.astype(np.float32),
        stable_rescue=unanimous["rescue_task"],
        upper_median=result["upper_median"],
        selected=result["selected"],
        rejected_by_mass_cap=result["rejected_by_mass_cap"],
        candidate_prediction=result["prediction"],
        fixed_task_prediction=task_candidate,
        fixed_clip_prediction=clip_candidate,
        confidence_prediction=confidence_prediction,
        class_count_shift=result["class_count_shift"],
        median_threshold=np.asarray(result["threshold"], dtype=np.float64),
    )
    label_free_metrics = {
        "full_samples": EXPECTED_FULL_SAMPLES,
        "proxy_samples_excluded": EXPECTED_PROXY_SAMPLES,
        "heldout_samples": EXPECTED_HOLDOUT_SAMPLES,
        "proxy_overlap": 0,
        "full_conflict_rows": int(full_query_index.size),
        "proxy_conflict_rows_excluded": int((~heldout_query_mask).sum()),
        "heldout_conflict_rows": int(query_index.size),
        "stable_rescues": int(unanimous["rescue_task"].sum()),
        "upper_median_candidates": int(result["upper_median"].sum()),
        "selected_samples": int(result["selected"].sum()),
        "selected_coverage_pct": float(result["selected"].mean() * 100.0),
        "rejected_by_mass_cap": int(result["rejected_by_mass_cap"].sum()),
        "median_full_head_margin": float(result["threshold"]),
        "class_count_cap": int(result["count_cap"]),
        "class_count_shift": result["class_count_shift"].tolist(),
        "class_mass_shift_pp": {
            name: float(class_mass_shift_pp[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "max_class_mass_shift_pp": float(np.abs(class_mass_shift_pp).max()),
        "head_partition_max_abs_error": max_partition_error,
    }
    lock = {
        "phase": "LABEL_FREE_VISDA_PATCH_CLS_RISK_CONTROL_HOLDOUT_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "target_list_suffixes_opaque_before_lock": True,
        "exploratory_oracle_summary_not_read_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "confirmatory_status": "disjoint_path_holdout_confirmation",
        "candidate_contract": {
            "design_subset": "validation_proxy25_seed2020_paths",
            "evaluation_subset": "full_target_complement_of_design_paths",
            "query": "locked_full_target_task_clip_top1_conflicts_in_evaluation_subset",
            "default_prediction": "fixed_clip_top1",
            "alternative_prediction": "fixed_task_top1",
            "image_view": "deterministic_official_clip_center_crop",
            "stable_rescue": "task_wins_full_even_and_odd_head_partitions",
            "confidence_control": "full_head_margin_at_or_above_heldout_stable_set_median",
            "class_mass_control": "descending_margin_greedy_acceptance_with_1pct_heldout_cap",
            "searched_fraction": False,
            "numerical_margin_threshold": False,
            "class_specific_route": False,
            "target_label_rule": False,
            "training_change_in_this_audit": False,
        },
        "predeclared_gate": {
            "selected_coverage_pct": [2.0, 10.0],
            "min_paired_adjudication_precision_pct": 60.0,
            "min_gain_vs_fixed_clip_pp": 1.0,
            "fixed_clip_paired_ci_lower": "> 0",
            "must_beat": ["fixed_task", "fixed_clip", "confidence_choice"],
            "min_heldout_macro_gain_pp": 0.20,
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
            "max_class_mass_shift_pp": 1.0,
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "inputs": {
            "full_target_source_signal": {
                "path": str(args.source_signal),
                "sha256": _sha256(args.source_signal),
            },
            "full_target_source_lock": {
                "path": str(args.source_lock),
                "sha256": _sha256(args.source_lock),
            },
            "opaque_exploratory_summary_sha256": _sha256(args.exploratory_summary),
            "full_target_list_opaque_sha256": _sha256(args.full_target_list),
            "proxy_list_opaque_sha256": _sha256(args.proxy_list),
            "class_names_sha256": _sha256(args.class_names),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "clip/model.py": _sha256(REPO_ROOT / "clip/model.py"),
            "src/utils/patch_cls_contribution_audit.py": _sha256(
                REPO_ROOT / "src/utils/patch_cls_contribution_audit.py"
            ),
            "src/utils/patch_cls_risk_control_audit.py": _sha256(
                REPO_ROOT / "src/utils/patch_cls_risk_control_audit.py"
            ),
            "tools/audit_visda_conflict_patch_cls_risk_control_holdout.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: target labels and prior exploratory oracle result are revealed
    # strictly after the held-out label-free signal has been locked.
    if (
        _sha256(args.full_target_list)
        != lock["inputs"]["full_target_list_opaque_sha256"]
    ):
        raise RuntimeError("Full target list changed after the signal lock")
    if (
        _sha256(args.exploratory_summary)
        != lock["inputs"]["opaque_exploratory_summary_sha256"]
    ):
        raise RuntimeError("Exploratory summary changed after the signal lock")
    labels = _parse_labels_after_lock(args.full_target_list)
    exploratory_summary = json.loads(args.exploratory_summary.read_text())
    exploratory_pass = (
        exploratory_summary.get("decision") == "PASS_EXPLORATORY_PATCH_CLS_RISK_CONTROL"
    )
    query_labels = labels[query_index]
    predictions = {
        "candidate": result["prediction"],
        "fixed_task": task_candidate,
        "fixed_clip": clip_candidate,
        "confidence_choice": confidence_prediction,
    }
    comparison_order = ("fixed_task", "fixed_clip", "confidence_choice")
    comparisons = {
        name: _comparison(
            predictions["candidate"],
            predictions[name],
            query_labels,
            repeats=args.bootstrap_repeats,
            seed=args.seed + offset,
        )
        for offset, name in enumerate(comparison_order)
    }
    selected = result["selected"]
    task_correct = predictions["fixed_task"] == query_labels
    clip_correct = predictions["fixed_clip"] == query_labels
    paired_resolved = selected & (task_correct | clip_correct)
    paired_precision = (
        float(task_correct[paired_resolved].mean() * 100.0)
        if paired_resolved.any()
        else 0.0
    )
    candidate_correct = predictions["candidate"] == query_labels

    heldout_indices = np.flatnonzero(holdout_sample_mask)
    heldout_labels = labels[heldout_indices]
    heldout_class_sizes = np.bincount(heldout_labels, minlength=EXPECTED_CLASSES)
    class_delta = np.zeros(EXPECTED_CLASSES, dtype=np.float64)
    class_rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_query = query_labels == class_index
        net_corrections = int(
            candidate_correct[class_query].sum() - clip_correct[class_query].sum()
        )
        class_delta[class_index] = (
            net_corrections / float(heldout_class_sizes[class_index]) * 100.0
        )
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "heldout_samples": int(heldout_class_sizes[class_index]),
                "heldout_conflict_samples": int(class_query.sum()),
                "selected_samples": int((selected & class_query).sum()),
                "net_corrections_vs_fixed_clip": net_corrections,
                "heldout_accuracy_delta_pp": float(class_delta[class_index]),
                "oracle_usage": "diagnostic_only_after_heldout_signal_lock",
            }
        )
    _write_csv(class_path, class_rows)
    oracle_rows: list[dict[str, Any]] = []
    for row, global_index in enumerate(query_index):
        oracle_rows.append(
            {
                "sample_index": int(global_index),
                "oracle_target_label": int(query_labels[row]),
                "selected_by_risk_control": bool(selected[row]),
                "candidate_prediction": int(predictions["candidate"][row]),
                "task_candidate": int(task_candidate[row]),
                "clip_candidate": int(clip_candidate[row]),
                "candidate_correct": bool(candidate_correct[row]),
                "fixed_clip_correct": bool(clip_correct[row]),
                "candidate_minus_fixed_clip_correct": int(candidate_correct[row])
                - int(clip_correct[row]),
                "oracle_usage": "diagnostic_only_after_heldout_signal_lock",
            }
        )
    _write_csv(oracle_path, oracle_rows)

    car_delta = float(class_delta[3])
    truck_delta = float(class_delta[11])
    car_truck_mean = float((car_delta + truck_delta) / 2.0)
    other_indices = [index for index in range(EXPECTED_CLASSES) if index not in (3, 11)]
    other_ten_mean = float(class_delta[other_indices].mean())
    heldout_macro_gain = float(class_delta.mean())
    gate = evaluate_patch_cls_holdout_gate(
        input_contract_valid=all(input_checks.values()),
        exploratory_pass_preserved=exploratory_pass,
        heldout_is_disjoint=label_free_metrics["proxy_overlap"] == 0,
        selected_coverage_pct=label_free_metrics["selected_coverage_pct"],
        paired_adjudication_precision_pct=paired_precision,
        comparisons=comparisons,
        heldout_macro_gain_pp=heldout_macro_gain,
        car_delta_pp=car_delta,
        truck_delta_pp=truck_delta,
        car_truck_mean_delta_pp=car_truck_mean,
        other_ten_mean_delta_pp=other_ten_mean,
        max_class_mass_shift_pp=label_free_metrics["max_class_mass_shift_pp"],
    )
    summary = {
        "decision": gate["decision"],
        "method": "frozen_patch_cls_upper_median_mass_cap_on_disjoint_holdout",
        "confirmatory_status": "disjoint_path_holdout_confirmation",
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "labels_read_only_after_signal_lock": True,
            "exploratory_pass_preserved": exploratory_pass,
            "selected_task_correct": int((selected & task_correct).sum()),
            "selected_clip_correct": int((selected & clip_correct).sum()),
            "selected_neither_correct": int(
                (selected & ~(task_correct | clip_correct)).sum()
            ),
            "paired_adjudication_precision_pct": paired_precision,
            "top1_union_coverage_pct": float(
                np.mean(task_correct | clip_correct) * 100.0
            ),
            "comparisons": comparisons,
            "heldout_macro_gain_pp": heldout_macro_gain,
            "classwise": class_rows,
            "car_delta_pp": car_delta,
            "truck_delta_pp": truck_delta,
            "car_truck_mean_delta_pp": car_truck_mean,
            "other_ten_mean_delta_pp": other_ten_mean,
        },
        "gate": gate,
        "outputs": {
            "label_free_signal": str(signal_path),
            "signal_lock": str(lock_path),
            "oracle_diagnostic": str(oracle_path),
            "classwise_oracle_diagnostic": str(class_path),
            "markdown": str(markdown_path),
        },
        "safety": {
            "source_or_task_model_loaded": False,
            "clip_model_frozen": True,
            "clip_image_forward_scope": "heldout_conflicts_only",
            "backward_calls": 0,
            "optimizer_constructed": False,
            "parameter_updates": 0,
            "proxy_training_started": False,
            "full_training_started": False,
            "full_training_authorized": False,
        },
        "scope_limit": (
            "PASS authorizes one exact parameter-impact audit only; it never "
            "authorizes proxy or full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(
        json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2)
    )
    print(f"Wrote held-out label-free signal: {signal_path}")
    print(f"Locked signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
