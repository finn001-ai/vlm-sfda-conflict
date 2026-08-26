#!/usr/bin/env python3
"""Frozen-CLIP patch-to-CLS contribution preflight for VisDA conflicts.

Phase 1 reads only the locked cycle-1 task/CLIP probabilities, uses a
deterministic CLIP center crop, and extracts final-block patch-specific terms
written into CLS.  Fixed CLIP remains the default; task is rescued only when
the full, even-head, and odd-head contribution readouts all prefer task's
conflicting top-1.  Target labels are parsed strictly after the signal lock.

This script loads no task/source checkpoint, creates no optimizer, performs no
backward pass, updates no parameter, and cannot start proxy or full training.
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
    evaluate_patch_cls_contribution_gate,
    unanimous_head_partition_rescue,
)


EXPECTED_SAMPLES = 13_847
EXPECTED_CLASSES = 12
EXPECTED_AGREEMENTS = 6_777
EXPECTED_CONFLICTS = 7_070
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
DEFAULT_BASE = Path(
    "output/uda/VISDA-C/TV/"
    "duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
)
STEM = "visda_conflict_patch_cls_contribution"


class _OpaquePathDataset(Dataset):
    """Load only image paths; the target-list suffix remains opaque."""

    def __init__(self, paths: list[str], sample_indices: np.ndarray, transform: Any):
        if len(paths) != sample_indices.size:
            raise ValueError("paths and sample indices must align")
        self.paths = paths
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        with open(self.paths[position], "rb") as handle:
            with Image.open(handle) as image:
                tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.sample_indices[position])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_BASE / "cycle2_conflict_memory_snapshots/pre_cycle01.npz",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=(
            DEFAULT_BASE
            / "cycle2_conflict_memory_audit"
            / "visda_cycle2_conflict_memory_signal_lock.json"
        ),
    )
    parser.add_argument(
        "--target-list",
        type=Path,
        default=Path("data/VISDA-C/validation_proxy25_seed2020_list.txt"),
    )
    parser.add_argument(
        "--class-names", type=Path, default=Path("data/VISDA-C/classname.txt")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE / "patch_cls_contribution_audit",
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


def _read_opaque_paths(path: Path) -> list[str]:
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
    if len(paths) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} paths, found {len(paths)}")
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
                f"Malformed target row {line_number}: {stripped}"
            ) from error
    result = np.asarray(labels, dtype=np.int64)
    if result.shape != (EXPECTED_SAMPLES,):
        raise ValueError(f"Expected {EXPECTED_SAMPLES} labels, found {result.size}")
    if np.any(result < 0) or np.any(result >= EXPECTED_CLASSES):
        raise ValueError("Target label outside class range")
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
    oracle = summary["oracle_diagnostic"]
    best_name = oracle["best_baseline_name"]
    best = oracle["comparisons"][best_name]
    lines = [
        "# VisDA Patch-to-CLS Contribution Preflight",
        "",
        "## Evidence table",
        "",
        "| Evidence | Result | Provenance |",
        "|---|---:|---|",
        (
            "| Task/CLIP top-1 union coverage | "
            f"`{oracle['top1_union_oracle_coverage_pct']:.6f}%` "
            "| Oracle diagnostic after lock |"
        ),
        (
            "| Contribution rescue coverage | "
            f"`{summary['label_free_metrics']['route_coverage_pct']:.6f}%` "
            "| Label-free signal lock |"
        ),
        (
            "| All/even/odd decision agreement | "
            f"`{summary['label_free_metrics']['all_partition_agreement_pct']:.6f}%` "
            "| Label-free signal lock |"
        ),
        (
            f"| Conflict gain vs best baseline `{best_name}` | "
            f"`{best['gain_pp']:.6f}` pp; CI "
            f"`{best['paired_bootstrap_95_ci_pp']}` "
            "| Oracle diagnostic after lock |"
        ),
        (
            "| Routed task precision | "
            f"`{oracle['routed_task_precision_pct']:.6f}%` "
            "| Oracle diagnostic after lock |"
        ),
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "## Label-free rule",
        "",
        "For each locked cycle-1 task/CLIP top-1 conflict, the audit extracts",
        "the final CLIP visual block's patch-specific value terms written into",
        "CLS. It compares the maximum local text response of the two top-1",
        "candidates. Fixed CLIP remains the default. Task is rescued only when",
        "the full-head, even-head, and odd-head readouts all prefer task. There",
        "is no fitted threshold, class route, label-conditioned rule, new model,",
        "optimizer, backward pass, or parameter update.",
        "",
        "## Scope",
        "",
        "This borrows only the patch-contribution equations from TraceCLIP,",
        "not its topology gate or segmentation method. The paper reports ViT-B/16",
        "and ViT-L/14; usefulness for DUET's ViT-B/32 is unverified and is the",
        "main preflight hypothesis.",
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
            "authorizes or starts proxy/full VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    if args.arch != "ViT-B/32":
        raise ValueError("This preflight is locked to DUET's CLIP ViT-B/32")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    for required in (
        args.snapshot,
        args.source_lock,
        args.target_list,
        args.class_names,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen CLIP forward preflight")

    snapshot_sha256 = _sha256(args.snapshot)
    source_lock = json.loads(args.source_lock.read_text())
    # Phase 1: never access the snapshot target_label key here.
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        cycle = int(np.asarray(snapshot["cycle"]).item())
        sample_index = np.asarray(snapshot["sample_index"], dtype=np.int64).copy()
        task_probability = np.asarray(snapshot["task_prob"], dtype=np.float64).copy()
        clip_probability = np.asarray(snapshot["clip_prob"], dtype=np.float64).copy()
        task_label = np.asarray(snapshot["source_label"], dtype=np.int64).copy()
        clip_label = np.asarray(snapshot["clip_label"], dtype=np.int64).copy()
        admitted = np.asarray(snapshot["label_mask"], dtype=bool).copy()
    agreement = task_label == clip_label
    conflict = ~agreement
    query = np.flatnonzero(conflict)
    class_names = _load_class_names(args.class_names)
    opaque_paths = _read_opaque_paths(args.target_list)
    query_paths = [opaque_paths[index] for index in query]

    # Reject a stale or mismatched artifact contract before loading CLIP or
    # spending any GPU time.
    pre_forward_checks = {
        "snapshot_matches_cycle_memory_lock": (
            snapshot_sha256 == source_lock.get("inputs", {}).get("pre_cycle1_sha256")
        ),
        "snapshot_is_pre_cycle1": cycle == 1,
        "probability_shapes": (
            task_probability.shape
            == clip_probability.shape
            == (EXPECTED_SAMPLES, EXPECTED_CLASSES)
        ),
        "label_free_probabilities_finite": all(
            np.isfinite(value).all() for value in (task_probability, clip_probability)
        ),
        "probabilities_normalized": all(
            np.allclose(value.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5)
            for value in (task_probability, clip_probability)
        ),
        "saved_predictions_match_probabilities": (
            np.array_equal(task_label, task_probability.argmax(axis=1))
            and np.array_equal(clip_label, clip_probability.argmax(axis=1))
        ),
        "admitted_mask_matches_cycle1_agreement": np.array_equal(admitted, agreement),
        "expected_agreement_and_conflict_counts": (
            int(agreement.sum()) == EXPECTED_AGREEMENTS
            and int(conflict.sum()) == EXPECTED_CONFLICTS
        ),
        "sample_indices_are_proxy_order": np.array_equal(
            sample_index, np.arange(EXPECTED_SAMPLES)
        ),
        "every_query_image_exists": bool(query_paths)
        and all(Path(path).is_file() for path in query_paths),
    }
    pre_forward_checks = {
        name: bool(value) for name, value in pre_forward_checks.items()
    }
    pre_forward_failed = [
        name for name, passed in pre_forward_checks.items() if not passed
    ]
    if pre_forward_failed:
        raise RuntimeError(
            f"Patch-to-CLS pre-forward contract failed: {pre_forward_failed}"
        )

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    clip_model, _embed_dim, preprocess = clip.load(args.arch, device=device, jit=False)
    clip_model.float().eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    prompt_prefix = args.ctx_init.replace("_", " ")
    prompts = [f"{prompt_prefix} {name}." for name in class_names]
    tokens = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)
    with torch.no_grad():
        text_feature = F.normalize(clip_model.encode_text(tokens).float(), dim=1)

    loader = DataLoader(
        _OpaquePathDataset(query_paths, query, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "sample_index",
            "full_task_peak",
            "full_clip_peak",
            "even_task_peak",
            "even_clip_peak",
            "odd_task_peak",
            "odd_clip_peak",
            "full_task_patch",
            "full_clip_patch",
        )
    }
    partition_errors: list[float] = []
    with torch.no_grad():
        for images, global_index in loader:
            images = images.to(device, non_blocking=True)
            global_index = global_index.numpy().astype(np.int64, copy=False)
            task_candidate = torch.from_numpy(task_label[global_index]).to(device)
            clip_candidate = torch.from_numpy(clip_label[global_index]).to(device)
            contribution = encode_last_block_patch_cls_contributions(clip_model, images)
            response = {
                partition: candidate_peak_response(
                    contribution[partition],
                    text_feature,
                    task_candidate,
                    clip_candidate,
                )
                for partition in ("all", "even", "odd")
            }
            collected["sample_index"].append(global_index)
            for partition, prefix in (
                ("all", "full"),
                ("even", "even"),
                ("odd", "odd"),
            ):
                collected[f"{prefix}_task_peak"].append(
                    response[partition]["task_peak"].cpu().numpy()
                )
                collected[f"{prefix}_clip_peak"].append(
                    response[partition]["clip_peak"].cpu().numpy()
                )
            collected["full_task_patch"].append(
                response["all"]["task_peak_patch"].cpu().numpy()
            )
            collected["full_clip_patch"].append(
                response["all"]["clip_peak_patch"].cpu().numpy()
            )
            partition_errors.append(
                float(contribution["head_partition_max_abs_error"].item())
            )
    arrays = {name: np.concatenate(value) for name, value in collected.items()}
    if not np.array_equal(arrays["sample_index"], query):
        raise RuntimeError("CLIP contribution loader changed locked conflict order")
    candidate = unanimous_head_partition_rescue(
        task_label[query],
        clip_label[query],
        arrays["full_task_peak"],
        arrays["full_clip_peak"],
        arrays["even_task_peak"],
        arrays["even_clip_peak"],
        arrays["odd_task_peak"],
        arrays["odd_clip_peak"],
    )

    confidence_prediction = np.where(
        task_probability.max(axis=1) >= clip_probability.max(axis=1),
        task_label,
        clip_label,
    )
    arithmetic_prediction = (0.5 * (task_probability + clip_probability)).argmax(axis=1)
    rms_prediction = np.sqrt(
        0.5 * (task_probability**2 + clip_probability**2)
    ).argmax(axis=1)
    candidate_full = clip_label.copy()
    candidate_full[query] = candidate["prediction"]
    class_mass_shift_pp = (
        (
            np.bincount(candidate_full, minlength=EXPECTED_CLASSES)
            - np.bincount(clip_label, minlength=EXPECTED_CLASSES)
        )
        / EXPECTED_SAMPLES
        * 100.0
    )
    max_partition_error = max(partition_errors, default=float("inf"))
    route_coverage_pct = float(candidate["rescue_task"].mean() * 100.0)
    partition_agreement_pct = float(candidate["all_partition_agreement"].mean() * 100.0)
    input_checks = {
        **pre_forward_checks,
        "label_free_contribution_arrays_finite": all(
            np.isfinite(value).all() for value in arrays.values()
        ),
        "query_order_is_exact_conflict_order": np.array_equal(
            arrays["sample_index"], query
        ),
        "candidate_is_only_task_or_clip_top1": bool(
            np.all(
                (candidate["prediction"] == task_label[query])
                | (candidate["prediction"] == clip_label[query])
            )
        ),
        "clip_model_is_frozen": not any(
            parameter.requires_grad for parameter in clip_model.parameters()
        ),
        "duet_architecture_locked": (
            args.arch == "ViT-B/32"
            and int(clip_model.visual.transformer.resblocks[-1].attn.num_heads) == 12
        ),
        "head_partition_identity_finite": np.isfinite(max_partition_error),
    }
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    failed = [name for name, passed in input_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Patch-to-CLS contribution input contract failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    signal_path = args.output_dir / f"{STEM}_label_free.npz"
    lock_path = args.output_dir / f"{STEM}_signal_lock.json"
    oracle_path = args.output_dir / f"{STEM}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{STEM}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{STEM}_summary.json"
    markdown_path = args.output_dir / f"{STEM}_summary.md"
    np.savez_compressed(
        signal_path,
        query_index=query,
        task_candidate=task_label[query],
        clip_candidate=clip_label[query],
        candidate_prediction=candidate["prediction"],
        rescue_task=candidate["rescue_task"],
        full_choose_task=candidate["full_choose_task"],
        even_choose_task=candidate["even_choose_task"],
        odd_choose_task=candidate["odd_choose_task"],
        all_partition_agreement=candidate["all_partition_agreement"],
        minimum_task_margin=candidate["minimum_task_margin"].astype(np.float32),
        full_task_peak=arrays["full_task_peak"].astype(np.float32),
        full_clip_peak=arrays["full_clip_peak"].astype(np.float32),
        even_task_peak=arrays["even_task_peak"].astype(np.float32),
        even_clip_peak=arrays["even_clip_peak"].astype(np.float32),
        odd_task_peak=arrays["odd_task_peak"].astype(np.float32),
        odd_clip_peak=arrays["odd_clip_peak"].astype(np.float32),
        full_task_peak_patch=arrays["full_task_patch"].astype(np.int16),
        full_clip_peak_patch=arrays["full_clip_patch"].astype(np.int16),
        fixed_task_prediction=task_label[query],
        fixed_clip_prediction=clip_label[query],
        confidence_prediction=confidence_prediction[query],
        arithmetic_prediction=arithmetic_prediction[query],
        rms_prediction=rms_prediction[query],
    )
    label_free_metrics = {
        "samples": EXPECTED_SAMPLES,
        "conflict_queries": EXPECTED_CONFLICTS,
        "architecture": args.arch,
        "patch_grid": "7x7",
        "attention_heads": 12,
        "route_samples": int(candidate["rescue_task"].sum()),
        "route_coverage_pct": route_coverage_pct,
        "even_odd_decision_agreement_pct": float(
            candidate["even_odd_agreement"].mean() * 100.0
        ),
        "all_partition_agreement_pct": partition_agreement_pct,
        "head_partition_max_abs_error": max_partition_error,
        "class_mass_shift_pp": {
            name: float(class_mass_shift_pp[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "max_class_mass_shift_pp": float(np.abs(class_mass_shift_pp).max()),
    }
    lock = {
        "phase": "LABEL_FREE_VISDA_PATCH_CLS_CONTRIBUTION_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "target_paths_loaded_before_lock": True,
        "target_list_suffix_treated_as_opaque_before_lock": True,
        "source_snapshot_contains_target_label_but_key_not_accessed_before_lock": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate_contract": {
            "query": "locked_cycle1_task_clip_top1_conflicts",
            "default_prediction": "fixed_clip_top1",
            "alternative_prediction": "fixed_task_top1",
            "image_view": "deterministic_official_clip_center_crop",
            "new_information": "final_visual_block_patch_specific_terms_written_into_CLS",
            "patch_score": "maximum_normalized_contribution_text_cosine",
            "rescue_rule": "task_wins_full_even_and_odd_head_partitions",
            "numerical_thresholds": False,
            "class_specific_routes": False,
            "fitted_parameters": False,
            "target_label_thresholds": False,
            "training_change_in_this_audit": False,
        },
        "predeclared_gate": {
            "max_head_partition_identity_error": 1e-5,
            "min_all_partition_decision_agreement_pct": 80.0,
            "route_coverage_pct": [2.0, 30.0],
            "min_routed_task_precision_pct": 60.0,
            "routed_net_corrections": "> 0",
            "min_accuracy_gain_vs_best_baseline_pp": 1.0,
            "best_baseline_paired_ci_lower": "> 0",
            "min_full_proxy_macro_gain_pp": 0.20,
            "max_individual_car_truck_regression_pp": 0.5,
            "car_truck_and_other10_mean_delta_pp": ">= 0",
            "max_class_mass_shift_pp": 1.0,
        },
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "literature_provenance": {
            "paper": "TraceCLIP: Recovering Local Semantics from Patch-to-CLS Contributions",
            "submitted": "2026-07-28",
            "paper_url": "https://arxiv.org/abs/2607.26107",
            "borrowed_information": "final_block_patch_specific_CLS_attention_contribution_equations",
            "not_borrowed": "semantic_geodesic_topology_gate_or_segmentation_pipeline",
            "backbone_limitation": "paper_evaluates_ViT-B16_and_ViT-L14_not_DUET_ViT-B32",
        },
        "inputs": {
            "pre_cycle1_snapshot": {
                "path": str(args.snapshot),
                "sha256": snapshot_sha256,
            },
            "cycle2_memory_signal_lock": {
                "path": str(args.source_lock),
                "sha256": _sha256(args.source_lock),
            },
            "target_list_opaque_sha256": _sha256(args.target_list),
            "class_names_sha256": _sha256(args.class_names),
        },
        "signal_npz": {"path": str(signal_path), "sha256": _sha256(signal_path)},
        "contract_sha256": {
            "clip/model.py": _sha256(REPO_ROOT / "clip/model.py"),
            "src/utils/patch_cls_contribution_audit.py": _sha256(
                REPO_ROOT / "src/utils/patch_cls_contribution_audit.py"
            ),
            "tools/audit_visda_conflict_patch_cls_contribution.py": _sha256(
                Path(__file__).resolve()
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: explicit oracle diagnostic, strictly after the signal lock.
    if _sha256(args.target_list) != lock["inputs"]["target_list_opaque_sha256"]:
        raise RuntimeError("Target-list hash changed after the label-free lock")
    labels = _parse_labels_after_lock(args.target_list)
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        embedded_labels = np.asarray(snapshot["target_label"], dtype=np.int64).copy()
    if not np.array_equal(labels, embedded_labels):
        raise RuntimeError("Oracle labels do not match the locked snapshot order")
    query_labels = labels[query]
    predictions = {
        "candidate": candidate["prediction"],
        "fixed_task": task_label[query],
        "fixed_clip": clip_label[query],
        "confidence_choice": confidence_prediction[query],
        "arithmetic": arithmetic_prediction[query],
        "rms": rms_prediction[query],
    }
    full_predictions = {
        "candidate": candidate_full,
        "fixed_task": task_label,
        "fixed_clip": clip_label,
        "confidence_choice": confidence_prediction,
        "arithmetic": arithmetic_prediction,
        "rms": rms_prediction,
    }
    comparison_order = (
        "fixed_task",
        "fixed_clip",
        "confidence_choice",
        "arithmetic",
        "rms",
    )
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
    best_baseline_name = max(
        comparisons,
        key=lambda name: (comparisons[name]["baseline_accuracy_pct"], name),
    )
    best_full_prediction = full_predictions[best_baseline_name]
    routed = candidate["rescue_task"]
    routed_task_correct = predictions["fixed_task"][routed] == query_labels[routed]
    routed_clip_correct = predictions["fixed_clip"][routed] == query_labels[routed]
    routed_task_precision_pct = (
        float(routed_task_correct.mean() * 100.0) if routed.any() else 0.0
    )
    routed_net_corrections = int(routed_task_correct.sum() - routed_clip_correct.sum())
    top1_union_coverage = (predictions["fixed_task"] == query_labels) | (
        predictions["fixed_clip"] == query_labels
    )
    available_task_rescue = predictions["fixed_task"] == query_labels
    recovered_task_rescue = routed & available_task_rescue

    oracle_rows: list[dict[str, Any]] = []
    for row, global_index in enumerate(query):
        oracle_rows.append(
            {
                "sample_index": int(global_index),
                "oracle_target_label": int(query_labels[row]),
                "oracle_top1_union_contains_target": bool(top1_union_coverage[row]),
                "task_candidate": int(predictions["fixed_task"][row]),
                "clip_candidate": int(predictions["fixed_clip"][row]),
                "rescue_task": bool(routed[row]),
                "candidate_prediction": int(predictions["candidate"][row]),
                "candidate_correct": bool(
                    predictions["candidate"][row] == query_labels[row]
                ),
                "fixed_clip_correct": bool(
                    predictions["fixed_clip"][row] == query_labels[row]
                ),
                "full_choose_task": bool(candidate["full_choose_task"][row]),
                "even_choose_task": bool(candidate["even_choose_task"][row]),
                "odd_choose_task": bool(candidate["odd_choose_task"][row]),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(oracle_path, oracle_rows)

    class_rows: list[dict[str, Any]] = []
    class_delta = np.empty(EXPECTED_CLASSES, dtype=np.float64)
    for class_index, class_name in enumerate(CLASS_NAMES):
        selected = query_labels == class_index
        selected_full = labels == class_index
        candidate_accuracy = float(
            (
                full_predictions["candidate"][selected_full] == labels[selected_full]
            ).mean()
            * 100.0
        )
        best_accuracy = float(
            (best_full_prediction[selected_full] == labels[selected_full]).mean()
            * 100.0
        )
        class_delta[class_index] = candidate_accuracy - best_accuracy
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "full_proxy_samples": int(selected_full.sum()),
                "conflict_samples": int(selected.sum()),
                "top1_union_oracle_coverage_pct": float(
                    top1_union_coverage[selected].mean() * 100.0
                ),
                "routed_samples": int((routed & selected).sum()),
                "candidate_full_proxy_accuracy_pct": candidate_accuracy,
                "best_baseline_name": best_baseline_name,
                "best_baseline_full_proxy_accuracy_pct": best_accuracy,
                "candidate_minus_best_baseline_full_proxy_pp": float(
                    class_delta[class_index]
                ),
                "oracle_usage": "diagnostic_only_after_label_free_lock",
            }
        )
    _write_csv(class_path, class_rows)
    car_delta = float(class_delta[3])
    truck_delta = float(class_delta[11])
    car_truck_mean_delta = float((car_delta + truck_delta) / 2.0)
    other_indices = [index for index in range(EXPECTED_CLASSES) if index not in (3, 11)]
    other_ten_mean_delta = float(class_delta[other_indices].mean())
    full_proxy_macro_gain_pp = float(class_delta.mean())
    gate = evaluate_patch_cls_contribution_gate(
        input_contract_valid=all(input_checks.values()),
        head_partition_max_abs_error=max_partition_error,
        all_partition_agreement_pct=partition_agreement_pct,
        route_coverage_pct=route_coverage_pct,
        comparisons=comparisons,
        routed_net_corrections=routed_net_corrections,
        routed_task_precision_pct=routed_task_precision_pct,
        full_proxy_macro_gain_pp=full_proxy_macro_gain_pp,
        car_delta_pp=car_delta,
        truck_delta_pp=truck_delta,
        car_truck_mean_delta_pp=car_truck_mean_delta,
        other_ten_mean_delta_pp=other_ten_mean_delta,
        max_class_mass_shift_pp=label_free_metrics["max_class_mass_shift_pp"],
    )
    summary = {
        "decision": gate["decision"],
        "method": "frozen_clip_final_block_patch_cls_unanimous_task_rescue",
        "input_contract_checks": input_checks,
        "label_free_metrics": label_free_metrics,
        "oracle_diagnostic": {
            "explicit_oracle_diagnostic": True,
            "top1_union_oracle_coverage_pct": float(top1_union_coverage.mean() * 100.0),
            "available_task_rescue_samples": int(available_task_rescue.sum()),
            "recovered_task_rescue_samples": int(recovered_task_rescue.sum()),
            "recovered_available_task_rescue_pct": float(
                recovered_task_rescue.sum()
                / max(1, available_task_rescue.sum())
                * 100.0
            ),
            "routed_task_precision_pct": routed_task_precision_pct,
            "routed_net_corrections": routed_net_corrections,
            "full_proxy_macro_gain_pp": full_proxy_macro_gain_pp,
            "comparisons": comparisons,
            "best_baseline_name": best_baseline_name,
            "classwise": class_rows,
            "car_delta_pp": car_delta,
            "truck_delta_pp": truck_delta,
            "car_truck_mean_delta_pp": car_truck_mean_delta,
            "other_ten_mean_delta_pp": other_ten_mean_delta,
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
            "task_model_loaded": False,
            "source_checkpoint_loaded": False,
            "clip_parameters_frozen": True,
            "optimizer_constructed": False,
            "backward_calls": 0,
            "parameter_updates": 0,
            "training_authorized": False,
        },
        "scope_limit": (
            "PASS authorizes one exact parameter-impact audit only. It never "
            "authorizes or starts proxy/full VisDA training."
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(
        json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2)
    )
    print(f"Wrote label-free patch contribution signal: {signal_path}")
    print(f"Locked patch contribution signal before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote classwise oracle diagnostic: {class_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
