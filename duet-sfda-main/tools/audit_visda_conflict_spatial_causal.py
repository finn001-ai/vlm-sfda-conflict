#!/usr/bin/env python
"""Frozen ViT-B/32 spatial-causal audit for DUET VisDA conflicts.

Phase 1 ignores loader labels, reproduces the released DUET task/CLIP forward
path, locks a prediction-defined pilot, and probes both frozen models with the
same fixed balanced occlusion bank.  Target labels are parsed only after the
signal CSV/NPZ and lock manifest have been written.  This script constructs no
optimizer, performs no backward pass, and cannot update model parameters.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import clip  # noqa: E402
from conf import cfg  # noqa: E402
from src.methods.oh.plmatch import clip_pre_text, data_load  # noqa: E402
from src.utils.conflict_boundary import paired_accuracy_bootstrap_ci  # noqa: E402
from src.utils.probability_fusion import (  # noqa: E402
    arithmetic_probability_fusion,
    rms_probability_fusion,
)
from src.utils.spatial_causal_audit import (  # noqa: E402
    balanced_binary_masks,
    deterministic_hash_sample,
    evaluate_spatial_gate,
    spatial_consensus_selector,
    topk_union_candidates,
)
from tools.export_conflict_diagnostics import (  # noqa: E402
    _build_source_model,
    _load_class_names,
    _prepare_cfg,
)


MASK_COUNT = 64
MASK_GRID_SIZE = 7
PAIR_SAMPLE_COUNT = 512
OTHER_SAMPLE_COUNT = 512
TOP_K = 2
BOOTSTRAP_REPEATS = 2_000
MASK_FORWARD_BATCH = 64
MASK_SAMPLE_CHUNK = 4
REPRODUCTION_MAX_ERROR = 1e-6
EXPECTED_VISDA = {
    "total_samples": 55_388,
    "agreement_samples": 27_165,
    "agreement_accuracy": 93.98,
    "task_accuracy": 51.45,
    "clip_accuracy": 82.87,
    "arithmetic_mix_accuracy": 73.83,
}
REPRODUCTION_TOLERANCES = {
    "agreement_count_fraction": 0.0025,
    "agreement_accuracy_pp": 0.25,
    "task_accuracy_pp": 0.10,
    "clip_accuracy_pp": 0.15,
    "arithmetic_mix_accuracy_pp": 0.10,
}


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> tuple[Any, Any, torch.Tensor, list[torch.Tensor]]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )


def _restore_rng_state(
    state: tuple[Any, Any, torch.Tensor, list[torch.Tensor]],
) -> None:
    python_state, numpy_state, torch_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    torch.cuda.set_rng_state_all(cuda_state)


def _pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_labels_after_lock(
    list_path: str | Path, ordered_image_paths: list[str]
) -> np.ndarray:
    labels_by_path: dict[str, int] = {}
    with Path(list_path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                image_path, label_text = stripped.rsplit(maxsplit=1)
                label = int(label_text)
            except ValueError as error:
                raise ValueError(
                    f"Malformed VisDA row {line_number} in {list_path}: {stripped}"
                ) from error
            if image_path in labels_by_path:
                raise ValueError(f"Duplicate image path in VisDA list: {image_path}")
            labels_by_path[image_path] = label
    missing = [path for path in ordered_image_paths if path not in labels_by_path]
    if missing:
        raise ValueError(f"VisDA list is missing audit path: {missing[0]}")
    if len(labels_by_path) != len(ordered_image_paths):
        raise ValueError("VisDA list and loader rows do not match")
    return np.asarray([labels_by_path[path] for path in ordered_image_paths], dtype=np.int64)


def _accuracy(prediction: np.ndarray, label: np.ndarray, mask: np.ndarray) -> float:
    return _pct(np.sum(prediction[mask] == label[mask]), np.sum(mask))


def _baseline_reproduction(
    observed: dict[str, Any], ordered_indices: np.ndarray
) -> dict[str, Any]:
    expected_count = EXPECTED_VISDA["total_samples"]
    agreement_delta = abs(
        observed["agreement_samples"] - EXPECTED_VISDA["agreement_samples"]
    )
    checks = {
        "full_target_count": observed["total_samples"] == expected_count,
        "contiguous_unique_indices": np.array_equal(
            ordered_indices, np.arange(ordered_indices.size)
        ),
        "full_adaptation_list": not bool(str(cfg.ACTIVE.ADAPTATION_LIST).strip()),
        "agreement_count_within_fraction": (
            agreement_delta / expected_count
            <= REPRODUCTION_TOLERANCES["agreement_count_fraction"]
        ),
        "agreement_accuracy_within_tolerance": (
            abs(observed["agreement_accuracy"] - EXPECTED_VISDA["agreement_accuracy"])
            <= REPRODUCTION_TOLERANCES["agreement_accuracy_pp"]
        ),
        "task_accuracy_within_tolerance": (
            abs(observed["task_accuracy"] - EXPECTED_VISDA["task_accuracy"])
            <= REPRODUCTION_TOLERANCES["task_accuracy_pp"]
        ),
        "clip_accuracy_within_tolerance": (
            abs(observed["clip_accuracy"] - EXPECTED_VISDA["clip_accuracy"])
            <= REPRODUCTION_TOLERANCES["clip_accuracy_pp"]
        ),
        "arithmetic_mix_accuracy_within_tolerance": (
            abs(
                observed["arithmetic_mix_accuracy"]
                - EXPECTED_VISDA["arithmetic_mix_accuracy"]
            )
            <= REPRODUCTION_TOLERANCES["arithmetic_mix_accuracy_pp"]
        ),
    }
    return {
        "expected": EXPECTED_VISDA,
        "tolerances": REPRODUCTION_TOLERANCES,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _paired_comparison(
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    candidate_correct = candidate_prediction[mask] == labels[mask]
    baseline_correct = baseline_prediction[mask] == labels[mask]
    candidate_accuracy = _pct(candidate_correct.sum(), candidate_correct.size)
    baseline_accuracy = _pct(baseline_correct.sum(), baseline_correct.size)
    ci = paired_accuracy_bootstrap_ci(
        candidate_correct,
        baseline_correct,
        repeats=BOOTSTRAP_REPEATS,
        seed=int(cfg.SETTING.SEED),
    )
    return {
        "samples": int(mask.sum()),
        "candidate_accuracy": candidate_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "gain_pp": candidate_accuracy - baseline_accuracy,
        "paired_bootstrap_95_ci_pp": list(ci),
        "net_corrections": int(candidate_correct.sum() - baseline_correct.sum()),
    }


def _masked_probabilities(
    images: torch.Tensor,
    masks: torch.Tensor,
    net_f: torch.nn.Module,
    net_b: torch.nn.Module,
    net_c: torch.nn.Module,
    clip_model: torch.nn.Module,
    text_features: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = images.size(0)
    mask_count = masks.size(0)
    class_count = text_features.size(0)
    task_result = np.empty((sample_count, mask_count, class_count), dtype=np.float32)
    clip_result = np.empty_like(task_result)
    for sample_start in range(0, sample_count, MASK_SAMPLE_CHUNK):
        sample_stop = min(sample_start + MASK_SAMPLE_CHUNK, sample_count)
        current = images[sample_start:sample_stop]
        masked = (current[:, None] * masks[None]).reshape(
            -1, current.size(1), current.size(2), current.size(3)
        )
        task_parts = []
        clip_parts = []
        with torch.no_grad():
            for start in range(0, masked.size(0), MASK_FORWARD_BATCH):
                stop = min(start + MASK_FORWARD_BATCH, masked.size(0))
                masked_batch = masked[start:stop]
                task_logits = net_c(net_b(net_f(masked_batch)))
                clip_features = F.normalize(
                    clip_model.encode_image(masked_batch), dim=1
                )
                clip_logits = (
                    clip_model.logit_scale.exp() * clip_features @ text_features.t()
                )
                task_parts.append(torch.softmax(task_logits, dim=1).cpu())
                clip_parts.append(torch.softmax(clip_logits, dim=1).cpu())
        current_count = sample_stop - sample_start
        task_result[sample_start:sample_stop] = (
            torch.cat(task_parts).reshape(current_count, mask_count, class_count).numpy()
        )
        clip_result[sample_start:sample_stop] = (
            torch.cat(clip_parts).reshape(current_count, mask_count, class_count).numpy()
        )
        del masked
    return task_result, clip_result


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    conflict = summary["full_conflict_oracle_diagnostic"]
    pilot = summary["pilot_oracle_diagnostic"]
    gate = summary["gate"]
    lines = [
        "# VisDA DUET Spatial-Causal Offline Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The signal uses the unchanged frozen source model and CLIP ViT-B/32.",
        "Target labels were parsed only after signal artifacts were SHA256-locked.",
        "No optimizer, backward pass, parameter update, or training is present.",
        "",
        "## Full conflict oracle diagnostic",
        "",
        f"- Conflicts: `{conflict['samples']}`.",
        f"- Top-1 task/CLIP union coverage: `{conflict['top1_union_coverage']:.4f}%`.",
        f"- Top-2 task/CLIP union coverage: `{conflict['top2_union_coverage']:.4f}%`.",
        f"- Top-2 coverage inside top-1-neither rows: "
        f"`{conflict['top2_coverage_within_top1_neither']:.4f}%`.",
        "",
        "## Balanced pilot oracle diagnostic",
        "",
        f"- Samples: `{pilot['samples']}`.",
        f"- Spatial selector accuracy: `{pilot['spatial_accuracy']:.4f}%`.",
        f"- Fixed CLIP accuracy: `{pilot['fixed_clip_accuracy']:.4f}%`.",
        f"- Gain: `{pilot['versus_fixed_clip']['gain_pp']:.4f} pp`; 95% CI "
        f"`{pilot['versus_fixed_clip']['paired_bootstrap_95_ci_pp']}`.",
        f"- Car/truck-pair gain: `{pilot['car_truck_versus_clip']['gain_pp']:.4f} pp`.",
        f"- Eligible top-2 rescue rate: `{pilot['eligible_top2_rescue_rate']:.4f}%`.",
        f"- Median split-half stability: `{pilot['median_split_half_stability']:.6f}`.",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(f"- {name}: `{passed}`" for name, passed in gate["checks"].items())
    lines.extend(
        [
            "",
            "Passing authorizes only a full offline audit after separate approval; it",
            "does not authorize proxy or full VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    _prepare_cfg()
    if cfg.SETTING.DATASET != "VISDA-C" or cfg.SETTING.S != 0 or cfg.SETTING.T != 1:
        raise ValueError("This audit is restricted to VisDA train->validation")
    if str(cfg.ACTIVE.ADAPTATION_LIST).strip():
        raise ValueError("Spatial-causal audit requires the full target list")
    if str(cfg.ACTIVE.ARCH) != "ViT-B/32":
        raise ValueError("Spatial-causal audit is locked to the DUET ViT-B/32 path")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VisDA spatial-causal audit")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    _seed_everything(int(cfg.SETTING.SEED))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    class_names = _load_class_names()
    normalized_names = [name.strip().lower() for name in class_names]
    if "car" not in normalized_names or "truck" not in normalized_names:
        raise ValueError("VisDA class list must contain car and truck")
    car_index = normalized_names.index("car")
    truck_index = normalized_names.index("truck")
    test_loader = data_load(cfg)["test_aug"]
    ordered_paths = [str(record[0]) for record in test_loader.dataset.imgs]
    if len(ordered_paths) != EXPECTED_VISDA["total_samples"]:
        raise ValueError(
            f"Expected {EXPECTED_VISDA['total_samples']} target rows, "
            f"found {len(ordered_paths)}"
        )

    net_f, net_b, net_c = _build_source_model(device)
    clip_model, _, _ = clip.load(cfg.ACTIVE.ARCH, device=device)
    clip_model.float()
    for module in (net_f, net_b, net_c, clip_model):
        _freeze(module)
    text_inputs = clip_pre_text(cfg).to(device)
    with torch.no_grad():
        text_features = F.normalize(clip_model.encode_text(text_inputs), dim=1).detach()

    full_parts: dict[str, list[np.ndarray]] = {
        "index": [],
        "task_probability": [],
        "clip_probability": [],
        "task_prediction": [],
        "clip_prediction": [],
        "task_confidence": [],
        "clip_confidence": [],
        "arithmetic_prediction": [],
        "rms_prediction": [],
    }
    first_pass_rng = _capture_rng_state()
    # Phase 1a: loader labels are deliberately ignored.
    for inputs, _ignored_labels, indices in test_loader:
        del _ignored_labels
        weak_x = inputs[1].to(device)
        with torch.no_grad():
            task_logits = net_c(net_b(net_f(weak_x)))
            clip_features = F.normalize(clip_model.encode_image(weak_x), dim=1)
            clip_logits = clip_model.logit_scale.exp() * clip_features @ text_features.t()
            task_probability = torch.softmax(task_logits, dim=1)
            clip_probability = torch.softmax(clip_logits, dim=1)
            task_confidence, task_prediction = task_probability.max(dim=1)
            clip_confidence, clip_prediction = clip_probability.max(dim=1)
            arithmetic_prediction = arithmetic_probability_fusion(
                task_probability, clip_probability
            ).argmax(dim=1)
            rms_prediction = rms_probability_fusion(
                task_probability, clip_probability
            ).argmax(dim=1)
        for name, value in (
            ("index", indices.long().cpu().numpy()),
            ("task_probability", task_probability.cpu().numpy()),
            ("clip_probability", clip_probability.cpu().numpy()),
            ("task_prediction", task_prediction.cpu().numpy()),
            ("clip_prediction", clip_prediction.cpu().numpy()),
            ("task_confidence", task_confidence.cpu().numpy()),
            ("clip_confidence", clip_confidence.cpu().numpy()),
            ("arithmetic_prediction", arithmetic_prediction.cpu().numpy()),
            ("rms_prediction", rms_prediction.cpu().numpy()),
        ):
            full_parts[name].append(value)

    full = {name: np.concatenate(parts) for name, parts in full_parts.items()}
    order = np.argsort(full["index"], kind="mergesort")
    full = {name: value[order] for name, value in full.items()}
    if not np.array_equal(full["index"], np.arange(len(ordered_paths))):
        raise RuntimeError("Full-target indices are not unique contiguous VisDA indices")

    conflict = full["task_prediction"] != full["clip_prediction"]
    car_truck_pair = (
        conflict
        & (full["task_prediction"] == car_index)
        & (full["clip_prediction"] == truck_index)
    )
    other_conflict = conflict & ~car_truck_pair
    pair_indices = deterministic_hash_sample(
        ordered_paths,
        car_truck_pair,
        count=PAIR_SAMPLE_COUNT,
        namespace="task-car_clip-truck",
    )
    other_indices = deterministic_hash_sample(
        ordered_paths,
        other_conflict,
        count=OTHER_SAMPLE_COUNT,
        namespace="other-conflict",
    )
    selected_indices = np.sort(np.concatenate((pair_indices, other_indices)))
    selected_stratum = np.where(
        car_truck_pair[selected_indices], "task_car_clip_truck", "other_conflict"
    )
    selected_task_probability = full["task_probability"][selected_indices]
    selected_clip_probability = full["clip_probability"][selected_indices]
    candidates = topk_union_candidates(
        selected_task_probability, selected_clip_probability, top_k=TOP_K
    )

    low_masks = balanced_binary_masks(
        mask_count=MASK_COUNT,
        grid_size=MASK_GRID_SIZE,
        seed=int(cfg.SETTING.SEED),
    )
    mask_tensor = torch.from_numpy(low_masks[:, None]).to(device)
    mask_tensor = F.interpolate(
        mask_tensor, size=(224, 224), mode="bilinear", align_corners=False
    )
    index_to_row = {int(index): row for row, index in enumerate(selected_indices)}
    selected_seen = np.zeros(selected_indices.size, dtype=bool)
    task_masked = np.empty(
        (selected_indices.size, MASK_COUNT, len(class_names)), dtype=np.float32
    )
    clip_masked = np.empty_like(task_masked)
    max_task_replay_error = 0.0
    max_clip_replay_error = 0.0

    # Phase 1b: replay the exact seeded weak-view stream and probe selected rows.
    _restore_rng_state(first_pass_rng)
    for inputs, _ignored_labels, indices in test_loader:
        del _ignored_labels
        batch_indices = indices.long().cpu().numpy()
        positions = [
            position
            for position, index in enumerate(batch_indices)
            if int(index) in index_to_row
        ]
        if not positions:
            continue
        weak_x = inputs[1][positions].to(device)
        rows = np.asarray(
            [index_to_row[int(batch_indices[position])] for position in positions],
            dtype=np.int64,
        )
        with torch.no_grad():
            replay_task_logits = net_c(net_b(net_f(weak_x)))
            replay_clip_features = F.normalize(clip_model.encode_image(weak_x), dim=1)
            replay_clip_logits = (
                clip_model.logit_scale.exp() * replay_clip_features @ text_features.t()
            )
            replay_task_probability = torch.softmax(replay_task_logits, dim=1).cpu().numpy()
            replay_clip_probability = torch.softmax(replay_clip_logits, dim=1).cpu().numpy()
        task_error = float(
            np.max(np.abs(replay_task_probability - selected_task_probability[rows]))
        )
        clip_error = float(
            np.max(np.abs(replay_clip_probability - selected_clip_probability[rows]))
        )
        max_task_replay_error = max(max_task_replay_error, task_error)
        max_clip_replay_error = max(max_clip_replay_error, clip_error)
        if task_error > REPRODUCTION_MAX_ERROR or clip_error > REPRODUCTION_MAX_ERROR:
            raise RuntimeError(
                "Seeded weak-view replay changed the locked predictions: "
                f"task_error={task_error}, clip_error={clip_error}"
            )
        batch_task_masked, batch_clip_masked = _masked_probabilities(
            weak_x, mask_tensor, net_f, net_b, net_c, clip_model, text_features
        )
        task_masked[rows] = batch_task_masked
        clip_masked[rows] = batch_clip_masked
        selected_seen[rows] = True

    if not selected_seen.all():
        missing = selected_indices[~selected_seen]
        raise RuntimeError(f"Selected VisDA row was not replayed: {int(missing[0])}")

    spatial = spatial_consensus_selector(
        task_masked,
        clip_masked,
        candidates,
        low_masks,
        full["clip_prediction"][selected_indices],
    )
    spatial_prediction = spatial["prediction"]

    out_dir = Path(cfg.output_dir) / "spatial_causal_audit"
    out_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_spatial_causal"
    signal_csv = out_dir / f"{stem}_signals.csv"
    signal_npz = out_dir / f"{stem}_signals.npz"
    lock_path = out_dir / f"{stem}_signal_lock.json"
    oracle_csv = out_dir / f"{stem}_oracle_diagnostic.csv"
    class_csv = out_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = out_dir / f"{stem}_summary.json"
    markdown_path = out_dir / f"{stem}_summary.md"

    signal_rows: list[dict[str, Any]] = []
    for row, index in enumerate(selected_indices):
        candidate_labels = [int(value) for value in candidates[row] if value >= 0]
        candidate_names = [class_names[value] for value in candidate_labels]
        scores = spatial["candidate_score"][row]
        score_text = ";".join(
            "invalid" if not np.isfinite(value) else f"{float(value):.9g}"
            for value in scores
        )
        signal_rows.append(
            {
                "index": int(index),
                "path": ordered_paths[int(index)],
                "stratum": str(selected_stratum[row]),
                "task_top1": int(full["task_prediction"][index]),
                "task_top1_name": class_names[int(full["task_prediction"][index])],
                "clip_top1": int(full["clip_prediction"][index]),
                "clip_top1_name": class_names[int(full["clip_prediction"][index])],
                "candidate_labels": ";".join(map(str, candidate_labels)),
                "candidate_names": ";".join(candidate_names),
                "spatial_scores": score_text,
                "spatial_prediction": int(spatial_prediction[row]),
                "spatial_prediction_name": class_names[int(spatial_prediction[row])],
                "has_spatial_choice": bool(spatial["has_spatial_choice"][row]),
                "changed_vs_fixed_clip": bool(
                    spatial_prediction[row] != full["clip_prediction"][index]
                ),
                "split_half_stability": float(
                    spatial["split_half_stability"][row]
                ),
            }
        )
    _write_csv(signal_csv, signal_rows)
    np.savez_compressed(
        signal_npz,
        index=selected_indices,
        stratum=selected_stratum,
        task_probability=selected_task_probability,
        clip_probability=selected_clip_probability,
        candidates=candidates,
        low_resolution_masks=low_masks,
        task_masked_probability=task_masked,
        clip_masked_probability=clip_masked,
        task_support=spatial["task_support"],
        clip_support=spatial["clip_support"],
        candidate_score=spatial["candidate_score"],
        spatial_prediction=spatial_prediction,
        has_spatial_choice=spatial["has_spatial_choice"],
        split_half_stability=spatial["split_half_stability"],
    )
    lock = {
        "phase": "LABEL_FREE_SIGNAL_LOCK",
        "contains_target_labels": False,
        "loader_labels_ignored": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "seed": int(cfg.SETTING.SEED),
        "clip_architecture": str(cfg.ACTIVE.ARCH),
        "source_architecture": str(cfg.MODEL.ARCH),
        "sampling": {
            "task_car_clip_truck": PAIR_SAMPLE_COUNT,
            "other_conflict": OTHER_SAMPLE_COUNT,
            "rule": "lowest SHA256(path) within prediction-defined stratum",
        },
        "mask_contract": {
            "count": MASK_COUNT,
            "grid_size": MASK_GRID_SIZE,
            "two_independent_complement_balanced_halves": True,
            "upsampling": "bilinear align_corners=False",
            "masked_fill_in_normalized_space": 0.0,
        },
        "candidate_contract": {
            "set": "stable deduplicated task/CLIP top-2 union",
            "selection": "maximum task/CLIP positive contrast-map cosine",
            "invalid_fallback": "fixed CLIP top-1",
            "target_label_thresholds": False,
        },
        "weak_view_replay": {
            "max_task_probability_error": max_task_replay_error,
            "max_clip_probability_error": max_clip_replay_error,
            "maximum_allowed_error": REPRODUCTION_MAX_ERROR,
        },
        "signal_csv": {"path": str(signal_csv), "sha256": _sha256(signal_csv)},
        "signal_npz": {"path": str(signal_npz), "sha256": _sha256(signal_npz)},
        "target_list_sha256": _sha256(cfg.t_dset_path),
        "source_checkpoint_sha256": {
            name: _sha256(Path(cfg.output_dir_src) / name)
            for name in ("source_F.pt", "source_B.pt", "source_C.pt")
        },
        "contract_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                REPO_ROOT / "cfgs/visda/plmatch.yaml",
                REPO_ROOT / "src/methods/oh/plmatch.py",
                REPO_ROOT / "clip/model.py",
                REPO_ROOT / "src/utils/spatial_causal_audit.py",
                Path(__file__).resolve(),
            )
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: target labels are revealed only for explicit oracle diagnostics.
    all_label = _parse_labels_after_lock(cfg.t_dset_path, ordered_paths)
    all_mask = np.ones(all_label.size, dtype=bool)
    agreement = full["task_prediction"] == full["clip_prediction"]
    baseline_observed = {
        "total_samples": int(all_label.size),
        "agreement_samples": int(agreement.sum()),
        "agreement_accuracy": _accuracy(
            full["task_prediction"], all_label, agreement
        ),
        "task_accuracy": _accuracy(
            full["task_prediction"], all_label, all_mask
        ),
        "clip_accuracy": _accuracy(
            full["clip_prediction"], all_label, all_mask
        ),
        "arithmetic_mix_accuracy": _accuracy(
            full["arithmetic_prediction"], all_label, all_mask
        ),
        "rms_accuracy": _accuracy(full["rms_prediction"], all_label, all_mask),
    }
    reproduction = _baseline_reproduction(baseline_observed, full["index"])

    conflict_indices = np.flatnonzero(conflict)
    conflict_label = all_label[conflict_indices]
    conflict_candidates = topk_union_candidates(
        full["task_probability"][conflict_indices],
        full["clip_probability"][conflict_indices],
        top_k=TOP_K,
    )
    top1_contains = (
        (full["task_prediction"][conflict_indices] == conflict_label)
        | (full["clip_prediction"][conflict_indices] == conflict_label)
    )
    top2_contains = (conflict_candidates == conflict_label[:, None]).any(axis=1)
    top1_neither = ~top1_contains

    pilot_label = all_label[selected_indices]
    pilot_task = full["task_prediction"][selected_indices]
    pilot_clip = full["clip_prediction"][selected_indices]
    pilot_confidence = np.where(
        full["task_confidence"][selected_indices]
        >= full["clip_confidence"][selected_indices],
        pilot_task,
        pilot_clip,
    )
    pilot_arithmetic = full["arithmetic_prediction"][selected_indices]
    pilot_rms = full["rms_prediction"][selected_indices]
    pair_mask = selected_stratum == "task_car_clip_truck"
    other_mask = ~pair_mask
    pilot_mask = np.ones(selected_indices.size, dtype=bool)
    versus_clip = _paired_comparison(
        spatial_prediction, pilot_clip, pilot_label, pilot_mask
    )
    pair_versus_clip = _paired_comparison(
        spatial_prediction, pilot_clip, pilot_label, pair_mask
    )
    other_versus_clip = _paired_comparison(
        spatial_prediction, pilot_clip, pilot_label, other_mask
    )
    eligible_rescue = (
        (pilot_task != pilot_label)
        & (pilot_clip != pilot_label)
        & (candidates == pilot_label[:, None]).any(axis=1)
    )
    eligible_rescue_rate = _accuracy(
        spatial_prediction, pilot_label, eligible_rescue
    )
    candidate_correct = spatial_prediction == pilot_label
    clip_correct = pilot_clip == pilot_label
    car_mask = pilot_label == car_index
    truck_mask = pilot_label == truck_index
    car_net = int(candidate_correct[car_mask].sum() - clip_correct[car_mask].sum())
    truck_net = int(
        candidate_correct[truck_mask].sum() - clip_correct[truck_mask].sum()
    )
    median_stability = float(np.median(spatial["split_half_stability"]))
    changed_coverage = _pct(np.sum(spatial_prediction != pilot_clip), pilot_clip.size)
    gate = evaluate_spatial_gate(
        reproduction_passed=reproduction["passed"],
        median_split_half_stability=median_stability,
        balanced_gain_pp=versus_clip["gain_pp"],
        balanced_ci=tuple(versus_clip["paired_bootstrap_95_ci_pp"]),
        car_truck_gain_pp=pair_versus_clip["gain_pp"],
        eligible_rescue_rate=eligible_rescue_rate,
        car_net_corrections=car_net,
        truck_net_corrections=truck_net,
        changed_vs_clip_coverage=changed_coverage,
    )

    oracle_rows: list[dict[str, Any]] = []
    for row, index in enumerate(selected_indices):
        label = int(pilot_label[row])
        oracle_rows.append(
            {
                "index": int(index),
                "label": label,
                "label_name": class_names[label],
                "stratum": str(selected_stratum[row]),
                "top1_union_contains_label": bool(
                    label in (int(pilot_task[row]), int(pilot_clip[row]))
                ),
                "top2_union_contains_label": bool(label in candidates[row]),
                "task_correct": bool(pilot_task[row] == label),
                "clip_correct": bool(pilot_clip[row] == label),
                "higher_confidence_correct": bool(pilot_confidence[row] == label),
                "arithmetic_correct": bool(pilot_arithmetic[row] == label),
                "rms_correct": bool(pilot_rms[row] == label),
                "spatial_correct": bool(spatial_prediction[row] == label),
                "eligible_top2_rescue": bool(eligible_rescue[row]),
                "split_half_stability": float(
                    spatial["split_half_stability"][row]
                ),
            }
        )
    _write_csv(oracle_csv, oracle_rows)

    class_rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        mask = pilot_label == class_index
        count = int(mask.sum())
        candidate_count = int(candidate_correct[mask].sum())
        clip_count = int(clip_correct[mask].sum())
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "samples": count,
                "spatial_accuracy": _pct(candidate_count, count),
                "fixed_clip_accuracy": _pct(clip_count, count),
                "delta_pp": _pct(candidate_count - clip_count, count),
                "net_corrections": candidate_count - clip_count,
            }
        )
    _write_csv(class_csv, class_rows)

    summary = {
        "dataset": "VISDA-C",
        "task": "train->validation",
        "seed": int(cfg.SETTING.SEED),
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "optimizer_constructed": False,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "model_parameters_updated": False,
        "training_authorized": False,
        "candidate_contract": lock["candidate_contract"],
        "compute_contract": {
            "pilot_samples": int(selected_indices.size),
            "masked_views_per_sample": MASK_COUNT,
            "frozen_model_image_forwards": int(
                selected_indices.size * MASK_COUNT * 2
            ),
            "full_target_unmasked_model_forwards": int(all_label.size * 2),
            "no_backward": True,
        },
        "baseline_observed": baseline_observed,
        "baseline_reproduction": reproduction,
        "full_conflict_oracle_diagnostic": {
            "samples": int(conflict_indices.size),
            "top1_union_coverage": _pct(top1_contains.sum(), top1_contains.size),
            "top2_union_coverage": _pct(top2_contains.sum(), top2_contains.size),
            "top1_neither_samples": int(top1_neither.sum()),
            "top2_coverage_within_top1_neither": _pct(
                top2_contains[top1_neither].sum(), top1_neither.sum()
            ),
        },
        "pilot_oracle_diagnostic": {
            "samples": int(pilot_label.size),
            "strata": {
                "task_car_clip_truck": int(pair_mask.sum()),
                "other_conflict": int(other_mask.sum()),
            },
            "fixed_task_accuracy": _accuracy(
                pilot_task, pilot_label, pilot_mask
            ),
            "fixed_clip_accuracy": _accuracy(
                pilot_clip, pilot_label, pilot_mask
            ),
            "higher_confidence_accuracy": _accuracy(
                pilot_confidence, pilot_label, pilot_mask
            ),
            "arithmetic_accuracy": _accuracy(
                pilot_arithmetic, pilot_label, pilot_mask
            ),
            "rms_accuracy": _accuracy(pilot_rms, pilot_label, pilot_mask),
            "spatial_accuracy": _accuracy(
                spatial_prediction, pilot_label, pilot_mask
            ),
            "versus_fixed_clip": versus_clip,
            "car_truck_versus_clip": pair_versus_clip,
            "other_conflict_versus_clip": other_versus_clip,
            "eligible_top2_rescue_samples": int(eligible_rescue.sum()),
            "eligible_top2_rescue_rate": eligible_rescue_rate,
            "median_split_half_stability": median_stability,
            "changed_vs_fixed_clip_coverage": changed_coverage,
            "car_net_corrections": car_net,
            "truck_net_corrections": truck_net,
        },
        "classwise_oracle_diagnostic": class_rows,
        "gate": gate,
        "decision": gate["decision"],
        "next": (
            "request approval for one full offline spatial audit; do not train"
            if gate["decision"] == "PASS_OFFLINE_GATE"
            else "stop; do not run proxy or full VisDA training"
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)

    print(f"Wrote label-free signals: {signal_csv}")
    print(f"Wrote label-free tensor signals: {signal_npz}")
    print(f"Locked signals before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_csv}")
    print(f"Wrote classwise oracle diagnostic: {class_csv}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))


if __name__ == "__main__":
    main()
