"""Pure helpers for the frozen-model VisDA spatial-causal audit.

The helpers in this module never access target labels.  They build a fixed
balanced occlusion bank, form a task/CLIP top-k candidate union, and measure
whether the two frozen models assign class-contrastive support to the same
image regions.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def balanced_binary_masks(
    *, mask_count: int = 64, grid_size: int = 7, seed: int = 2_020
) -> np.ndarray:
    """Return two independent, exactly balanced mask halves.

    Each half contains random masks and their complements, so every spatial
    cell is visible in exactly half of the probes.  This reduces Monte Carlo
    bias while preserving an independent split-half stability check.
    """
    if mask_count <= 0 or mask_count % 4:
        raise ValueError("mask_count must be a positive multiple of four")
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than one")

    rng = np.random.default_rng(seed)
    halves = []
    base_count = mask_count // 4
    for _ in range(2):
        base = rng.integers(
            0, 2, size=(base_count, grid_size, grid_size), dtype=np.int8
        )
        half = np.concatenate((base, 1 - base), axis=0)
        half = half[rng.permutation(half.shape[0])]
        halves.append(half)
    return np.concatenate(halves, axis=0).astype(np.float32)


def deterministic_hash_sample(
    paths: list[str], eligible: np.ndarray, *, count: int, namespace: str
) -> np.ndarray:
    """Select the lowest SHA256 priorities without labels or RNG state."""
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.ndim != 1 or eligible.size != len(paths):
        raise ValueError("eligible must align with paths")
    if count <= 0:
        raise ValueError("count must be positive")
    indices = np.flatnonzero(eligible)
    if indices.size < count:
        raise ValueError(
            f"Requested {count} rows from {namespace}, found {indices.size}"
        )

    def priority(index: int) -> tuple[bytes, str, int]:
        path = str(paths[index])
        digest = hashlib.sha256(
            f"{namespace}\0{path}".encode("utf-8")
        ).digest()
        return digest, path, int(index)

    selected = sorted((int(index) for index in indices), key=priority)[:count]
    return np.asarray(selected, dtype=np.int64)


def topk_union_candidates(
    task_probability: np.ndarray,
    clip_probability: np.ndarray,
    *,
    top_k: int = 2,
) -> np.ndarray:
    """Return a stable, deduplicated task/CLIP top-k union per sample."""
    task_probability = np.asarray(task_probability, dtype=np.float64)
    clip_probability = np.asarray(clip_probability, dtype=np.float64)
    if (
        task_probability.ndim != 2
        or task_probability.shape != clip_probability.shape
        or task_probability.shape[0] == 0
    ):
        raise ValueError("probability arrays must be non-empty and same-shaped")
    if not (
        np.isfinite(task_probability).all()
        and np.isfinite(clip_probability).all()
    ):
        raise ValueError("probability arrays must be finite")
    class_count = task_probability.shape[1]
    if not 1 <= top_k <= class_count:
        raise ValueError("top_k is outside the class range")

    task_order = np.argsort(-task_probability, axis=1, kind="mergesort")[:, :top_k]
    clip_order = np.argsort(-clip_probability, axis=1, kind="mergesort")[:, :top_k]
    result = np.full((task_probability.shape[0], 2 * top_k), -1, dtype=np.int64)
    for row in range(result.shape[0]):
        write = 0
        for candidate in np.concatenate((task_order[row], clip_order[row])):
            candidate = int(candidate)
            if candidate not in result[row, :write]:
                result[row, write] = candidate
                write += 1
    return result


def contrastive_support_maps(
    masked_probability: np.ndarray,
    candidates: np.ndarray,
    low_resolution_masks: np.ndarray,
    *,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    """Estimate positive class-contrastive support on the low-res mask grid.

    A spatial cell receives positive support when making it visible increases a
    candidate's probability relative to the other candidates in the locked
    union.  Centering the masks removes the generic image-level response.
    """
    probability = np.asarray(masked_probability, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.int64)
    masks = np.asarray(low_resolution_masks, dtype=np.float64)
    if probability.ndim != 3:
        raise ValueError("masked_probability must be [sample, mask, class]")
    if candidates.ndim != 2 or candidates.shape[0] != probability.shape[0]:
        raise ValueError("candidates must align with masked_probability")
    if masks.ndim != 3 or masks.shape[0] != probability.shape[1]:
        raise ValueError("masks must align with the masked probability axis")
    if not np.isfinite(probability).all() or not np.isfinite(masks).all():
        raise ValueError("probabilities and masks must be finite")

    stop = probability.shape[1] if stop is None else int(stop)
    if not 0 <= start < stop <= probability.shape[1]:
        raise ValueError("invalid mask slice")
    probability = probability[:, start:stop]
    mask_flat = masks[start:stop].reshape(stop - start, -1)
    centered_mask = mask_flat - mask_flat.mean(axis=0, keepdims=True)

    valid = candidates >= 0
    valid_count = valid.sum(axis=1)
    if np.any(valid_count < 2):
        raise ValueError("each sample requires at least two unique candidates")
    safe_candidates = np.maximum(candidates, 0)
    gathered = np.take_along_axis(
        probability,
        np.broadcast_to(
            safe_candidates[:, None, :],
            (probability.shape[0], probability.shape[1], candidates.shape[1]),
        ),
        axis=2,
    )
    gathered = np.where(valid[:, None, :], gathered, 0.0)
    total = gathered.sum(axis=2, keepdims=True)
    competitor_mean = (total - gathered) / (valid_count[:, None, None] - 1)
    contrast = np.where(valid[:, None, :], gathered - competitor_mean, 0.0)
    support = np.einsum("nmk,mh->nkh", contrast, centered_mask, optimize=True)
    support /= float(stop - start)
    support = np.maximum(support, 0.0)
    support[~valid] = 0.0
    return support.reshape(
        probability.shape[0], candidates.shape[1], masks.shape[1], masks.shape[2]
    )


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity over every row, returning zero for a zero map."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("cosine inputs must be same-shaped with a row axis")
    left_flat = left.reshape(-1, int(np.prod(left.shape[1:])))
    right_flat = right.reshape(-1, int(np.prod(right.shape[1:])))
    numerator = np.einsum("ij,ij->i", left_flat, right_flat)
    denominator = np.linalg.norm(left_flat, axis=1) * np.linalg.norm(
        right_flat, axis=1
    )
    similarity = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    return similarity.reshape(left.shape[:-1] if left.ndim == 2 else left.shape[:-2])


def spatial_consensus_selector(
    task_masked_probability: np.ndarray,
    clip_masked_probability: np.ndarray,
    candidates: np.ndarray,
    low_resolution_masks: np.ndarray,
    clip_fallback: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select the candidate with strongest task/CLIP spatial agreement."""
    if task_masked_probability.shape != clip_masked_probability.shape:
        raise ValueError("task and CLIP masked probabilities must match")
    clip_fallback = np.asarray(clip_fallback, dtype=np.int64)
    if clip_fallback.shape != (task_masked_probability.shape[0],):
        raise ValueError("clip_fallback must contain one label per sample")

    task_support = contrastive_support_maps(
        task_masked_probability, candidates, low_resolution_masks
    )
    clip_support = contrastive_support_maps(
        clip_masked_probability, candidates, low_resolution_masks
    )
    sample_count, slot_count = candidates.shape
    scores = cosine_rows(
        task_support.reshape(sample_count * slot_count, -1),
        clip_support.reshape(sample_count * slot_count, -1),
    ).reshape(sample_count, slot_count)
    valid = candidates >= 0
    task_norm = np.linalg.norm(task_support.reshape(sample_count, slot_count, -1), axis=2)
    clip_norm = np.linalg.norm(clip_support.reshape(sample_count, slot_count, -1), axis=2)
    score_valid = valid & (task_norm > 1e-12) & (clip_norm > 1e-12)
    scores = np.where(score_valid, scores, -np.inf)
    best_slot = np.argmax(scores, axis=1)
    has_spatial_choice = score_valid.any(axis=1)
    prediction = candidates[np.arange(sample_count), best_slot]
    prediction = np.where(has_spatial_choice, prediction, clip_fallback)

    half = low_resolution_masks.shape[0] // 2
    task_first = contrastive_support_maps(
        task_masked_probability, candidates, low_resolution_masks, start=0, stop=half
    )
    task_second = contrastive_support_maps(
        task_masked_probability,
        candidates,
        low_resolution_masks,
        start=half,
        stop=low_resolution_masks.shape[0],
    )
    clip_first = contrastive_support_maps(
        clip_masked_probability, candidates, low_resolution_masks, start=0, stop=half
    )
    clip_second = contrastive_support_maps(
        clip_masked_probability,
        candidates,
        low_resolution_masks,
        start=half,
        stop=low_resolution_masks.shape[0],
    )
    row = np.arange(sample_count)
    task_stability = cosine_rows(task_first[row, best_slot], task_second[row, best_slot])
    clip_stability = cosine_rows(clip_first[row, best_slot], clip_second[row, best_slot])
    stability = 0.5 * (task_stability + clip_stability)
    stability = np.where(has_spatial_choice, stability, 0.0)

    return {
        "prediction": prediction.astype(np.int64),
        "candidate_score": scores,
        "has_spatial_choice": has_spatial_choice,
        "split_half_stability": stability,
        "task_support": task_support.astype(np.float32),
        "clip_support": clip_support.astype(np.float32),
    }


def evaluate_spatial_gate(
    *,
    reproduction_passed: bool,
    median_split_half_stability: float,
    balanced_gain_pp: float,
    balanced_ci: tuple[float, float],
    car_truck_gain_pp: float,
    eligible_rescue_rate: float,
    car_net_corrections: int,
    truck_net_corrections: int,
    changed_vs_clip_coverage: float,
    min_stability: float = 0.80,
    min_car_truck_gain_pp: float = 3.0,
    min_rescue_rate: float = 20.0,
) -> dict[str, Any]:
    """Apply the locked oracle gate without fitting a label threshold."""
    checks = {
        "baseline_reproduced": bool(reproduction_passed),
        "median_split_half_stability_at_least_0.80": (
            median_split_half_stability >= min_stability
        ),
        "balanced_gain_positive": balanced_gain_pp > 0.0,
        "balanced_gain_ci_lower_positive": balanced_ci[0] > 0.0,
        "car_truck_gain_at_least_3pp": (
            car_truck_gain_pp >= min_car_truck_gain_pp
        ),
        "eligible_top2_rescue_at_least_20pct": (
            eligible_rescue_rate >= min_rescue_rate
        ),
        "car_net_corrections_nonnegative": car_net_corrections >= 0,
        "truck_net_corrections_nonnegative": truck_net_corrections >= 0,
        "selector_changes_fixed_clip": changed_vs_clip_coverage > 0.0,
    }
    return {
        "decision": "PASS_OFFLINE_GATE" if all(checks.values()) else "REJECT",
        "thresholds": {
            "min_median_split_half_stability": min_stability,
            "balanced_gain_must_be_positive": True,
            "balanced_ci_lower_must_be_positive": True,
            "min_car_truck_gain_pp": min_car_truck_gain_pp,
            "min_eligible_top2_rescue_rate": min_rescue_rate,
            "car_and_truck_net_corrections_must_be_nonnegative": True,
            "selector_must_change_fixed_clip": True,
        },
        "checks": checks,
    }
