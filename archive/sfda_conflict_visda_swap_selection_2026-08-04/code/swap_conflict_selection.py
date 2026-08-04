"""Label-free hard-label selection for pure-swap (bidirectional) conflicts.

This module adapts the archived analysis rule
``archive/sfda_conflict_visda_topk_swap_analysis_2026-08-04/code/
analyze_topk_swap_selection.py`` so it can be reused as a training-path
decision function.  The math is kept identical to the archived script
(``log(max(x, EPS))`` differences and a decision-strength gate on
``|log(eA) - log(eB)|``); nothing is rewritten.

Scope: only ``bidirectional_cross_support`` (pure-swap) conflicts enter the
rule.  For such a sample the two views point at each other's top-1:

    A = task top1 = clip top2
    B = clip top1 = task top2
    A != B

Rule:

    cycle 0:          always pick B (CLIP top1), no gate.
    cycle >= 1:       eA = pA * qA, eB = pB * qB
                      pick A if log(eA) - log(eB) >= D
                      pick B if log(eB) - log(eA) >= D
                      otherwise abstain (no pseudo label)

where pA/pB are the task top1/top2 probabilities and qA/qB are the CLIP
top2/top1 softmax scores.  Ground truth is never read here; it is used only
by callers for evaluation (oracle-diagnostic principle).
"""

from __future__ import annotations

import numpy as np


EPS = 1e-9
DEFAULT_GATE_D = 4.0

# Offline-locked cycle-0 direction accuracy table: for each swap orientation
# (A=task top1, B=clip top1), the probability that the CLIP top1 (B) is the
# ground-truth label.  Computed once from the archived cycle-0 conflict CSV
# (task_TV_seed_2020, pre-training CLIP + source model) and frozen here so the
# training path never needs ground truth.  This is a static CLIP reliability
# property of each orientation, not a per-run oracle.  Orientations with fewer
# samples have noisy values but tiny impact; thresholding at 0.8 mostly keeps
# motorcycle->bicycle / bus<->train / truck->car and drops car->truck,
# car->bus and car->motorcycle, which are the orientations that previously
# taught the model to mislabel cars as trucks/buses/motorcycles.
CYCLE0_DIRECTION_ACCURACY = {
    (3, 11): 0.6916,  # n=509
    (6, 1): 0.9016,  # n=488
    (3, 2): 0.7232,  # n=177
    (10, 2): 0.8160,  # n=125
    (3, 6): 0.4636,  # n=110
    (2, 10): 0.9583,  # n=72
    (6, 4): 0.8776,  # n=49
    (6, 3): 0.5909,  # n=44
    (11, 3): 0.8636,  # n=44
    (3, 10): 0.9000,  # n=30
    (3, 9): 0.6207,  # n=29
    (3, 0): 0.9167,  # n=24
    (0, 9): 0.8824,  # n=17
    (2, 11): 0.8824,  # n=17
    (1, 6): 0.6923,  # n=13
    (6, 9): 0.7692,  # n=13
    (2, 3): 0.7500,  # n=12
    (3, 4): 0.0833,  # n=12
    (9, 5): 1.0000,  # n=10
    (10, 0): 0.7000,  # n=10
    (2, 0): 1.0000,  # n=9
    (10, 11): 0.7778,  # n=9
    (0, 4): 1.0000,  # n=7
    (3, 1): 0.0000,  # n=7
    (3, 8): 0.8571,  # n=7
    (0, 8): 1.0000,  # n=6
    (6, 0): 1.0000,  # n=6
    (7, 9): 0.3333,  # n=6
    (0, 10): 1.0000,  # n=5
    (6, 7): 1.0000,  # n=5
    (8, 4): 0.6000,  # n=5
    (10, 8): 1.0000,  # n=5
    (8, 3): 0.7500,  # n=4
    (10, 3): 0.7500,  # n=4
    (10, 4): 1.0000,  # n=4
    (3, 7): 0.6667,  # n=3
    (8, 1): 0.3333,  # n=3
    (8, 7): 0.0000,  # n=3
    (8, 9): 0.0000,  # n=3
    (9, 3): 0.6667,  # n=3
    (11, 0): 1.0000,  # n=3
    (11, 2): 0.3333,  # n=3
    (11, 7): 0.6667,  # n=3
    (1, 8): 1.0000,  # n=2
    (1, 9): 1.0000,  # n=2
    (4, 10): 0.5000,  # n=2
    (5, 9): 1.0000,  # n=2
    (6, 8): 1.0000,  # n=2
    (9, 0): 0.5000,  # n=2
    (9, 7): 1.0000,  # n=2
    (9, 8): 1.0000,  # n=2
    (10, 1): 1.0000,  # n=2
    (0, 3): 0.0000,  # n=1
    (4, 3): 1.0000,  # n=1
    (4, 6): 1.0000,  # n=1
    (4, 8): 1.0000,  # n=1
    (5, 3): 0.0000,  # n=1
    (5, 7): 1.0000,  # n=1
    (6, 2): 1.0000,  # n=1
    (6, 11): 0.0000,  # n=1
    (7, 4): 1.0000,  # n=1
    (7, 10): 1.0000,  # n=1
    (9, 1): 1.0000,  # n=1
    (9, 6): 1.0000,  # n=1
    (11, 10): 1.0000,  # n=1
}


def _as_float_array(value: object) -> np.ndarray:
    """Convert tensors/arrays to a float64 numpy array (detached, CPU)."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError("probability must have shape [sample, class>=2]")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("probability must be finite and non-negative")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("probability rows must sum to one")
    return array


def swap_evidence(
    task_probability: object,
    clip_probability: object,
) -> dict[str, np.ndarray]:
    """Detect pure-swap conflicts and extract their evidence vectors.

    Returns full-length arrays ``A``, ``B``, ``pA``, ``pB``, ``qA``, ``qB``
    plus a boolean ``swap_mask`` marking bidirectional_cross_support conflicts
    (task top1 = A, task top2 = B, clip top1 = B, clip top2 = A with A != B).
    Non-swap rows have undefined evidence values and ``swap_mask=False``.
    """
    task = _as_float_array(task_probability)
    clip = _as_float_array(clip_probability)
    if task.shape != clip.shape:
        raise ValueError("task and CLIP probability shapes must match")
    task_rank = np.argsort(-task, axis=1, kind="stable")[:, :2]
    clip_rank = np.argsort(-clip, axis=1, kind="stable")[:, :2]
    task_top1 = task_rank[:, 0]
    task_top2 = task_rank[:, 1]
    clip_top1 = clip_rank[:, 0]
    clip_top2 = clip_rank[:, 1]
    swap = (
        (task_top1 != clip_top1)
        & (task_top2 == clip_top1)
        & (clip_top2 == task_top1)
    )
    rows = np.arange(task.shape[0])
    return {
        "swap_mask": swap,
        "A": task_top1,
        "B": clip_top1,
        "pA": task[rows, task_top1],
        "pB": task[rows, task_top2],
        "qA": clip[rows, clip_top2],
        "qB": clip[rows, clip_top1],
    }


def decide_swap_evidence(
    pA: object,
    pB: object,
    qA: object,
    qB: object,
    *,
    cycle: int,
    gate_D: float = DEFAULT_GATE_D,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Decision layer: same math as the archived selection script.

    Inputs are the per-sample evidence columns (pA/pB = task top1/top2
    probabilities, qA/qB = CLIP top2/top1 scores) exactly as stored in the
    archived ``conflict_samples.csv``.  Returns ``(prefer_A, decided)``:

      - ``prefer_A[i]`` is True when A should be chosen, False for B;
      - ``decided[i]`` is False for abstained samples (gate not reached).

    Cycle 0 always decides B (prefer_A=False, decided=True) with no gate.
    Ties in later cycles fall back to B, matching the archived script.
    """
    pA = np.asarray(pA, dtype=np.float64)
    pB = np.asarray(pB, dtype=np.float64)
    qA = np.asarray(qA, dtype=np.float64)
    qB = np.asarray(qB, dtype=np.float64)
    if not (pA.ndim == pB.ndim == qA.ndim == qB.ndim == 1):
        raise ValueError("evidence columns must be 1-D arrays")
    if not (pA.shape == pB.shape == qA.shape == qB.shape):
        raise ValueError("evidence columns must have matching lengths")
    if cycle < 0:
        raise ValueError("cycle must be non-negative")
    if gate_D < 0.0:
        raise ValueError("gate_D must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if np.any(pA < 0.0) or np.any(pB < 0.0) or np.any(qA < 0.0) or np.any(qB < 0.0):
        raise ValueError("evidence values must be non-negative")

    if cycle == 0:
        return np.zeros(pA.shape, dtype=bool), np.ones(pA.shape, dtype=bool)

    # log(eA) - log(eB) = log(pA) - log(pB) - (log(qB) - log(qA))
    log_ratio_task = np.log(np.maximum(pA, eps)) - np.log(np.maximum(pB, eps))
    log_ratio_clip = np.log(np.maximum(qB, eps)) - np.log(np.maximum(qA, eps))
    diff = log_ratio_task - log_ratio_clip
    decided = np.abs(diff) >= gate_D
    prefer_A = diff > 0.0
    return prefer_A, decided


def select_swap_labels(
    task_probability: object,
    clip_probability: object,
    *,
    cycle: int,
    gate_D: float = DEFAULT_GATE_D,
    eps: float = EPS,
    min_direction_accuracy: float = 0.0,
    last_active_cycle: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized swap selection over a full batch.

    Returns ``(labels, selected)`` with one entry per sample:
      - ``labels[i]`` is the chosen class id (A or B) for selected samples and
        ``-1`` otherwise;
      - ``selected[i]`` is True for samples that receive a hard pseudo label,
        i.e. every swap conflict in cycle 0 and gate-passing swap conflicts in
        later cycles.  Abstained swap conflicts stay unselected so callers can
        keep them out of the training loss.

    ``min_direction_accuracy > 0`` additionally abstains any orientation whose
    offline-locked cycle-0 CLIP accuracy (``CYCLE0_DIRECTION_ACCURACY``) is
    below the threshold.  This protects hard classes such as car/truck, where
    blindly following CLIP is only ~69% reliable and was shown to drag final
    car accuracy down.  The threshold applies to every cycle, including the
    cycle-0 special case.

    ``last_active_cycle`` (1-based) stops producing new labels from that cycle
    onward.  Late-cycle swap labels are much less reliable (~60-65% precision
    in cycles 7-8), cover mostly samples that were already labeled in earlier
    cycles, and add a net of only ~+85 correct labels while injecting ~148
    wrong ones; stopping early avoids that pollution without meaningful loss.
    """
    task = _as_float_array(task_probability)
    evidence = swap_evidence(task, clip_probability)
    swap = evidence["swap_mask"]
    A = evidence["A"]
    B = evidence["B"]
    if not 0.0 <= min_direction_accuracy <= 1.0:
        raise ValueError("min_direction_accuracy must be in [0, 1]")
    if last_active_cycle < 1:
        raise ValueError("last_active_cycle must be >= 1")

    labels = np.full(task.shape[0], -1, dtype=np.int64)
    selected = np.zeros(task.shape[0], dtype=bool)
    if cycle + 1 > last_active_cycle:
        return labels, selected
    rows = np.flatnonzero(swap)
    if rows.size == 0:
        return labels, selected

    prefer_A, decided = decide_swap_evidence(
        evidence["pA"][rows],
        evidence["pB"][rows],
        evidence["qA"][rows],
        evidence["qB"][rows],
        cycle=cycle,
        gate_D=gate_D,
        eps=eps,
    )
    chosen = np.where(prefer_A, A[rows], B[rows])
    if min_direction_accuracy > 0.0:
        direction_accuracy = np.asarray(
            [
                CYCLE0_DIRECTION_ACCURACY.get(
                    (int(A[row]), int(B[row])), 0.0
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        decided &= direction_accuracy >= min_direction_accuracy
    labels[rows[decided]] = chosen[decided]
    selected[rows[decided]] = True
    return labels, selected


def summarize_swap_decisions(
    task_probability: object,
    clip_probability: object,
    labels_gt: object,
    *,
    cycle: int,
    gate_D: float = DEFAULT_GATE_D,
    eps: float = EPS,
    min_direction_accuracy: float = 0.0,
    last_active_cycle: int = 8,
) -> dict[str, int | float]:
    """Oracle-diagnostic summary: GT is used only to evaluate fixed decisions."""
    task = _as_float_array(task_probability)
    gt = np.asarray(labels_gt, dtype=np.int64)
    if gt.shape != (task.shape[0],):
        raise ValueError("labels_gt must contain one value per sample")
    swap_labels, selected = select_swap_labels(
        task_probability,
        clip_probability,
        cycle=cycle,
        gate_D=gate_D,
        eps=eps,
        min_direction_accuracy=min_direction_accuracy,
        last_active_cycle=last_active_cycle,
    )
    swap_count = int(selected.sum())
    swap_conflicts = int(swap_evidence(task_probability, clip_probability)["swap_mask"].sum())
    correct = int((selected & (swap_labels == gt)).sum() if swap_count else 0)
    return {
        "cycle": int(cycle),
        "gate_D": float(gate_D),
        "swap_conflicts": swap_conflicts,
        "decisions": swap_count,
        "abstain": swap_conflicts - swap_count,
        "correct": correct,
        "precision_pct": (
            100.0 * correct / swap_count if swap_count else 0.0
        ),
    }
