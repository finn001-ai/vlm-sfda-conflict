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
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized swap selection over a full batch.

    Returns ``(labels, selected)`` with one entry per sample:
      - ``labels[i]`` is the chosen class id (A or B) for selected samples and
        ``-1`` otherwise;
      - ``selected[i]`` is True for samples that receive a hard pseudo label,
        i.e. every swap conflict in cycle 0 and gate-passing swap conflicts in
        later cycles.  Abstained swap conflicts stay unselected so callers can
        keep them out of the training loss.
    """
    task = _as_float_array(task_probability)
    evidence = swap_evidence(task, clip_probability)
    swap = evidence["swap_mask"]
    A = evidence["A"]
    B = evidence["B"]

    labels = np.full(task.shape[0], -1, dtype=np.int64)
    selected = np.zeros(task.shape[0], dtype=bool)
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
