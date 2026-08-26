"""Keep DAC's sample history active during cyclic Task/CLIP refinement.

Released DUET replaces the DAC teacher with the current CLIP distribution and
accumulates every past Task/CLIP agreement as a hard label.  This module makes
the opposite state transition on current conflicts: the existing DAC memory
is retained, and it may move again only after Task and CLIP reach agreement.
Target labels are never accepted by this module.
"""

from __future__ import annotations

import math

import torch

from src.utils.duet_delayed_credit import update_delayed_credit


_REQUIRED_STATE_KEYS = {
    "memory",
    "previous_task",
    "previous_clip",
    "task_loss_sum",
    "clip_loss_sum",
    "feedback_mass",
    "task_weight",
    "clip_weight",
}


def validate_credit_state(
    state: dict[str, torch.Tensor],
    *,
    sample_count: int,
    class_count: int,
) -> None:
    """Validate the row/class contract shared by DAC and the refinement run."""
    missing = sorted(_REQUIRED_STATE_KEYS.difference(state))
    if missing:
        raise ValueError("DAC credit state is missing keys: {}".format(missing))
    expected_probability_shape = (int(sample_count), int(class_count))
    for key in ("memory", "previous_task", "previous_clip"):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError("DAC credit state {} must be a tensor".format(key))
        if tuple(value.shape) != expected_probability_shape:
            raise ValueError(
                "DAC credit state {} has shape {}, expected {}".format(
                    key,
                    tuple(value.shape),
                    expected_probability_shape,
                )
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("DAC credit state {} contains non-finite values".format(key))
    expected_vector_shape = (int(sample_count),)
    for key in (
        "task_loss_sum",
        "clip_loss_sum",
        "feedback_mass",
        "task_weight",
        "clip_weight",
    ):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError("DAC credit state {} must be a tensor".format(key))
        if tuple(value.shape) != expected_vector_shape:
            raise ValueError(
                "DAC credit state {} has shape {}, expected {}".format(
                    key,
                    tuple(value.shape),
                    expected_vector_shape,
                )
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("DAC credit state {} contains non-finite values".format(key))


def _normalize(probability: torch.Tensor, epsilon: float) -> torch.Tensor:
    probability = probability.float().cpu().clamp_min(epsilon)
    return probability / probability.sum(dim=1, keepdim=True)


@torch.no_grad()
def credit_preserving_refinement_step(
    state: dict[str, torch.Tensor],
    task_probability: torch.Tensor,
    clip_probability: torch.Tensor,
    *,
    conflict_hard_fraction: float = 0.8,
    soft_replacement_mode: str = "all_conflicts",
    memory_write_mode: str = "locked",
    decay: float = 0.9,
    credit_eta: float = 4.0,
    memory_update_rate: float = 0.5,
    credit_mode: str = "delayed",
    feedback_mode: str = "agreement_temporal",
    epsilon: float = 1e-8,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build one GT-free refinement teacher and its current hard-label set.

    The full method lets agreement rows accept the delayed-credit update and
    locks conflict rows to their previous full-distribution memory.  Explicit
    modes expose one-factor memory-write and residual-target ablations without
    changing coverage or training length.
    """
    if not 0.0 <= conflict_hard_fraction <= 1.0:
        raise ValueError("conflict_hard_fraction must be in [0, 1]")
    if soft_replacement_mode not in {
        "none",
        "all_conflicts",
        "task_supported",
    }:
        raise ValueError(
            "soft_replacement_mode must be none, all_conflicts, "
            "or task_supported"
        )
    if memory_write_mode not in {"locked", "writable", "frozen"}:
        raise ValueError(
            "memory_write_mode must be locked, writable, or frozen"
        )
    task_probability = _normalize(task_probability, epsilon)
    clip_probability = _normalize(clip_probability, epsilon)
    if task_probability.shape != clip_probability.shape:
        raise ValueError("Task and CLIP probabilities must have the same shape")
    validate_credit_state(
        state,
        sample_count=int(task_probability.shape[0]),
        class_count=int(task_probability.shape[1]),
    )

    memory_before = _normalize(state["memory"], epsilon)
    proposed_state, diagnostics = update_delayed_credit(
        state,
        task_probability,
        clip_probability,
        decay=decay,
        credit_eta=credit_eta,
        memory_update_rate=memory_update_rate,
        credit_mode=credit_mode,
        feedback_mode=feedback_mode,
        epsilon=epsilon,
    )
    task_label = task_probability.argmax(dim=1)
    clip_label = clip_probability.argmax(dim=1)
    agreement = task_label == clip_label
    conflict = ~agreement

    # ``locked`` is the full CLM transition.  The other two modes are explicit
    # ablations; they must never be selected by target-label performance.
    if memory_write_mode == "writable":
        memory = proposed_state["memory"]
        memory_preserved = torch.zeros_like(conflict)
    elif memory_write_mode == "frozen":
        memory = memory_before
        memory_preserved = torch.ones_like(conflict)
    else:
        memory = torch.where(
            agreement.unsqueeze(1),
            proposed_state["memory"],
            memory_before,
        )
        memory_preserved = conflict
    memory = _normalize(memory, epsilon)
    proposed_state["memory"] = memory

    row = torch.arange(memory.shape[0])
    memory_task = memory[row, task_label]
    memory_clip = memory[row, clip_label]
    choose_task = memory_task >= memory_clip
    hard_label = torch.where(choose_task, task_label, clip_label)
    pair_log_odds = (
        memory_task.clamp_min(epsilon).log()
        - memory_clip.clamp_min(epsilon).log()
    ).abs()

    conflict_indices = torch.nonzero(conflict, as_tuple=False).flatten()
    selected = torch.zeros_like(conflict)
    requested = int(
        math.ceil(float(conflict_indices.numel()) * conflict_hard_fraction)
    )
    if requested > 0:
        order = torch.argsort(
            pair_log_odds[conflict_indices],
            descending=True,
            stable=True,
        )
        selected[conflict_indices[order[:requested]]] = True

    if soft_replacement_mode == "none":
        soft_replaced = torch.zeros_like(conflict)
    elif soft_replacement_mode == "task_supported":
        soft_replaced = conflict & choose_task
    else:
        soft_replaced = conflict

    # Agreements keep the current CLIP target.  The residual mode also keeps
    # CLIP whenever DAC history supports CLIP; only the opposite historical
    # direction receives an anti-erasure correction.
    soft_target = clip_probability.clone()
    soft_target[soft_replaced] = memory[soft_replaced]

    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "agreement_mask": agreement,
            "conflict_mask": conflict,
            "hard_selected": selected,
            "hard_label": hard_label,
            "pair_log_odds": pair_log_odds,
            "soft_target": soft_target,
            "soft_replaced": soft_replaced,
            "memory_preserved": memory_preserved,
            "memory_shift_l1": (memory - memory_before).abs().sum(dim=1),
        }
    )
    return proposed_state, diagnostics
