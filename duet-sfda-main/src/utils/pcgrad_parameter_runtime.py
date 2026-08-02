"""Resident, zero-update GPU runtime for the VisDA PCGrad preflight."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.pcgrad_parameter_audit import (
    AUDIT_BATCH_COUNT,
    AUDIT_BATCH_SIZE,
    AUDITED_CONFLICTS,
    CONFLICTS_PER_BATCH,
    build_locked_parameter_audit_batches,
    symmetric_pcgrad_output_correction,
)


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
GROUP_INDICES = {
    "car": (3,),
    "person": (7,),
    "truck": (11,),
    "other_nine": (0, 1, 2, 4, 5, 6, 8, 9, 10),
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _model_sha256(models: tuple[torch.nn.Module, ...]) -> str:
    digest = hashlib.sha256()
    for model_index, model in enumerate(models):
        for name, value in model.state_dict().items():
            tensor = value.detach().contiguous().cpu()
            digest.update(f"{model_index}:{name}".encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _buffer_snapshot(models: tuple[torch.nn.Module, ...]) -> list[dict[str, torch.Tensor]]:
    return [
        {name: value.detach().clone() for name, value in model.named_buffers()}
        for model in models
    ]


def _restore_buffers(
    models: tuple[torch.nn.Module, ...],
    snapshots: list[dict[str, torch.Tensor]],
) -> None:
    with torch.no_grad():
        for model, snapshot in zip(models, snapshots):
            current = dict(model.named_buffers())
            if set(current) != set(snapshot):
                raise RuntimeError("Model buffer contract changed during audit")
            for name, value in snapshot.items():
                current[name].copy_(value)


def _gradient_dot(
    first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return sum(
        (left * right).sum() for left, right in zip(first, second)
    )


def _gradient_norm(gradient: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.sqrt(sum(value.square().sum() for value in gradient))


def _alignment_metrics(
    baseline: tuple[torch.Tensor, ...],
    correction: tuple[torch.Tensor, ...],
    oracle: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    baseline_dot = _gradient_dot(baseline, oracle)
    correction_dot = _gradient_dot(correction, oracle)
    candidate_dot = baseline_dot + correction_dot
    baseline_norm = _gradient_norm(baseline)
    correction_norm = _gradient_norm(correction)
    candidate_norm = torch.sqrt(
        baseline_norm.square()
        + correction_norm.square()
        + 2.0 * _gradient_dot(baseline, correction)
    ).clamp_min(0.0)
    oracle_norm = _gradient_norm(oracle)
    epsilon = torch.finfo(baseline_norm.dtype).eps
    return {
        "baseline_first_order": float(baseline_dot.detach().cpu()),
        "candidate_first_order": float(candidate_dot.detach().cpu()),
        "candidate_minus_baseline_first_order": float(
            correction_dot.detach().cpu()
        ),
        "baseline_oracle_unit_projection": float(
            (baseline_dot / oracle_norm.clamp_min(epsilon)).detach().cpu()
        ),
        "candidate_oracle_unit_projection": float(
            (candidate_dot / oracle_norm.clamp_min(epsilon)).detach().cpu()
        ),
        "baseline_cosine": float(
            (
                baseline_dot
                / (baseline_norm * oracle_norm).clamp_min(epsilon)
            ).detach().cpu()
        ),
        "candidate_cosine": float(
            (
                candidate_dot
                / (candidate_norm * oracle_norm).clamp_min(epsilon)
            ).detach().cpu()
        ),
        "baseline_norm": float(baseline_norm.detach().cpu()),
        "candidate_norm": float(candidate_norm.detach().cpu()),
        "correction_norm": float(correction_norm.detach().cpu()),
        "oracle_norm": float(oracle_norm.detach().cpu()),
    }


def _label_free_views(
    dataset: Any,
    indices: np.ndarray,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    weak_views = []
    strong_views = []
    try:
        for index in indices:
            # Deliberately access only tuple element zero.  The real target in
            # element one remains unread until after the batch signal lock.
            image_path = dataset.imgs[int(index)][0]
            image = dataset.loader(image_path)
            views = dataset.transform(image)
            if not isinstance(views, (list, tuple)) or len(views) != 3:
                raise RuntimeError("DUET audit requires test/weak/strong views")
            weak_views.append(views[1])
            strong_views.append(views[2])
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    return torch.stack(weak_views), torch.stack(strong_views)


def _oracle_labels_after_lock(dataset: Any, indices: np.ndarray) -> torch.Tensor:
    labels = torch.tensor(
        [int(dataset.imgs[int(index)][1]) for index in indices],
        dtype=torch.long,
    )
    if labels.shape != (AUDIT_BATCH_SIZE,) or not bool(
        ((labels >= 0) & (labels < len(CLASS_NAMES))).all()
    ):
        raise RuntimeError("Oracle labels changed after the label-free lock")
    return labels


def run_exact_pcgrad_parameter_audit(
    cfg: Any,
    *,
    netF: torch.nn.Module,
    netB: torch.nn.Module,
    netC: torch.nn.Module,
    target_dataset: Any,
    mem_label: torch.Tensor,
    label_mask: torch.Tensor,
    kl_soft: torch.Tensor,
    audit_payload: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Capture exact parameter evidence and stop before cycle-2 optimization."""
    started = time.monotonic()
    output_dir = Path(cfg.output_dir) / str(cfg.PCGRAD_PARAMETER_AUDIT.DIR)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite parameter audit: {output_dir}")
    output_dir.mkdir(parents=True)
    batch_lock_dir = output_dir / "batch_signal_locks"
    batch_lock_dir.mkdir()

    source_label = audit_payload["source_label"].cpu().numpy().astype(np.int64)
    clip_label = audit_payload["clip_label"].cpu().numpy().astype(np.int64)
    admitted = label_mask.cpu().numpy().astype(bool)
    unresolved = (~admitted) & (source_label != clip_label)
    conflict_indices = np.flatnonzero(unresolved)
    admitted_indices = np.flatnonzero(admitted)
    batches, conflict_position = build_locked_parameter_audit_batches(
        conflict_indices, admitted_indices
    )
    if len(target_dataset) != source_label.size:
        raise RuntimeError("Target dataset and cycle-2 pseudo-label state disagree")
    if not (
        float(cfg.ACTIVE.CON_PAR) == 0.2
        and float(cfg.ACTIVE.KL_PAR) == 0.4
        and float(cfg.ACTIVE.CLS_PAR) == 0.4
        and int(cfg.TEST.BATCH_SIZE) == AUDIT_BATCH_SIZE
    ):
        raise RuntimeError("Released DUET loss or batch-size contract changed")

    selection_path = output_dir / "visda_conflict_pcgrad_parameter_label_free.npz"
    signal_lock_path = output_dir / "visda_conflict_pcgrad_parameter_signal_lock.json"
    raw_summary_path = output_dir / "visda_conflict_pcgrad_parameter_runtime_raw.json"
    oracle_path = output_dir / "visda_conflict_pcgrad_parameter_oracle_diagnostic.csv"
    group_path = output_dir / "visda_conflict_pcgrad_parameter_groupwise_oracle_diagnostic.csv"
    np.savez_compressed(
        selection_path,
        batch_indices=batches,
        conflict_position=conflict_position,
        all_unresolved_indices=conflict_indices,
    )
    resident_state_sha256 = _model_sha256((netF, netB, netC))
    source_paths = [
        Path(cfg.output_dir_src) / f"source_{suffix}.pt"
        for suffix in ("F", "B", "C")
    ]
    signal_lock = {
        "phase": "LABEL_FREE_EXACT_PCGRAD_PARAMETER_SELECTION_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "labels_read_only_after_each_batch_signal_lock": True,
        "selection_uses_target_labels": False,
        "selection": {
            "rule": "equal_spacing_in_proxy_order",
            "batch_count": AUDIT_BATCH_COUNT,
            "batch_size": AUDIT_BATCH_SIZE,
            "conflicts_per_batch": CONFLICTS_PER_BATCH,
            "audited_conflicts": AUDITED_CONFLICTS,
            "unresolved_conflicts": int(conflict_indices.size),
            "audited_conflict_coverage_pct": float(
                AUDITED_CONFLICTS / conflict_indices.size * 100.0
            ),
            "view_seed_rule": "SETTING.SEED + one_based_batch_id",
        },
        "candidate_contract": {
            "baseline": "released_arithmetic_DUET_full_batch_loss",
            "changed_term": "unresolved_conflict_consistency_clip_output_gradient_combination",
            "rule": "symmetric_two_objective_PCGrad_only_when_row_dot_negative",
            "unchanged": [
                "pseudo_labels",
                "admission_mask",
                "clip_target",
                "loss_coefficients",
                "batch_size",
                "optimizer",
                "weak_strong_augmentations",
            ],
            "thresholds_fitted": False,
            "target_labels_used_by_rule": False,
        },
        "inputs": {
            "source_checkpoint_sha256": {
                path.name: _sha256(path) for path in source_paths
            },
            "resident_pre_cycle2_model_sha256": resident_state_sha256,
        },
        "selection_path": str(selection_path),
        "selection_sha256": _sha256(selection_path),
    }
    signal_lock_path.write_text(json.dumps(signal_lock, indent=2) + "\n")

    parameters = tuple(
        parameter
        for model in (netF, netB)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    for parameter in parameters:
        parameter.grad = None
    models = (netF, netB)
    original_training = tuple(model.training for model in models)
    netF.train()
    netB.train()
    buffer_state = _buffer_snapshot(models)
    device = next(netF.parameters()).device
    mem_label_cpu = mem_label.detach().cpu().long()
    kl_soft_cpu = kl_soft.detach().cpu().float()
    label_mask_cpu = label_mask.detach().cpu().bool()
    batch_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    try:
        for batch_number in range(AUDIT_BATCH_COUNT):
            indices = batches[batch_number]
            conflict_in_batch = torch.from_numpy(conflict_position[batch_number])
            weak_cpu, strong_cpu = _label_free_views(
                target_dataset,
                indices,
                seed=int(cfg.SETTING.SEED) + batch_number + 1,
            )
            view_sha256 = _tensor_sha256(weak_cpu, strong_cpu)
            weak_x = weak_cpu.to(device, non_blocking=True)
            strong_x = strong_cpu.to(device, non_blocking=True)
            conflict_gpu = conflict_in_batch.to(device)
            admitted_gpu = label_mask_cpu[indices].to(device)
            clip_target = kl_soft_cpu[indices].to(device)
            pseudo_label = mem_label_cpu[indices].to(device)

            weak_logits = netC(netB(netF(weak_x)))
            strong_logits = netC(netB(netF(strong_x)))
            weak_probability = F.softmax(weak_logits, dim=1)
            strong_probability = F.softmax(strong_logits, dim=1)
            consistency_loss = F.kl_div(
                strong_probability.log(), weak_probability, reduction="batchmean"
            )
            clip_loss = F.kl_div(
                weak_probability.log(), clip_target, reduction="batchmean"
            )
            if not bool(admitted_gpu.any()):
                raise RuntimeError("Locked audit batch contains no admitted context")
            pseudo_ce = F.cross_entropy(
                weak_logits[admitted_gpu], pseudo_label[admitted_gpu]
            )
            baseline_loss = (
                float(cfg.ACTIVE.CON_PAR) * consistency_loss
                + float(cfg.ACTIVE.CLS_PAR) * pseudo_ce
                + float(cfg.ACTIVE.KL_PAR) * clip_loss
            )

            con_sum = float(cfg.ACTIVE.CON_PAR) * F.kl_div(
                strong_probability[conflict_gpu].log(),
                weak_probability[conflict_gpu],
                reduction="sum",
            )
            con_weak_grad, con_strong_grad = torch.autograd.grad(
                con_sum,
                (weak_logits, strong_logits),
                retain_graph=True,
            )
            clip_sum = float(cfg.ACTIVE.KL_PAR) * F.kl_div(
                weak_probability[conflict_gpu].log(),
                clip_target[conflict_gpu],
                reduction="sum",
            )
            clip_weak_grad = torch.autograd.grad(
                clip_sum, weak_logits, retain_graph=True
            )[0]
            consistency_descent = torch.cat(
                (-con_weak_grad, -con_strong_grad), dim=1
            ).detach()
            clip_descent = torch.cat(
                (-clip_weak_grad, torch.zeros_like(clip_weak_grad)), dim=1
            ).detach()
            surgery = symmetric_pcgrad_output_correction(
                consistency_descent, clip_descent, conflict_gpu
            )
            correction = surgery["correction"]
            class_count = weak_logits.shape[1]
            correction_grad = torch.autograd.grad(
                (weak_logits, strong_logits),
                parameters,
                grad_outputs=(
                    -correction[:, :class_count] / AUDIT_BATCH_SIZE,
                    -correction[:, class_count:] / AUDIT_BATCH_SIZE,
                ),
                retain_graph=True,
            )
            baseline_grad = torch.autograd.grad(
                baseline_loss, parameters, retain_graph=True
            )
            baseline_norm = _gradient_norm(baseline_grad)
            correction_norm = _gradient_norm(correction_grad)
            candidate_norm = torch.sqrt(
                baseline_norm.square()
                + correction_norm.square()
                + 2.0 * _gradient_dot(baseline_grad, correction_grad)
            ).clamp_min(0.0)
            active_conflict = surgery["active"] & conflict_gpu
            conflict_cosine = surgery["component_cosine"][conflict_gpu]
            batch_lock = {
                "phase": "LABEL_FREE_EXACT_PCGRAD_PARAMETER_BATCH_LOCK",
                "batch": batch_number + 1,
                "contains_target_labels": False,
                "labels_read_after_this_manifest": True,
                "selection_lock_sha256": _sha256(signal_lock_path),
                "indices": [int(index) for index in indices],
                "conflict_positions": [
                    bool(value) for value in conflict_in_batch.tolist()
                ],
                "view_sha256": view_sha256,
                "losses": {
                    "consistency": float(consistency_loss.detach().cpu()),
                    "pseudo_ce": float(pseudo_ce.detach().cpu()),
                    "clip_kl": float(clip_loss.detach().cpu()),
                    "weighted_full_baseline": float(baseline_loss.detach().cpu()),
                },
                "label_free_gradient_metrics": {
                    "output_pcgrad_active_conflicts": int(
                        active_conflict.sum().detach().cpu()
                    ),
                    "mean_conflict_component_cosine": float(
                        conflict_cosine.mean().detach().cpu()
                    ),
                    "baseline_parameter_gradient_norm": float(
                        baseline_norm.detach().cpu()
                    ),
                    "correction_parameter_gradient_norm": float(
                        correction_norm.detach().cpu()
                    ),
                    "candidate_parameter_gradient_norm": float(
                        candidate_norm.detach().cpu()
                    ),
                },
            }
            batch_lock_path = batch_lock_dir / (
                f"visda_conflict_pcgrad_parameter_batch{batch_number + 1:02d}_lock.json"
            )
            batch_lock_path.write_text(json.dumps(batch_lock, indent=2) + "\n")

            # Oracle diagnostic begins only after both the global selection
            # lock and this batch's exact label-free gradient lock exist.
            labels = _oracle_labels_after_lock(target_dataset, indices).to(device)
            oracle_loss = 0.5 * (
                F.cross_entropy(weak_logits, labels)
                + F.cross_entropy(strong_logits, labels)
            )
            oracle_grad = torch.autograd.grad(
                oracle_loss, parameters, retain_graph=True
            )
            metrics = _alignment_metrics(
                baseline_grad, correction_grad, oracle_grad
            )
            batch_rows.append(
                {
                    "batch": batch_number + 1,
                    "samples": AUDIT_BATCH_SIZE,
                    "conflict_samples": CONFLICTS_PER_BATCH,
                    "output_pcgrad_active_conflicts": int(
                        active_conflict.sum().detach().cpu()
                    ),
                    "mean_conflict_component_cosine": float(
                        conflict_cosine.mean().detach().cpu()
                    ),
                    **metrics,
                    "batch_signal_lock_sha256": _sha256(batch_lock_path),
                }
            )
            del oracle_grad

            present_groups = [
                name
                for name, class_indices in GROUP_INDICES.items()
                if bool(
                    torch.isin(
                        labels,
                        torch.tensor(class_indices, device=device),
                    ).any()
                )
            ]
            for group_position, group_name in enumerate(present_groups):
                class_indices = torch.tensor(
                    GROUP_INDICES[group_name], device=device
                )
                group_mask = torch.isin(labels, class_indices)
                group_oracle_loss = 0.5 * (
                    F.cross_entropy(weak_logits[group_mask], labels[group_mask])
                    + F.cross_entropy(
                        strong_logits[group_mask], labels[group_mask]
                    )
                )
                group_oracle_grad = torch.autograd.grad(
                    group_oracle_loss,
                    parameters,
                    retain_graph=group_position < len(present_groups) - 1,
                )
                group_rows.append(
                    {
                        "batch": batch_number + 1,
                        "group": group_name,
                        "samples": int(group_mask.sum().detach().cpu()),
                        "candidate_minus_baseline_first_order": float(
                            _gradient_dot(
                                correction_grad, group_oracle_grad
                            ).detach().cpu()
                        ),
                    }
                )
                del group_oracle_grad
            _restore_buffers(models, buffer_state)
            del (
                weak_x,
                strong_x,
                weak_logits,
                strong_logits,
                weak_probability,
                strong_probability,
                baseline_grad,
                correction_grad,
            )
            torch.cuda.empty_cache()
    finally:
        _restore_buffers(models, buffer_state)
        for model, was_training in zip(models, original_training):
            model.train(was_training)
        for parameter in parameters:
            parameter.grad = None

    _write_csv(oracle_path, batch_rows)
    _write_csv(group_path, group_rows)
    raw_summary = {
        "dataset": "VISDA-C",
        "seed": int(cfg.SETTING.SEED),
        "decision": "EXACT_PARAMETER_EVIDENCE_CAPTURED",
        "oracle_diagnostic": True,
        "labels_used_only_after_global_and_batch_signal_locks": True,
        "selection_signal_lock": str(signal_lock_path),
        "selection_signal_lock_sha256": _sha256(signal_lock_path),
        "batch_signal_lock_count": len(list(batch_lock_dir.glob("*.json"))),
        "unresolved_conflicts": int(conflict_indices.size),
        "audited_conflicts": AUDITED_CONFLICTS,
        "audited_conflict_coverage_pct": float(
            AUDITED_CONFLICTS / conflict_indices.size * 100.0
        ),
        "output_pcgrad_active_conflicts": int(
            sum(row["output_pcgrad_active_conflicts"] for row in batch_rows)
        ),
        "cycle2_optimizer_steps": 0,
        "parameters_updated_by_audit": False,
        "model_buffers_restored_after_each_batch": True,
        "target_labels_used_by_candidate": False,
        "thresholds_fitted": False,
        "batch_oracle_diagnostic": str(oracle_path),
        "groupwise_oracle_diagnostic": str(group_path),
        "runtime_seconds": time.monotonic() - started,
    }
    raw_summary_path.write_text(json.dumps(raw_summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "decision": raw_summary["decision"],
                "unresolved_conflicts": raw_summary["unresolved_conflicts"],
                "audited_conflicts": raw_summary["audited_conflicts"],
                "output_pcgrad_active_conflicts": raw_summary[
                    "output_pcgrad_active_conflicts"
                ],
                "cycle2_optimizer_steps": 0,
            },
            indent=2,
        )
    )
    return raw_summary
