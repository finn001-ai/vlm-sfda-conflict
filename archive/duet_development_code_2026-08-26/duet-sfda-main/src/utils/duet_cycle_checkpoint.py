"""Exact cycle-boundary checkpoint I/O for DUET diagnostic reruns."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CYCLE_CHECKPOINT_FORMAT_VERSION = 1


def capture_process_rng_state() -> dict[str, Any]:
    """Capture every process RNG that can affect the next DUET cycle."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().cpu(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_process_rng_state(state: dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_process_rng_state`."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise ValueError(
            "cycle checkpoint RNG state is missing: {}".format(sorted(missing))
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("cycle checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "cycle checkpoint CUDA device count differs: saved={}, current={}".format(
                    len(cuda_states), torch.cuda.device_count()
                )
            )
        torch.cuda.set_rng_state_all(cuda_states)


def save_cycle_checkpoint(path: str, payload: dict[str, Any]) -> Path:
    """Atomically save a new checkpoint without overwriting an existing cache."""
    checkpoint_path = Path(path).expanduser()
    if not str(path).strip():
        raise ValueError("cycle checkpoint save path must be non-empty")
    if checkpoint_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing cycle checkpoint: {}".format(
                checkpoint_path
            )
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(
            "stale temporary cycle checkpoint exists: {}".format(temporary_path)
        )
    stored_payload = dict(payload)
    stored_payload["format_version"] = CYCLE_CHECKPOINT_FORMAT_VERSION
    try:
        torch.save(stored_payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return checkpoint_path


def load_cycle_checkpoint(path: str) -> dict[str, Any]:
    """Load a trusted local DUET checkpoint onto CPU and validate its format."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "cycle checkpoint not found: {}".format(checkpoint_path)
        )
    # This file is generated locally by save_cycle_checkpoint and contains
    # Python/NumPy RNG tuples in addition to tensor weights.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("cycle checkpoint payload must be a dictionary")
    version = payload.get("format_version")
    if version != CYCLE_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported cycle checkpoint format: saved={}, expected={}".format(
                version, CYCLE_CHECKPOINT_FORMAT_VERSION
            )
        )
    return payload


def validate_cycle_checkpoint_contract(
    saved: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Reject cache reuse when any Cycle-1-affecting setting changed."""
    mismatches = []
    for key in sorted(set(saved) | set(expected)):
        saved_value = saved.get(key, "<missing>")
        expected_value = expected.get(key, "<missing>")
        if saved_value != expected_value:
            mismatches.append(
                "{}: saved={!r}, current={!r}".format(
                    key, saved_value, expected_value
                )
            )
    if mismatches:
        raise ValueError(
            "cycle checkpoint contract mismatch:\n  " + "\n  ".join(mismatches)
        )
