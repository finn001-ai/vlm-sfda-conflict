"""Shared, side-effect-free snapshot writer for matched failure audits."""

from __future__ import annotations

import logging
import os
import os.path as osp
from typing import Any

import numpy as np
import torch


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def save_failure_audit_snapshot(
    cfg,
    filename: str,
    *,
    feature_keys: tuple[str, ...] = ("task_feature", "adapted_task_feature"),
    **payload: Any,
) -> str | None:
    """Write one compressed audit snapshot without changing training state."""
    if not cfg.FAILURE_AUDIT.ENABLED:
        return None
    if cfg.FAILURE_AUDIT.FEATURE_DTYPE not in {"float16", "float32"}:
        raise ValueError("FAILURE_AUDIT.FEATURE_DTYPE must be float16 or float32")
    if not filename.endswith(".npz") or osp.basename(filename) != filename:
        raise ValueError("Failure-audit filename must be a local .npz basename")

    arrays = {key: _as_numpy(value) for key, value in payload.items()}
    row_counts = {
        int(value.shape[0])
        for key, value in arrays.items()
        if key not in {"cycle", "task", "phase"} and value.ndim > 0
    }
    if len(row_counts) > 1:
        raise ValueError(f"Failure-audit arrays have mismatched rows: {row_counts}")

    feature_dtype = np.float16 if cfg.FAILURE_AUDIT.FEATURE_DTYPE == "float16" else np.float32
    for key in feature_keys:
        if key in arrays:
            arrays[key] = arrays[key].astype(feature_dtype, copy=False)

    out_dir = osp.join(cfg.output_dir, cfg.FAILURE_AUDIT.DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, filename)
    np.savez_compressed(out_path, **arrays)
    logging.info("Failure audit wrote: %s", out_path)
    return out_path
