"""Consistency objectives shared by DCCL implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prediction_consistency_kl(
    weak_logits: torch.Tensor,
    strong_logits: torch.Tensor,
    *,
    stop_gradient: bool = False,
) -> torch.Tensor:
    """Match strong-view predictions to the weak-view teacher distribution."""

    weak_prob = F.softmax(weak_logits, dim=1)
    if stop_gradient:
        weak_prob = weak_prob.detach()
    strong_prob = F.softmax(strong_logits, dim=1)
    return F.kl_div(strong_prob.log(), weak_prob, reduction="batchmean")
