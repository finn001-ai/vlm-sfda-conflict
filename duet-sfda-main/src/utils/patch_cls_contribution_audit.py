"""Patch-to-CLS contribution helpers for a frozen CLIP preflight.

The extraction follows the patch-specific value terms written into the CLS
attention output of the final visual transformer block.  It deliberately
excludes the value/output biases because those terms are not patch-specific.
No parameter is learned or updated here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def final_block_patch_cls_contributions(
    block_input: torch.Tensor,
    block: torch.nn.Module,
    visual_projection: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return final-block patch contributions for all/even/odd heads.

    ``block_input`` is the sequence-first input to an OpenAI-CLIP
    ``ResidualAttentionBlock`` with shape ``[token, batch, width]``.  The
    returned contribution tensors have shape ``[batch, patch, embed]``.
    """
    if block_input.ndim != 3 or block_input.shape[0] < 2:
        raise ValueError("block_input must have shape [token>=2, batch, width]")
    attention = block.attn
    token_count, batch_size, width = block_input.shape
    if attention.embed_dim != width or width % attention.num_heads:
        raise ValueError("attention width/head contract is incompatible")
    if visual_projection.ndim != 2 or visual_projection.shape[0] != width:
        raise ValueError("visual_projection must have shape [width, embed]")
    if attention.in_proj_weight.shape != (3 * width, width):
        raise ValueError("only packed QKV MultiheadAttention is supported")

    normalized = block.ln_1(block_input)
    q_weight, k_weight, v_weight = attention.in_proj_weight.chunk(3, dim=0)
    if attention.in_proj_bias is None:
        q_bias = k_bias = None
    else:
        q_bias, k_bias, _v_bias = attention.in_proj_bias.chunk(3, dim=0)

    query = F.linear(normalized[0], q_weight, q_bias)
    key = F.linear(normalized, k_weight, k_bias)
    # TraceCLIP's patch-specific term is W_OV LN(x_i), not a repeated value
    # bias.  Excluding that bias also prevents a shared constant from being
    # spuriously attributed to every patch.
    value = F.linear(normalized, v_weight, bias=None)

    head_count = int(attention.num_heads)
    head_width = width // head_count
    query = query.reshape(batch_size, head_count, head_width)
    key = (
        key.permute(1, 0, 2)
        .reshape(batch_size, token_count, head_count, head_width)
        .permute(0, 2, 1, 3)
    )
    value = (
        value.permute(1, 0, 2)
        .reshape(batch_size, token_count, head_count, head_width)
        .permute(0, 2, 1, 3)
    )
    attention_logit = torch.einsum("bhd,bhld->bhl", query, key)
    attention_logit = attention_logit * (head_width**-0.5)
    if block.attn_mask is not None:
        mask = block.attn_mask.to(
            device=attention_logit.device, dtype=attention_logit.dtype
        )
        attention_logit = attention_logit + mask[0].reshape(1, 1, -1)
    attention_probability = attention_logit.softmax(dim=-1)
    per_head_token = attention_probability[..., None] * value

    def project(head_mask: torch.Tensor) -> torch.Tensor:
        selected = per_head_token * head_mask.reshape(1, head_count, 1, 1)
        concatenated = selected.permute(0, 2, 1, 3).reshape(
            batch_size, token_count, width
        )
        attention_space = F.linear(concatenated, attention.out_proj.weight, bias=None)
        return attention_space @ visual_projection

    head_index = torch.arange(head_count, device=block_input.device)
    all_mask = torch.ones(
        head_count, dtype=block_input.dtype, device=block_input.device
    )
    even_mask = (head_index.remainder(2) == 0).to(block_input.dtype)
    odd_mask = 1.0 - even_mask
    all_token = project(all_mask)
    even_token = project(even_mask)
    odd_token = project(odd_mask)
    partition_error = (all_token - even_token - odd_token).abs().max()
    return {
        "all": all_token[:, 1:],
        "even": even_token[:, 1:],
        "odd": odd_token[:, 1:],
        "cls": all_token[:, :1],
        "attention_probability": attention_probability,
        "head_partition_max_abs_error": partition_error,
    }


def encode_last_block_patch_cls_contributions(
    clip_model: torch.nn.Module,
    images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run one frozen CLIP image forward and capture final-block contributions."""
    visual = clip_model.visual
    if not all(hasattr(visual, name) for name in ("transformer", "proj", "conv1")):
        raise ValueError("patch contribution audit requires a CLIP VisionTransformer")
    blocks = visual.transformer.resblocks
    if len(blocks) == 0:
        raise ValueError("visual transformer has no residual blocks")
    captured: dict[str, torch.Tensor] = {}

    def capture(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if len(args) != 1:
            raise RuntimeError("unexpected residual-block input contract")
        captured["input"] = args[0]

    handle = blocks[-1].register_forward_pre_hook(capture)
    try:
        image_feature = clip_model.encode_image(images)
    finally:
        handle.remove()
    if "input" not in captured:
        raise RuntimeError("final visual block input was not captured")
    contribution = final_block_patch_cls_contributions(
        captured["input"], blocks[-1], visual.proj
    )
    contribution["image_feature"] = image_feature
    return contribution


def candidate_peak_response(
    contribution: torch.Tensor,
    text_feature: torch.Tensor,
    task_candidate: torch.Tensor,
    clip_candidate: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the maximum patch response for the two declared candidates."""
    if contribution.ndim != 3 or contribution.shape[1] == 0:
        raise ValueError("contribution must have shape [batch, patch, embed]")
    if text_feature.ndim != 2 or text_feature.shape[1] != contribution.shape[2]:
        raise ValueError("text_feature is incompatible with contributions")
    batch_size = contribution.shape[0]
    if task_candidate.shape != (batch_size,) or clip_candidate.shape != (batch_size,):
        raise ValueError("candidate vectors must align with the batch")
    if torch.any(task_candidate == clip_candidate):
        raise ValueError("the patch contribution query must be a top-1 conflict")
    if torch.any(task_candidate < 0) or torch.any(
        task_candidate >= text_feature.shape[0]
    ):
        raise ValueError("task candidate is outside the class range")
    if torch.any(clip_candidate < 0) or torch.any(
        clip_candidate >= text_feature.shape[0]
    ):
        raise ValueError("CLIP candidate is outside the class range")

    normalized_contribution = F.normalize(contribution.float(), dim=-1)
    normalized_text = F.normalize(text_feature.float(), dim=-1)
    task_text = normalized_text[task_candidate]
    clip_text = normalized_text[clip_candidate]
    task_patch_response = torch.einsum("bpd,bd->bp", normalized_contribution, task_text)
    clip_patch_response = torch.einsum("bpd,bd->bp", normalized_contribution, clip_text)
    task_peak, task_patch = task_patch_response.max(dim=1)
    clip_peak, clip_patch = clip_patch_response.max(dim=1)
    return {
        "task_peak": task_peak,
        "clip_peak": clip_peak,
        "task_peak_patch": task_patch,
        "clip_peak_patch": clip_patch,
        "choose_task": task_peak > clip_peak,
    }


def unanimous_head_partition_rescue(
    task_candidate: np.ndarray,
    clip_candidate: np.ndarray,
    full_task_peak: np.ndarray,
    full_clip_peak: np.ndarray,
    even_task_peak: np.ndarray,
    even_clip_peak: np.ndarray,
    odd_task_peak: np.ndarray,
    odd_clip_peak: np.ndarray,
) -> dict[str, np.ndarray]:
    """Default to CLIP and rescue task only under three strict victories."""
    task = np.asarray(task_candidate, dtype=np.int64)
    clip = np.asarray(clip_candidate, dtype=np.int64)
    score_arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            full_task_peak,
            full_clip_peak,
            even_task_peak,
            even_clip_peak,
            odd_task_peak,
            odd_clip_peak,
        )
    )
    if task.ndim != 1 or clip.shape != task.shape:
        raise ValueError("candidate vectors must be aligned and one-dimensional")
    if np.any(task == clip):
        raise ValueError("task and CLIP candidates must conflict")
    if any(value.shape != task.shape for value in score_arrays):
        raise ValueError("every score vector must align with candidates")
    if not all(np.isfinite(value).all() for value in score_arrays):
        raise ValueError("candidate peak responses must be finite")
    ft, fc, et, ec, ot, oc = score_arrays
    full_choice = ft > fc
    even_choice = et > ec
    odd_choice = ot > oc
    rescue = full_choice & even_choice & odd_choice
    prediction = np.where(rescue, task, clip)
    minimum_margin = np.minimum.reduce((ft - fc, et - ec, ot - oc))
    return {
        "prediction": prediction,
        "rescue_task": rescue,
        "full_choose_task": full_choice,
        "even_choose_task": even_choice,
        "odd_choose_task": odd_choice,
        "minimum_task_margin": minimum_margin,
        "even_odd_agreement": even_choice == odd_choice,
        "all_partition_agreement": (full_choice == even_choice)
        & (full_choice == odd_choice),
    }


def evaluate_patch_cls_contribution_gate(
    *,
    input_contract_valid: bool,
    head_partition_max_abs_error: float,
    all_partition_agreement_pct: float,
    route_coverage_pct: float,
    comparisons: dict[str, dict[str, Any]],
    routed_net_corrections: int,
    routed_task_precision_pct: float,
    full_proxy_macro_gain_pp: float,
    car_delta_pp: float,
    truck_delta_pp: float,
    car_truck_mean_delta_pp: float,
    other_ten_mean_delta_pp: float,
    max_class_mass_shift_pp: float,
) -> dict[str, Any]:
    """Predeclared oracle gate; every rule above is label-free."""
    required = {"fixed_task", "fixed_clip", "confidence_choice", "arithmetic", "rms"}
    if set(comparisons) != required:
        raise ValueError("comparisons must contain the five matched selectors")
    best_name = max(
        comparisons,
        key=lambda name: (comparisons[name]["baseline_accuracy_pct"], name),
    )
    best = comparisons[best_name]
    checks = {
        "input_contract_valid": bool(input_contract_valid),
        "head_partition_max_abs_error_at_most_1e_5": (
            float(head_partition_max_abs_error) <= 1e-5
        ),
        "all_partition_decision_agreement_at_least_80pct": (
            float(all_partition_agreement_pct) >= 80.0
        ),
        "route_coverage_between_2_and_30pct": (
            2.0 <= float(route_coverage_pct) <= 30.0
        ),
        "routed_task_precision_at_least_60pct": (
            float(routed_task_precision_pct) >= 60.0
        ),
        "routed_net_corrections_positive": int(routed_net_corrections) > 0,
        "accuracy_gain_vs_best_baseline_at_least_1pp": (float(best["gain_pp"]) >= 1.0),
        "accuracy_gain_vs_best_baseline_ci_lower_positive": (
            float(best["paired_bootstrap_95_ci_pp"][0]) > 0.0
        ),
        "beats_every_matched_baseline": all(
            float(value["gain_pp"]) > 0.0 for value in comparisons.values()
        ),
        "full_proxy_macro_gain_at_least_0_20pp": (
            float(full_proxy_macro_gain_pp) >= 0.20
        ),
        "car_regression_at_most_0_5pp": float(car_delta_pp) >= -0.5,
        "truck_regression_at_most_0_5pp": float(truck_delta_pp) >= -0.5,
        "car_truck_mean_nonnegative": float(car_truck_mean_delta_pp) >= 0.0,
        "other_ten_mean_nonnegative": float(other_ten_mean_delta_pp) >= 0.0,
        "max_class_mass_shift_at_most_1pp": float(max_class_mass_shift_pp) <= 1.0,
    }
    return {
        "decision": (
            "PASS_PATCH_CLS_CONTRIBUTION_PREFLIGHT"
            if all(checks.values())
            else "REJECT"
        ),
        "checks": checks,
        "best_baseline_name": best_name,
        "training_authorized": False,
        "proxy_authorized": False,
        "parameter_audit_authorized": all(checks.values()),
    }
