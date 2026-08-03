from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clip.model import ResidualAttentionBlock
from src.utils.patch_cls_contribution_audit import (
    candidate_peak_response,
    evaluate_patch_cls_contribution_gate,
    final_block_patch_cls_contributions,
    unanimous_head_partition_rescue,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_head_partitions_sum_to_full_patch_contribution() -> None:
    torch.manual_seed(7)
    # OpenAI CLIP's custom LayerNorm evaluates in float32 even when its input
    # has another dtype, so this test mirrors the actual DUET float32 path.
    block = ResidualAttentionBlock(d_model=8, n_head=4).float()
    block_input = torch.randn(5, 3, 8, dtype=torch.float32)
    projection = torch.randn(8, 6, dtype=torch.float32)
    result = final_block_patch_cls_contributions(block_input, block, projection)

    assert result["all"].shape == (3, 4, 6)
    torch.testing.assert_close(result["all"], result["even"] + result["odd"])
    assert float(result["head_partition_max_abs_error"].detach()) < 1e-6
    torch.testing.assert_close(
        result["attention_probability"].sum(dim=-1),
        torch.ones(3, 4, dtype=torch.float32),
    )


def test_semantic_terms_plus_shared_bias_reconstruct_cls_attention_output() -> None:
    torch.manual_seed(11)
    block = ResidualAttentionBlock(d_model=8, n_head=4).float()
    block_input = torch.randn(5, 2, 8)
    result = final_block_patch_cls_contributions(block_input, block, torch.eye(8))
    normalized = block.ln_1(block_input)
    actual_cls = block.attention(normalized)[0]
    _query_bias, _key_bias, value_bias = block.attn.in_proj_bias.chunk(3)
    shared_bias = F.linear(
        value_bias, block.attn.out_proj.weight, block.attn.out_proj.bias
    )
    reconstructed = result["cls"][:, 0] + result["all"].sum(dim=1) + shared_bias
    torch.testing.assert_close(reconstructed, actual_cls, atol=1e-6, rtol=1e-5)


def test_peak_response_compares_only_declared_candidates() -> None:
    contribution = torch.tensor(
        [
            [[1.0, 0.0], [0.8, 0.2]],
            [[0.0, 1.0], [0.1, 0.9]],
        ]
    )
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    result = candidate_peak_response(
        contribution,
        text,
        torch.tensor([0, 2]),
        torch.tensor([1, 1]),
    )
    assert result["choose_task"].tolist() == [True, False]
    assert result["task_peak_patch"].tolist() == [0, 0]


def test_unanimous_rescue_defaults_to_clip_on_any_partition_disagreement() -> None:
    result = unanimous_head_partition_rescue(
        np.array([1, 1, 1]),
        np.array([2, 2, 2]),
        np.array([0.8, 0.8, 0.8]),
        np.array([0.2, 0.2, 0.2]),
        np.array([0.7, 0.1, 0.7]),
        np.array([0.3, 0.9, 0.3]),
        np.array([0.6, 0.6, 0.1]),
        np.array([0.4, 0.4, 0.9]),
    )
    assert result["rescue_task"].tolist() == [True, False, False]
    assert result["prediction"].tolist() == [1, 2, 2]
    np.testing.assert_allclose(result["minimum_task_margin"], [0.2, -0.8, -0.8])


def _comparisons(gain: float = 1.2, low: float = 0.3) -> dict:
    return {
        name: {
            "candidate_accuracy_pct": 73.5,
            "baseline_accuracy_pct": 72.3 - offset * 0.1,
            "gain_pp": gain + offset * 0.1,
            "paired_bootstrap_95_ci_pp": [low, 2.0],
        }
        for offset, name in enumerate(
            ("fixed_task", "fixed_clip", "confidence_choice", "arithmetic", "rms")
        )
    }


def test_gate_requires_gain_stability_and_class_safety() -> None:
    kwargs = {
        "input_contract_valid": True,
        "head_partition_max_abs_error": 1e-7,
        "all_partition_agreement_pct": 90.0,
        "route_coverage_pct": 10.0,
        "comparisons": _comparisons(),
        "routed_net_corrections": 100,
        "routed_task_precision_pct": 65.0,
        "full_proxy_macro_gain_pp": 0.3,
        "car_delta_pp": 0.2,
        "truck_delta_pp": 0.1,
        "car_truck_mean_delta_pp": 0.15,
        "other_ten_mean_delta_pp": 0.3,
        "max_class_mass_shift_pp": 0.8,
    }
    passing = evaluate_patch_cls_contribution_gate(**kwargs)
    assert passing["decision"] == "PASS_PATCH_CLS_CONTRIBUTION_PREFLIGHT"
    assert passing["training_authorized"] is False
    assert passing["proxy_authorized"] is False
    assert passing["parameter_audit_authorized"] is True

    kwargs["truck_delta_pp"] = -0.8
    rejected = evaluate_patch_cls_contribution_gate(**kwargs)
    assert rejected["decision"] == "REJECT"
    assert not rejected["checks"]["truck_regression_at_most_0_5pp"]


def test_entrypoint_is_frozen_forward_only_and_locks_before_oracle() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_patch_cls_contribution_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_patch_cls_contribution.py"
    ).read_text()
    assert "source_F.pt" not in runner
    assert "source_B.pt" not in runner
    assert "source_C.pt" not in runner
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert 'args.arch != "ViT-B/32"' in audit
    assert '"training_authorized": False' in audit
    assert audit.index("lock_path.write_text") < audit.rindex(
        "labels = _parse_labels_after_lock"
    )
    assert audit.index("lock_path.write_text") < audit.rindex(
        'snapshot["target_label"]'
    )
