from pathlib import Path

import numpy as np
import pytest

from src.utils.support_conditioned_clip_audit import (
    evaluate_support_conditioned_clip_gate,
    full_target_class_mass_shift_pp,
    negative_first_order_burden,
    probability_entropy,
    support_conditioned_probability,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_support_conditioning_preserves_relative_mass_and_zeros_tail() -> None:
    probability = np.array([[0.5, 0.3, 0.15, 0.05]])
    mask = np.array([[True, False, True, False]])
    result = support_conditioned_probability(probability, mask)
    np.testing.assert_allclose(result["retained_mass"], [0.65])
    np.testing.assert_allclose(
        result["probability"], [[0.5 / 0.65, 0.0, 0.15 / 0.65, 0.0]]
    )
    assert result["probability"].argmax(axis=1).item() == 0


def test_support_conditioning_accepts_float32_simplex_drift() -> None:
    probability = np.array([[0.7, 0.2, 0.09999994]], dtype=np.float32)
    result = support_conditioned_probability(
        probability, np.array([[True, True, False]])
    )
    np.testing.assert_allclose(result["probability"].sum(axis=1), 1.0, atol=1e-14)


def test_entropy_mass_shift_and_negative_burden_are_exact() -> None:
    baseline = np.array([[0.5, 0.5], [0.2, 0.8]])
    candidate = np.array([[0.6, 0.4], [0.4, 0.6]])
    shift = full_target_class_mass_shift_pp(candidate, baseline, full_target_samples=4)
    np.testing.assert_allclose(shift, [7.5, -7.5])
    np.testing.assert_allclose(
        probability_entropy(np.array([[1.0, 0.0], [0.5, 0.5]])),
        [0.0, np.log(2.0)],
    )
    assert negative_first_order_burden(np.array([-2.0, 1.0, -1.0, 3.0])) == -0.75


def _comparison(value: float = 0.1) -> dict:
    metric = {
        "mean_difference": value,
        "paired_bootstrap_95_ci": [value / 2.0, value * 1.5],
    }
    return {
        "cosine": dict(metric),
        "oracle_unit_projection": dict(metric),
        "first_order": dict(metric),
    }


def test_gate_requires_alignment_class_safety_burden_and_mass_control() -> None:
    passing = evaluate_support_conditioned_clip_gate(
        input_contract_valid=True,
        versus_clip=_comparison(),
        versus_top1_union=_comparison(),
        minimum_class_first_order_delta_vs_clip=0.001,
        candidate_negative_burden=-0.08,
        clip_negative_burden=-0.09,
        top1_union_negative_burden=-0.085,
        clip_top2_negative_burden=-0.10,
        candidate_max_full_mass_shift_pp=0.7,
        top1_union_max_full_mass_shift_pp=1.2,
        clip_top2_max_full_mass_shift_pp=1.1,
        candidate_top1_matches_clip=True,
    )
    assert passing["decision"] == "PASS_SUPPORT_CONDITIONED_CLIP_PREFLIGHT"
    assert passing["training_authorized"] is False

    failing = evaluate_support_conditioned_clip_gate(
        input_contract_valid=True,
        versus_clip=_comparison(),
        versus_top1_union=_comparison(value=-0.1),
        minimum_class_first_order_delta_vs_clip=-1e-6,
        candidate_negative_burden=-0.11,
        clip_negative_burden=-0.09,
        top1_union_negative_burden=-0.085,
        clip_top2_negative_burden=-0.10,
        candidate_max_full_mass_shift_pp=1.3,
        top1_union_max_full_mass_shift_pp=1.2,
        clip_top2_max_full_mass_shift_pp=1.1,
        candidate_top1_matches_clip=False,
    )
    assert failing["decision"] == "REJECT"
    assert not failing["checks"]["candidate_top1_matches_clip"]
    assert not failing["checks"]["every_class_first_order_delta_vs_clip_nonnegative"]
    assert not failing["checks"]["candidate_full_mass_shift_at_most_1pp"]


def test_support_validation_rejects_empty_or_wrong_shape() -> None:
    probability = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="non-empty"):
        support_conditioned_probability(probability, np.array([[False, False]]))
    with pytest.raises(ValueError, match="shapes must match"):
        support_conditioned_probability(probability, np.array([[True]]))


def test_cloud_entrypoint_is_cpu_only_and_locks_before_oracle_labels() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_support_conditioned_clip_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_support_conditioned_clip.py"
    ).read_text()
    assert 'CUDA_VISIBLE_DEVICES="" python' in runner
    assert "image_target_of_oh_vs.py" not in runner
    assert "import torch" not in audit
    assert "import clip" not in audit
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert '"target_images_loaded": False' in audit
    assert '"model_forward_calls": 0' in audit
    assert '"training_code_modified": False' in audit
    assert '"training_authorized": False' in audit
    assert audit.index("lock_path.write_text") < audit.index(
        "target_list_hash = _sha256(args.target_list)"
    )
    assert audit.index("lock_path.write_text") < audit.index(
        "_parse_labels_after_lock(args.target_list)"
    )
