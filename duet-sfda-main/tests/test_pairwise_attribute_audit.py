from pathlib import Path

import numpy as np

from src.utils.pairwise_attribute_audit import (
    ATTRIBUTE_FAMILIES,
    PROMPT_TEMPLATES,
    VISDA_VISIBLE_ATTRIBUTES,
    build_visda_attribute_prompt_manifest,
    evaluate_pairwise_attribute_gate,
    pairwise_attribute_task_rescue,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _class_names() -> list[str]:
    return list(VISDA_VISIBLE_ATTRIBUTES)


def test_prompt_contract_is_complete_deterministic_and_label_free() -> None:
    first = build_visda_attribute_prompt_manifest(_class_names())
    second = build_visda_attribute_prompt_manifest(_class_names())

    assert first == second
    assert first["shape"] == [12, 2, 4]
    assert len(first["flat_prompts"]) == 96
    assert not first["target_images_used"]
    assert not first["target_labels_used"]
    assert not first["external_visual_models_used"]
    assert first["attribute_families"] == list(ATTRIBUTE_FAMILIES)
    assert first["prompt_templates"] == list(PROMPT_TEMPLATES)
    assert any("cargo bed" in prompt for prompt in first["flat_prompts"])


def test_task_rescue_requires_all_families_and_stable_templates() -> None:
    text = np.zeros((2, 2, 4, 2), dtype=np.float64)
    text[0, :, :, 0] = 1.0
    text[1, :, :, 1] = 1.0
    image = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = pairwise_attribute_task_rescue(
        image,
        text,
        task_prediction=np.array([0, 0]),
        clip_prediction=np.array([1, 1]),
    )

    assert result["task_rescue"].tolist() == [True, False]
    assert result["routed_prediction"].tolist() == [0, 1]
    assert result["descriptor_prediction"].tolist() == [0, 1]
    np.testing.assert_allclose(result["template_stability"], [1.0, 1.0])


def test_unstable_template_evidence_cannot_rescue_task() -> None:
    text = np.zeros((2, 2, 4, 2), dtype=np.float64)
    text[0, 0, :, 0] = 1.0
    text[1, 0, :, 1] = 1.0
    text[0, 1, :, 0] = 0.51
    text[0, 1, :, 1] = 0.49
    text[1, 1, :, 0] = 0.49
    text[1, 1, :, 1] = 0.51
    # Reverse alternating family strengths in the second template.  Its mean
    # still favors task, but the template margin vectors are not stable.
    text[0, 1, 1::2] = np.array([0.49, 0.51])
    text[1, 1, 1::2] = np.array([0.51, 0.49])
    image = np.array([[1.0, 0.0]])
    result = pairwise_attribute_task_rescue(
        image,
        text,
        task_prediction=np.array([0]),
        clip_prediction=np.array([1]),
    )

    assert not result["task_rescue"][0]
    assert result["routed_prediction"].tolist() == [1]


def test_gate_requires_gain_coverage_precision_and_no_car_truck_exchange() -> None:
    passing = evaluate_pairwise_attribute_gate(
        reproduction_passed=True,
        conflict_gain_pp=1.2,
        conflict_gain_ci=(0.2, 2.2),
        rescue_coverage=5.5,
        adjudication_precision=61.0,
        median_routed_stability=0.95,
        car_net_corrections=3,
        truck_net_corrections=1,
    )
    exchange = evaluate_pairwise_attribute_gate(
        reproduction_passed=True,
        conflict_gain_pp=1.2,
        conflict_gain_ci=(0.2, 2.2),
        rescue_coverage=5.5,
        adjudication_precision=61.0,
        median_routed_stability=0.95,
        car_net_corrections=3,
        truck_net_corrections=-1,
    )

    assert passing["decision"] == "PASS_OFFLINE_GATE"
    assert exchange["decision"] == "REJECT"
    assert not exchange["checks"]["truck_net_corrections_nonnegative"]


def test_cloud_entrypoint_is_frozen_and_locks_before_oracle_labels() -> None:
    runner = (
        REPO_ROOT / "tools/run_visda_conflict_pairwise_attribute_audit.sh"
    ).read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_conflict_pairwise_attributes.py"
    ).read_text()

    assert "image_target_of_oh_vs.py" not in runner
    assert "optimizer.step" not in audit
    assert ".backward(" not in audit
    assert 'str(cfg.ACTIVE.ARCH) != "ViT-B/32"' in audit
    assert '"optimizer_steps": 0' in audit
    assert '"training_authorized": False' in audit
    assert '"top2_candidates": False' in audit
    assert audit.index("lock_path.write_text") < audit.rindex(
        "_parse_labels_after_lock("
    )
