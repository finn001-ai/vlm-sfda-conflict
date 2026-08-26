from pathlib import Path

import numpy as np
import torch

from src.utils.attribute_mass_audit import entropy_anchored_attribute_mass
from src.utils.attribute_reliability import (
    entropy_anchored_attribute_target,
    pairwise_attribute_margin,
)
from tools.analyze_duet_attribute_reliability_proxy import analyze


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_torch_training_target_matches_locked_numpy_audit_formula() -> None:
    task_probability = np.array(
        [[0.70, 0.20, 0.10], [0.05, 0.90, 0.05]], dtype=np.float64
    )
    clip_probability = np.array(
        [[0.20, 0.10, 0.70], [0.55, 0.40, 0.05]], dtype=np.float64
    )
    task_prediction = np.array([0, 1], dtype=np.int64)
    clip_prediction = np.array([2, 0], dtype=np.int64)
    margin = np.linspace(-0.03, 0.04, 16, dtype=np.float64).reshape(2, 2, 4)
    numpy_result = entropy_anchored_attribute_mass(
        task_probability,
        clip_probability,
        task_prediction,
        clip_prediction,
        margin,
        clip_logit_scale=100.0,
    )
    torch_result = entropy_anchored_attribute_target(
        torch.from_numpy(task_probability),
        torch.from_numpy(clip_probability),
        torch.from_numpy(task_prediction),
        torch.from_numpy(clip_prediction),
        torch.from_numpy(margin),
        clip_logit_scale=100.0,
    )

    for key in (
        "probability",
        "attribute_mean_margin",
        "clip_pair_fraction",
        "task_pair_fraction",
        "clip_pair_entropy",
        "task_pair_entropy",
        "attribute_weight",
        "anchored_fraction",
        "pair_mass",
    ):
        np.testing.assert_allclose(
            torch_result[key].numpy(),
            numpy_result[key],
            atol=1e-12,
            rtol=1e-12,
        )


def test_pairwise_attribute_margin_uses_only_two_candidate_descriptions() -> None:
    image = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text = torch.zeros(3, 2, 4, 2)
    text[0, :, :, 0] = 1.0
    text[1, :, :, 1] = 1.0
    text[2, :, :, 0] = -1.0
    margin = pairwise_attribute_margin(
        image,
        text,
        task_prediction=torch.tensor([0, 1]),
        clip_prediction=torch.tensor([2, 0]),
    )

    np.testing.assert_allclose(margin[0].numpy(), np.full((2, 4), 2.0))
    np.testing.assert_allclose(margin[1].numpy(), np.full((2, 4), 1.0))


def _summary(accuracy: float, class_accuracy: list[float]) -> dict:
    return {
        "num_checkpoints": 16,
        "final": {
            "accuracy": accuracy,
            "class_accuracy": class_accuracy,
            "cycle": 4,
        },
    }


def test_proxy_gate_requires_macro_hard_and_other9_improvement() -> None:
    control_classes = [80.0] * 12
    control = _summary(80.0, control_classes)
    passing = _summary(80.3, [80.3] * 12)
    exchanged = [80.3] * 12
    for index in (0, 1, 2, 4, 5, 6, 8, 9, 10):
        exchanged[index] = 79.9
    failing = _summary(80.3, exchanged)

    assert analyze(control, passing)["decision"] == (
        "PASS_ATTRIBUTE_RELIABILITY_PROXY_GATE"
    )
    report = analyze(control, failing)
    assert report["decision"] == "REJECT_ATTRIBUTE_RELIABILITY_PROXY"
    assert not report["checks"]["other_nine_mean_noninferior"]


def test_candidate_changes_only_first_cycle_conflict_kl_target() -> None:
    plmatch = (REPO_ROOT / "src/methods/oh/plmatch.py").read_text()
    wrapper = (
        REPO_ROOT / "src/methods/oh/duet_attribute_reliability_kl.py"
    ).read_text()
    runner = (
        REPO_ROOT / "tools/run_visda_duet_attribute_reliability_proxy25.sh"
    ).read_text()

    assert "attribute_reliability_kl=False" in plmatch
    assert (
        "collect_attribute = bool(attribute_reliability_kl and curr_cycle == 0)"
        in plmatch
    )
    assert "active_conflict = (~label_mask) & (~matching_indices)" in plmatch
    assert 'kl_soft_output[active_conflict] = reliability["probability"]' in plmatch
    assert "clip_soft_batch = kl_soft[tar_idx]" in plmatch
    assert "attribute_reliability_kl=True" in wrapper
    assert "ACTIVE.CYCLE 4" in runner
    assert "run_visda_plmatch_proxy25_control.sh" not in runner
    assert "Even PASS does not authorize or start a full VisDA run" in runner
