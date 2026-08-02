from pathlib import Path

import numpy as np
import torch

from src.utils.support_conditioned_clip import (
    condition_clip_on_task_clip_top2_union,
)
from src.utils.support_conditioned_clip_audit import (
    support_conditioned_probability,
)
from tools.analyze_duet_support_conditioned_clip_proxy import analyze


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_torch_training_formula_matches_locked_numpy_audit() -> None:
    task = np.array(
        [[0.50, 0.30, 0.10, 0.10], [0.05, 0.60, 0.25, 0.10]],
        dtype=np.float64,
    )
    clip = np.array(
        [[0.10, 0.20, 0.60, 0.10], [0.55, 0.10, 0.05, 0.30]],
        dtype=np.float64,
    )
    task_top2 = np.argsort(-task, axis=1, kind="stable")[:, :2]
    clip_top2 = np.argsort(-clip, axis=1, kind="stable")[:, :2]
    support = np.zeros_like(task, dtype=bool)
    rows = np.arange(task.shape[0])[:, None]
    support[rows, task_top2] = True
    support[rows, clip_top2] = True

    expected = support_conditioned_probability(clip, support)
    actual = condition_clip_on_task_clip_top2_union(
        torch.from_numpy(task), torch.from_numpy(clip)
    )
    np.testing.assert_allclose(actual["probability"].numpy(), expected["probability"])
    np.testing.assert_array_equal(actual["support"].numpy(), support)
    np.testing.assert_array_equal(actual["probability"].argmax(1).numpy(), clip.argmax(1))


def test_support_conditioning_rejects_invalid_probabilities() -> None:
    task = torch.tensor([[0.7, 0.3]])
    clip = torch.tensor([[0.4, 0.7]])
    try:
        condition_clip_on_task_clip_top2_union(task, clip)
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("invalid CLIP probability was accepted")


def _summary(accuracy: float, class_accuracy: list[float]) -> dict:
    return {
        "num_checkpoints": 16,
        "final": {
            "accuracy": accuracy,
            "class_accuracy": class_accuracy,
            "cycle": 4,
        },
    }


def test_proxy_gate_requires_gain_and_limits_car_truck_exchange() -> None:
    control = _summary(87.93, [80.0] * 12)
    passing_classes = [80.3] * 12
    passing_classes[3] = 80.8
    passing_classes[11] = 79.7
    passing = analyze(control, _summary(88.23, passing_classes))
    assert passing["decision"] == "PASS_SUPPORT_CONDITIONED_CLIP_PROXY_GATE"
    assert passing["car_truck_exchange_observed"]

    failing_classes = [80.3] * 12
    failing_classes[3] = 81.2
    failing_classes[11] = 79.4
    failing = analyze(control, _summary(88.23, failing_classes))
    assert failing["decision"] == "REJECT_SUPPORT_CONDITIONED_CLIP_PROXY"
    assert not failing["checks"]["individual_car_truck_regression_at_most_0.50pp"]


def test_candidate_changes_only_first_cycle_conflict_kl_target() -> None:
    plmatch = (REPO_ROOT / "src/methods/oh/plmatch.py").read_text()
    wrapper = (
        REPO_ROOT / "src/methods/oh/duet_support_conditioned_clip.py"
    ).read_text()
    runner = (
        REPO_ROOT / "tools/run_visda_duet_support_conditioned_clip_proxy25.sh"
    ).read_text()

    assert "support_conditioned_clip=False" in plmatch
    assert (
        "support_conditioned_clip and curr_cycle == 0" in plmatch
    )
    assert "active_conflict = (~label_mask) & (~matching_indices)" in plmatch
    assert 'kl_soft_output[active_conflict] = conditioned["probability"]' in plmatch
    assert "clip_soft_batch = kl_soft[tar_idx]" in plmatch
    assert "support_conditioned_clip=True" in wrapper
    assert "ACTIVE.CYCLE 4" in runner
    assert "run_visda_plmatch_proxy25_control.sh" not in runner
    assert "Even PASS does not authorize or start a full VisDA run" in runner
