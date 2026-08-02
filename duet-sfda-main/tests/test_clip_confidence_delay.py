from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from src.utils.clip_confidence_delay import (
    class_balanced_clip_confidence_delay,
)
from tools.analyze_duet_clip_confidence_delay_proxy import analyze


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_delay_selects_lowest_clip_confidence_per_pseudo_class() -> None:
    matching = torch.ones(20, dtype=torch.bool)
    prediction = torch.tensor([0] * 10 + [1] * 10)
    confidence = np.array(
        [0.90, 0.80, 0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
        * 2
    )
    probability = np.zeros((20, 2), dtype=np.float32)
    probability[:10, 0] = confidence[:10]
    probability[:10, 1] = 1.0 - confidence[:10]
    probability[10:, 1] = confidence[10:]
    probability[10:, 0] = 1.0 - confidence[10:]
    result = class_balanced_clip_confidence_delay(
        matching,
        prediction,
        torch.from_numpy(probability),
    )
    expected = torch.zeros(20, dtype=torch.bool)
    expected[[9, 19]] = True
    torch.testing.assert_close(result["delayed"], expected)
    assert result["counts_by_class"] == {0: 1, 1: 1}
    torch.testing.assert_close(result["retained_matching"], matching & ~expected)


def test_delay_rejects_fraction_tuning() -> None:
    matching = torch.tensor([True, True])
    prediction = torch.tensor([0, 1])
    probability = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    try:
        class_balanced_clip_confidence_delay(
            matching, prediction, probability, fraction=0.2
        )
    except ValueError as error:
        assert "locked to 0.10" in str(error)
    else:
        raise AssertionError("target-label tuning of the delay fraction was accepted")


def _summary(accuracy: float, classes: list[float]) -> dict:
    return {
        "num_checkpoints": 16,
        "final": {"accuracy": accuracy, "class_accuracy": classes, "cycle": 4},
    }


def test_proxy_gate_uses_final_checkpoint_and_hard_class_guards() -> None:
    control = _summary(87.93, [80.0] * 12)
    passing = analyze(control, _summary(88.23, [80.3] * 12))
    assert passing["decision"] == "PASS_CLIP_CONFIDENCE_DELAY_PROXY_GATE"
    assert passing["candidate"] == "cycle1_class_balanced_bottom10_clip_confidence_delay"

    failing_classes = [80.3] * 12
    failing_classes[11] = 79.4
    failing = analyze(control, _summary(88.23, failing_classes))
    assert failing["decision"] == "REJECT_CLIP_CONFIDENCE_DELAY_PROXY"


def test_training_and_runner_change_only_cycle1_admission() -> None:
    plmatch = (REPO_ROOT / "src/methods/oh/plmatch.py").read_text()
    wrapper = (
        REPO_ROOT / "src/methods/oh/duet_clip_confidence_delay.py"
    ).read_text()
    runner = (
        REPO_ROOT / "tools/run_visda_duet_clip_confidence_delay_proxy25.sh"
    ).read_text()
    entrypoint = (REPO_ROOT / "image_target_of_oh_vs.py").read_text()
    assert "clip_confidence_delay=False" in plmatch
    assert "clip_confidence_delay and curr_cycle == 0" in plmatch
    assert "admission_matching" in plmatch
    assert "kl_soft_output = clip_all_output" in plmatch
    assert "clip_confidence_delay=True" in wrapper
    assert 'startswith("duet_clip_confidence_delay_")' in entrypoint
    assert "ACTIVE.CYCLE 4" in runner
    assert "wrong captured=197/399" in runner
    assert "Production selector exactly matches" in runner
    assert "run_visda_plmatch_proxy25_control.sh" not in runner
    assert "Even PASS does not authorize or start a full VisDA run" in runner


def test_proxy_analyzer_is_directly_executable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/analyze_duet_clip_confidence_delay_proxy.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
