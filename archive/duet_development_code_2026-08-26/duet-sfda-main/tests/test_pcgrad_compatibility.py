from pathlib import Path

import numpy as np
import torch

from src.utils.pcgrad_compatibility import (
    compatibility_fraction_from_norms,
    merge_compatible_parameter_correction_,
    reconstruct_fractional_metrics,
)
from tools.analyze_duet_pcgrad_compatibility_proxy import analyze


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_projection_fraction_is_parameter_free_and_clipped() -> None:
    baseline = np.array([1.0, 1.0, 1.0])
    correction = np.array([1.0, 1.0, 1.0])
    # Corresponding dots are -0.5, 0.5, and 2.0.
    candidate = np.sqrt(np.array([1.0, 3.0, 6.0]))
    result = compatibility_fraction_from_norms(
        baseline, candidate, correction
    )
    np.testing.assert_allclose(
        result["baseline_correction_dot"], [-0.5, 0.5, 2.0]
    )
    np.testing.assert_allclose(result["fraction"], [0.0, 0.5, 1.0])


def test_fractional_metric_reconstruction_preserves_endpoints() -> None:
    baseline_norm = np.array([2.0, 2.0])
    correction_norm = np.array([1.0, 1.0])
    dot = np.array([-0.5, 0.5])
    candidate_norm = np.sqrt(
        baseline_norm**2 + correction_norm**2 + 2.0 * dot
    )
    result = reconstruct_fractional_metrics(
        fraction=np.array([0.0, 1.0]),
        baseline_norm=baseline_norm,
        candidate_norm=candidate_norm,
        correction_norm=correction_norm,
        baseline_unit_projection=np.array([1.0, 1.0]),
        candidate_unit_projection=np.array([1.2, 1.3]),
        baseline_first_order=np.array([4.0, 5.0]),
        candidate_first_order=np.array([4.5, 6.0]),
    )
    np.testing.assert_allclose(result["norm"], [2.0, candidate_norm[1]])
    np.testing.assert_allclose(result["oracle_unit_projection"], [1.0, 1.3])
    np.testing.assert_allclose(result["first_order"], [4.0, 6.0])
    np.testing.assert_allclose(result["cosine"], [0.5, 1.3 / candidate_norm[1]])


def test_parameter_merge_uses_exact_nonnegative_projection_fraction() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([1.0, 0.0])
    result = merge_compatible_parameter_correction_(
        (parameter,), (torch.tensor([1.0, 1.0]),)
    )
    assert result["fraction"] == 0.5
    torch.testing.assert_close(parameter.grad, torch.tensor([1.5, 0.5]))

    parameter.grad = torch.tensor([1.0, 0.0])
    rejected = merge_compatible_parameter_correction_(
        (parameter,), (torch.tensor([-1.0, 1.0]),)
    )
    assert rejected["fraction"] == 0.0
    torch.testing.assert_close(parameter.grad, torch.tensor([1.0, 0.0]))


def _summary(accuracy: float, classes: list[float]) -> dict:
    return {
        "num_checkpoints": 16,
        "final": {"cycle": 4, "accuracy": accuracy, "class_accuracy": classes},
    }


def test_proxy_gate_keeps_matched_final_only_contract() -> None:
    control = _summary(87.93, [80.0] * 12)
    candidate = _summary(88.23, [80.3] * 12)
    report = analyze(control, candidate)
    assert report["decision"] == "PASS_SUPPORT_CONDITIONED_CLIP_PROXY_GATE"
    assert report["checks"]["matched_four_cycle_contract"] is True


def test_training_and_runner_are_cycle2_only_and_label_free() -> None:
    trainer = (REPO_ROOT / "src/methods/oh/plmatch.py").read_text()
    helper = (REPO_ROOT / "src/utils/pcgrad_compatibility.py").read_text()
    wrapper = (
        REPO_ROOT / "src/methods/oh/duet_pcgrad_compatibility.py"
    ).read_text()
    entrypoint = (REPO_ROOT / "image_target_of_oh_vs.py").read_text()
    audit = (
        REPO_ROOT / "tools/audit_visda_pcgrad_compatibility.py"
    ).read_text()
    runner = (
        REPO_ROOT / "tools/run_visda_duet_pcgrad_compatibility_proxy25.sh"
    ).read_text()
    assert "pcgrad_compatibility and curr_cycle == 1" in trainer
    assert "pcgrad_compatibility=True" in wrapper
    assert 'startswith("duet_pcgrad_compatibility_")' in entrypoint
    assert "target_labels" not in helper
    assert audit.index("lock_path.write_text") < audit.index(
        "# Oracle diagnostic begins"
    )
    assert "ACTIVE.CYCLE 4" in runner
    assert "run_visda_plmatch_proxy25_control.sh" not in runner
    assert "seed_sweep_forbidden" in runner
