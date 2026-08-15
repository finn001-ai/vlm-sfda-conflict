import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.duet_cycle_checkpoint import (
    capture_process_rng_state,
    load_cycle_checkpoint,
    restore_process_rng_state,
    save_cycle_checkpoint,
    validate_cycle_checkpoint_contract,
)


def test_cycle_checkpoint_round_trip_and_refuses_overwrite(tmp_path):
    checkpoint = tmp_path / "cycle1.pt"
    payload = {
        "contract": {"seed": 2020, "proxy": "abc"},
        "completed_cycles": 1,
        "weights": torch.tensor([1.0, 2.0]),
        "rng_state": capture_process_rng_state(),
    }
    saved = save_cycle_checkpoint(str(checkpoint), payload)
    loaded = load_cycle_checkpoint(str(checkpoint))

    assert saved == checkpoint
    assert loaded["completed_cycles"] == 1
    assert torch.equal(loaded["weights"], payload["weights"])
    assert loaded["contract"] == payload["contract"]
    assert not Path(str(checkpoint) + ".tmp").exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_cycle_checkpoint(str(checkpoint), payload)


def test_cycle_checkpoint_restores_all_cpu_process_rngs():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = capture_process_rng_state()
    expected = (
        random.random(),
        float(np.random.rand()),
        torch.rand(4),
    )
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    restore_process_rng_state(state)
    actual = (
        random.random(),
        float(np.random.rand()),
        torch.rand(4),
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_cycle_checkpoint_contract_reports_changed_settings():
    validate_cycle_checkpoint_contract(
        {"seed": 2020, "proxy": "abc"},
        {"seed": 2020, "proxy": "abc"},
    )
    with pytest.raises(ValueError, match="proxy.*saved='abc'.*current='def'"):
        validate_cycle_checkpoint_contract(
            {"seed": 2020, "proxy": "abc"},
            {"seed": 2020, "proxy": "def"},
        )


def test_context_method_and_proxy_runner_save_every_required_cycle1_state():
    method = Path(
        "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
    ).read_text()
    for state_name in (
        '"netF"',
        '"netB"',
        '"netC"',
        '"optimizer"',
        '"clip_visual"',
        '"clip_optimizer"',
        '"prev_label_mask"',
        '"q_value"',
        '"context_comparator"',
        '"context_comparator_optimizer"',
        '"context_replay_memory"',
        '"rng_state"',
    ):
        assert state_name in method

    runner = Path(
        "tools/run_visda_real_conflict_gt_feature_probe.sh"
    ).read_text()
    assert "duet_fcp_context_visda_proxy25_seed2020_cycle1.pt" in runner
    assert "CYCLE_CHECKPOINT_RESUME_PATH" in runner
    assert "CYCLE_CHECKPOINT_SAVE_PATH" in runner
    assert "expected_task_checkpoints=4" in runner
    assert "expected_task_checkpoints=8" in runner


def test_real_multiview_runner_dispatches_to_checkpoint_aware_method():
    runner = Path(
        "tools/run_visda_real_multiview_comparator_proxy25.sh"
    ).read_text()
    assert (
        'method="duet_first_cycle_prior_context_transformer_real_multiview_'
        'visda_proxy25_seed${seed}"' in runner
    )
    assert "duet_first_cycle_prior_context_transformer_*)" in runner
