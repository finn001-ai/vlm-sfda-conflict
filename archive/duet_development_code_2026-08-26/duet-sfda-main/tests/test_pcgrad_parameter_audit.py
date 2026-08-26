import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from src.utils.pcgrad_parameter_audit import (
    AUDIT_BATCH_COUNT,
    AUDIT_BATCH_SIZE,
    AUDITED_CONFLICTS,
    build_locked_parameter_audit_batches,
    evaluate_exact_parameter_gate,
    symmetric_pcgrad_output_correction,
)
from src.utils.pcgrad_parameter_runtime import run_exact_pcgrad_parameter_audit


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_locked_batches_are_label_free_unique_and_cover_five_percent() -> None:
    conflict = np.arange(1_978)
    admitted = np.arange(2_000, 2_600)
    batches, mask = build_locked_parameter_audit_batches(conflict, admitted)
    assert batches.shape == (AUDIT_BATCH_COUNT, AUDIT_BATCH_SIZE)
    assert mask.shape == batches.shape
    assert int(mask.sum()) == AUDITED_CONFLICTS
    assert np.unique(batches).size == batches.size
    assert set(batches[mask]).issubset(set(conflict))
    assert AUDITED_CONFLICTS / conflict.size * 100.0 >= 5.0


def test_symmetric_output_pcgrad_changes_only_negative_conflicts() -> None:
    first = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    second = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [1.0, 1.0]])
    conflict = torch.tensor([True, False, True])
    result = symmetric_pcgrad_output_correction(first, second, conflict)
    assert result["active"].tolist() == [True, False, False]
    assert torch.linalg.norm(result["correction"][0]) > 0
    assert torch.equal(result["correction"][1:], torch.zeros(2, 2))


def _comparison(low: float = 0.01) -> dict:
    return {
        "mean_difference": 0.02,
        "paired_bootstrap_95_ci": [low, 0.03],
    }


def test_exact_parameter_gate_authorizes_only_one_proxy() -> None:
    gate = evaluate_exact_parameter_gate(
        input_contract_valid=True,
        cycle1_max_accuracy_error_pp=0.06,
        audited_conflict_coverage_pct=5.05,
        output_active_coverage_pct=30.0,
        comparisons={
            "cosine": _comparison(),
            "oracle_unit_projection": _comparison(),
            "first_order": _comparison(),
        },
        baseline_negative_burden=-0.2,
        candidate_negative_burden=-0.1,
        helpful_retention_pct=101.0,
        mean_norm_ratio=1.02,
        positive_batch_fraction_pct=90.0,
        group_first_order_delta={
            "car": 0.01,
            "person": 0.01,
            "truck": 0.02,
            "other_nine": 0.01,
        },
    )
    assert gate["decision"] == "PASS_EXACT_PARAMETER_PREFLIGHT"
    assert gate["matched_proxy_authorized"] is True
    assert gate["full_training_authorized"] is False
    assert gate["seed_sweep_authorized"] is False

    rejected = evaluate_exact_parameter_gate(
        input_contract_valid=True,
        cycle1_max_accuracy_error_pp=0.11,
        audited_conflict_coverage_pct=5.05,
        output_active_coverage_pct=30.0,
        comparisons={
            "cosine": _comparison(-0.01),
            "oracle_unit_projection": _comparison(),
            "first_order": _comparison(),
        },
        baseline_negative_burden=-0.2,
        candidate_negative_burden=-0.1,
        helpful_retention_pct=101.0,
        mean_norm_ratio=1.02,
        positive_batch_fraction_pct=90.0,
        group_first_order_delta={
            "car": 0.01,
            "person": 0.01,
            "truck": -0.02,
            "other_nine": 0.01,
        },
    )
    assert rejected["decision"] == "REJECT"


class _TinyDataset:
    def __init__(self, size: int) -> None:
        self.imgs = [(f"fake-{index}", index % 12) for index in range(size)]
        self.transform = self._transform

    def __len__(self) -> int:
        return len(self.imgs)

    @staticmethod
    def loader(path: str) -> float:
        return float(path.split("-")[1])

    @staticmethod
    def _transform(value: float) -> list[torch.Tensor]:
        base = torch.tensor(
            [value % 7, value % 11, value % 13, value % 17],
            dtype=torch.float32,
        ) / 17.0
        return [base, base + 0.01, base + 0.03]


def test_tiny_runtime_locks_each_batch_and_updates_no_parameter(tmp_path: Path) -> None:
    torch.manual_seed(2020)
    size = 700
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for suffix in ("F", "B", "C"):
        torch.save({"value": torch.tensor([1.0])}, source_dir / f"source_{suffix}.pt")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = SimpleNamespace(
        output_dir=str(run_dir),
        output_dir_src=str(source_dir),
        SETTING=SimpleNamespace(SEED=2020),
        ACTIVE=SimpleNamespace(CON_PAR=0.2, KL_PAR=0.4, CLS_PAR=0.4),
        TEST=SimpleNamespace(BATCH_SIZE=64),
        PCGRAD_PARAMETER_AUDIT=SimpleNamespace(
            DIR="conflict_pcgrad_parameter_audit"
        ),
    )
    netF = torch.nn.Linear(4, 4)
    netB = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 4))
    netC = torch.nn.Linear(4, 12)
    for parameter in netC.parameters():
        parameter.requires_grad = False
    before = [parameter.detach().clone() for model in (netF, netB) for parameter in model.parameters()]
    label_mask = torch.ones(size, dtype=torch.bool)
    label_mask[:120] = False
    source_label = torch.zeros(size, dtype=torch.long)
    clip_label = torch.zeros(size, dtype=torch.long)
    clip_label[:120] = 1
    logits = torch.randn(size, 12)
    kl_soft = torch.softmax(logits, dim=1)
    raw = run_exact_pcgrad_parameter_audit(
        cfg,
        netF=netF,
        netB=netB,
        netC=netC,
        target_dataset=_TinyDataset(size),
        mem_label=torch.arange(size) % 12,
        label_mask=label_mask,
        kl_soft=kl_soft,
        audit_payload={"source_label": source_label, "clip_label": clip_label},
    )
    after = [parameter.detach() for model in (netF, netB) for parameter in model.parameters()]
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    assert raw["cycle2_optimizer_steps"] == 0
    audit_dir = run_dir / "conflict_pcgrad_parameter_audit"
    assert len(list((audit_dir / "batch_signal_locks").glob("*.json"))) == 10
    assert (audit_dir / "visda_conflict_pcgrad_parameter_signal_lock.json").is_file()


def test_cloud_runner_is_single_audit_and_never_starts_proxy_or_full() -> None:
    runner = (REPO_ROOT / "tools/run_visda_conflict_pcgrad_parameter_audit.sh").read_text()
    trainer = (REPO_ROOT / "src/methods/oh/plmatch.py").read_text()
    runtime = (REPO_ROOT / "src/utils/pcgrad_parameter_runtime.py").read_text()
    assert "ACTIVE.CYCLE 2" in runner
    assert "PCGRAD_PARAMETER_AUDIT.ENABLED True" in runner
    assert "feature_jacobian" in runner
    assert "run_visda_plmatch_proxy25_control.sh" not in runner
    assert "optimizer.step" not in runtime
    assert runtime.index("batch_lock_path.write_text") < runtime.index(
        "_oracle_labels_after_lock(target_dataset, indices)"
    )
    assert trainer.index("run_exact_pcgrad_parameter_audit(") < trainer.index(
        "kl_soft = kl_soft.cuda()"
    )
    assert "cycle2_optimizer_steps=0" in trainer


def test_finalizer_emits_pass_only_for_matched_positive_fixture(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    lock_dir = audit_dir / "batch_signal_locks"
    lock_dir.mkdir(parents=True)
    selection_lock = audit_dir / "visda_conflict_pcgrad_parameter_signal_lock.json"
    selection_lock.write_text(json.dumps({"contains_target_labels": False}))
    selection_hash = hashlib.sha256(selection_lock.read_bytes()).hexdigest()
    raw = {
        "decision": "EXACT_PARAMETER_EVIDENCE_CAPTURED",
        "selection_signal_lock_sha256": selection_hash,
        "cycle2_optimizer_steps": 0,
        "parameters_updated_by_audit": False,
        "unresolved_conflicts": 1_978,
        "audited_conflicts": 100,
        "audited_conflict_coverage_pct": 100 / 1_978 * 100,
    }
    (audit_dir / "visda_conflict_pcgrad_parameter_runtime_raw.json").write_text(
        json.dumps(raw)
    )
    for index in range(10):
        (lock_dir / f"batch{index:02d}.json").write_text(
            json.dumps({"labels_read_after_this_manifest": True})
        )
    batch_path = audit_dir / "visda_conflict_pcgrad_parameter_oracle_diagnostic.csv"
    batch_rows = []
    for index in range(10):
        batch_rows.append(
            {
                "batch": index + 1,
                "samples": 64,
                "conflict_samples": 10,
                "output_pcgrad_active_conflicts": 3,
                "baseline_first_order": 1.0,
                "candidate_first_order": 1.2,
                "baseline_oracle_unit_projection": 0.5,
                "candidate_oracle_unit_projection": 0.6,
                "baseline_cosine": 0.4,
                "candidate_cosine": 0.5,
                "baseline_norm": 1.0,
                "candidate_norm": 1.02,
            }
        )
    with batch_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(batch_rows[0]))
        writer.writeheader()
        writer.writerows(batch_rows)
    group_path = audit_dir / "visda_conflict_pcgrad_parameter_groupwise_oracle_diagnostic.csv"
    group_rows = [
        {
            "batch": batch + 1,
            "group": group,
            "samples": 2,
            "candidate_minus_baseline_first_order": 0.1,
        }
        for batch in range(10)
        for group in ("car", "person", "truck", "other_nine")
    ]
    with group_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    checkpoints = "".join(
        f"Task: TV, Iter:{iteration}/868; Cycle: 1/2; Accuracy = {accuracy:.2f}%\n"
        for iteration, accuracy in ((217, 80.0), (434, 82.0), (651, 82.4), (868, 82.2))
    )
    control_log = tmp_path / "control.txt"
    control_log.write_text(checkpoints)
    audit_log = tmp_path / "audit.txt"
    audit_log.write_text(
        checkpoints
        + "PCGrad exact parameter audit stop: after_pre_cycle=2; cycle2_optimizer_steps=0; parameters_updated_by_audit=False\n"
    )
    control_summary = tmp_path / "control.json"
    control_summary.write_text(json.dumps({"final": {"accuracy": 87.93}}))
    feature_summary = tmp_path / "feature.json"
    feature_summary.write_text(
        json.dumps(
            {
                "decision": "NEEDS_EXACT_CONTROL_PARAMETER_AUDIT",
                "labels_used_only_after_signal_lock": True,
            }
        )
    )
    output = audit_dir / "visda_conflict_pcgrad_parameter_summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/finalize_visda_pcgrad_parameter_audit.py"),
            "--audit-dir",
            str(audit_dir),
            "--audit-log",
            str(audit_log),
            "--control-log",
            str(control_log),
            "--control-summary",
            str(control_summary),
            "--feature-summary",
            str(feature_summary),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(output.read_text())
    assert summary["decision"] == "PASS_EXACT_PARAMETER_PREFLIGHT"
    assert summary["matched_proxy_authorized"] is True
    assert summary["full_training_authorized"] is False
    assert "PASS_EXACT_PARAMETER_PREFLIGHT" in result.stdout
