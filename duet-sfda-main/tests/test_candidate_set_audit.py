from pathlib import Path

import numpy as np
import pytest

from src.utils.candidate_set_audit import (
    candidate_coverage,
    evaluate_candidate_set_gate,
    stable_topk,
    union_candidate_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stable_topk_breaks_ties_by_class_index() -> None:
    probability = np.array([[0.4, 0.4, 0.1, 0.1]])
    np.testing.assert_array_equal(stable_topk(probability, 3), [[0, 1, 2]])


def test_union_candidate_mask_contains_both_rankings_without_duplicates() -> None:
    task = np.array([[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]])
    clip = np.array([[0.2, 0.5, 0.3], [0.1, 0.3, 0.6]])
    top1 = union_candidate_mask(task, clip, k=1)
    top2 = union_candidate_mask(task, clip, k=2)
    np.testing.assert_array_equal(top1["set_size"], [2, 2])
    np.testing.assert_array_equal(top2["set_size"], [3, 3])
    assert np.all(~top1["union_mask"] | top2["union_mask"])


def test_candidate_coverage_reports_locked_set_membership() -> None:
    mask = np.array([[True, False, True], [False, True, True]])
    np.testing.assert_array_equal(
        candidate_coverage(mask, np.array([2, 0])), [True, False]
    )


def test_gate_requires_coverage_recovery_class_safety_and_compactness() -> None:
    passing = evaluate_candidate_set_gate(
        input_contract_valid=True,
        top2_coverage_pct=92.0,
        recovered_top1_misses_pct=70.0,
        minimum_class_coverage_pct=86.0,
        car_coverage_pct=91.0,
        truck_coverage_pct=90.0,
        mean_set_size=3.4,
    )
    broad_failure = evaluate_candidate_set_gate(
        input_contract_valid=True,
        top2_coverage_pct=92.0,
        recovered_top1_misses_pct=70.0,
        minimum_class_coverage_pct=86.0,
        car_coverage_pct=91.0,
        truck_coverage_pct=90.0,
        mean_set_size=3.6,
    )
    truck_failure = evaluate_candidate_set_gate(
        input_contract_valid=True,
        top2_coverage_pct=92.0,
        recovered_top1_misses_pct=70.0,
        minimum_class_coverage_pct=86.0,
        car_coverage_pct=91.0,
        truck_coverage_pct=89.9,
        mean_set_size=3.4,
    )
    assert passing["decision"] == "PASS_CANDIDATE_SET_PREFLIGHT"
    assert passing["training_authorized"] is False
    assert broad_failure["decision"] == "REJECT"
    assert not broad_failure["checks"]["mean_candidate_set_size_at_most_3_5"]
    assert truck_failure["decision"] == "REJECT"
    assert not truck_failure["checks"]["truck_top2_coverage_at_least_90pct"]


def test_probability_validation_rejects_bad_rows_and_bad_k() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        stable_topk(np.array([[0.8, 0.8]]), 1)
    with pytest.raises(ValueError, match="inside the class range"):
        stable_topk(np.array([[0.5, 0.5]]), 3)


def test_cloud_entrypoint_is_cpu_only_and_locks_before_labels() -> None:
    runner = (REPO_ROOT / "tools/run_visda_conflict_candidate_set_audit.sh").read_text()
    audit = (REPO_ROOT / "tools/audit_visda_conflict_candidate_set.py").read_text()
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
        "target_list_sha256 = _sha256(args.target_list)"
    )
    assert audit.index("lock_path.write_text") < audit.index(
        "_parse_labels_after_lock(args.target_list)"
    )
