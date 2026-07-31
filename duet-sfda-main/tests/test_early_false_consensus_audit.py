from __future__ import annotations

import numpy as np

from tools.analyze_early_false_consensus import analyze


def _snapshot(
    labels: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    cycle: int,
) -> dict[str, np.ndarray]:
    return {
        "cycle": np.array(cycle),
        "mix_label": labels,
        "label_mask": mask,
        "task_prob": probabilities,
        "strong_task_prob": probabilities.copy(),
        "target_label": targets,
        "sample_index": np.arange(labels.size),
    }


def test_audit_finds_stable_high_confidence_early_errors() -> None:
    sample_count = 100
    early_labels = np.zeros(sample_count, dtype=np.int64)
    targets = early_labels.copy()
    targets[:5] = 1
    mask = np.ones(sample_count, dtype=bool)

    early_prob = np.tile(np.array([[0.9, 0.1]]), (sample_count, 1))
    late_prob = early_prob.copy()
    late_prob[:5] = np.array([0.01, 0.99])
    late_prob[5:20] = np.array([0.2, 0.8])
    strong_prob = late_prob.copy()

    early = _snapshot(early_labels, mask, early_prob, targets, cycle=1)
    late = _snapshot(
        np.argmax(late_prob, axis=1),
        mask,
        late_prob,
        targets,
        cycle=3,
    )
    late["strong_task_prob"] = strong_prob

    report, arrays = analyze(
        early,
        late,
        loss_fraction=0.20,
        confidence_fraction=0.20,
        stability_fraction=1.0,
        min_suspicious_precision=0.20,
        min_error_enrichment=3.0,
        max_cut_ratio=0.20,
        min_after_cut_coverage=0.80,
    )

    assert report["verdict"] == "PASS"
    assert report["counts"]["suspicious_wrong"] == 5
    assert arrays["suspicious"][:5].all()


def test_audit_rejects_when_late_mask_breaks_monotonicity() -> None:
    labels = np.array([0, 1], dtype=np.int64)
    targets = labels.copy()
    early_mask = np.array([True, True])
    late_mask = np.array([True, False])
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])
    early = _snapshot(labels, early_mask, probabilities, targets, cycle=1)
    late = _snapshot(labels, late_mask, probabilities, targets, cycle=3)

    try:
        analyze(
            early,
            late,
            loss_fraction=0.1,
            confidence_fraction=0.2,
            stability_fraction=0.2,
            min_suspicious_precision=0.7,
            min_error_enrichment=3.0,
            max_cut_ratio=0.01,
            min_after_cut_coverage=0.955,
        )
    except ValueError as error:
        assert "monotonic" in str(error)
    else:
        raise AssertionError("Expected monotonicity validation to fail")
