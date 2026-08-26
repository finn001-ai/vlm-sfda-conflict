#!/usr/bin/env python
"""Offline first-order boundary-distance audit for DUET VisDA conflicts.

The signal and its fixed top-20% selection are computed without target labels.
Target labels are revealed only afterwards for an explicitly oracle diagnostic.
This tool performs no adaptation and never updates model parameters.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import clip  # noqa: E402
from conf import cfg  # noqa: E402
from src.methods.oh.plmatch import clip_pre_text, data_load  # noqa: E402
from src.utils.conflict_boundary import (  # noqa: E402
    boundary_choice_and_separation,
    fixed_fraction_mask,
    paired_accuracy_bootstrap_ci,
    pairwise_first_order_boundary,
)
from src.utils.probability_fusion import (  # noqa: E402
    arithmetic_probability_fusion,
    rms_probability_fusion,
)
from tools.export_conflict_diagnostics import (  # noqa: E402
    _build_source_model,
    _load_class_names,
    _prepare_cfg,
)


TOP_FRACTION = 0.20
MIN_GAIN_PP = 2.0
BASELINE_TOLERANCE_PP = 0.05
BOOTSTRAP_REPEATS = 2_000
EXPECTED_VISDA = {
    "total_samples": 55_388,
    "agreement_samples": 27_165,
    "agreement_accuracy": 93.98,
    "task_accuracy": 51.45,
    "clip_accuracy": 82.87,
    "arithmetic_mix_accuracy": 73.83,
}


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _accuracy(prediction: np.ndarray, label: np.ndarray, mask: np.ndarray) -> float:
    return _pct(np.sum(prediction[mask] == label[mask]), np.sum(mask))


def _paired_comparison(
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    candidate_correct = candidate_prediction[mask] == label[mask]
    baseline_correct = baseline_prediction[mask] == label[mask]
    candidate_accuracy = _pct(candidate_correct.sum(), candidate_correct.size)
    baseline_accuracy = _pct(baseline_correct.sum(), baseline_correct.size)
    ci_low, ci_high = paired_accuracy_bootstrap_ci(
        candidate_correct,
        baseline_correct,
        repeats=BOOTSTRAP_REPEATS,
        seed=cfg.SETTING.SEED,
    )
    return {
        "candidate_accuracy": candidate_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "gain_pp": candidate_accuracy - baseline_accuracy,
        "paired_bootstrap_95_ci_pp": [ci_low, ci_high],
        "net_corrections": int(candidate_correct.sum() - baseline_correct.sum()),
    }


def _classwise_oracle(
    class_names: list[str],
    label: np.ndarray,
    selected: np.ndarray,
    boundary_prediction: np.ndarray,
    clip_prediction: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for class_index, class_name in enumerate(class_names):
        mask = selected & (label == class_index)
        count = int(mask.sum())
        if not count:
            rows.append(
                {
                    "class_index": class_index,
                    "class": class_name,
                    "selected": 0,
                    "boundary_accuracy": None,
                    "fixed_clip_accuracy": None,
                    "delta_pp": None,
                    "net_corrections": 0,
                }
            )
            continue
        boundary_correct = boundary_prediction[mask] == label[mask]
        clip_correct = clip_prediction[mask] == label[mask]
        boundary_accuracy = _pct(boundary_correct.sum(), count)
        clip_accuracy = _pct(clip_correct.sum(), count)
        rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "selected": count,
                "boundary_accuracy": boundary_accuracy,
                "fixed_clip_accuracy": clip_accuracy,
                "delta_pp": boundary_accuracy - clip_accuracy,
                "net_corrections": int(boundary_correct.sum() - clip_correct.sum()),
            }
        )
    return rows


def _baseline_reproduction(observed: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "full_target_count": observed["total_samples"] == EXPECTED_VISDA["total_samples"],
        "agreement_count": observed["agreement_samples"] == EXPECTED_VISDA["agreement_samples"],
        "full_adaptation_list": not bool(str(cfg.ACTIVE.ADAPTATION_LIST).strip()),
    }
    for metric in (
        "agreement_accuracy",
        "task_accuracy",
        "clip_accuracy",
        "arithmetic_mix_accuracy",
    ):
        checks[metric] = (
            abs(float(observed[metric]) - float(EXPECTED_VISDA[metric]))
            <= BASELINE_TOLERANCE_PP
        )
    return {
        "expected": EXPECTED_VISDA,
        "tolerance_pp": BASELINE_TOLERANCE_PP,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty audit CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    observed = summary["baseline_observed"]
    conflict = summary["conflict_oracle_diagnostic"]
    clip_comparison = summary["gate"]["versus_fixed_clip"]
    confidence_comparison = summary["gate"]["versus_higher_confidence"]
    lines = [
        "# VisDA DUET Conflict Boundary-Distance Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "Target labels are used only in the oracle-diagnostic sections below.",
        "The boundary score and fixed top-20% selection are label-free and locked first.",
        "",
        "## Baseline reproduction",
        "",
        "| Metric | Observed |",
        "|---|---:|",
        f"| Total samples | {observed['total_samples']} |",
        f"| Agreement samples | {observed['agreement_samples']} |",
        f"| Agreement accuracy | {observed['agreement_accuracy']:.4f}% |",
        f"| Task accuracy | {observed['task_accuracy']:.4f}% |",
        f"| CLIP accuracy | {observed['clip_accuracy']:.4f}% |",
        f"| Arithmetic mix accuracy | {observed['arithmetic_mix_accuracy']:.4f}% |",
        f"| Normalized RMS accuracy | {observed['rms_accuracy']:.4f}% |",
        "",
        "## Conflict oracle diagnostic",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Conflict samples | {conflict['samples']} |",
        f"| Candidate coverage | {conflict['candidate_coverage']:.4f}% |",
        f"| Selected samples | {conflict['selected_samples']} |",
        f"| Selected coverage | {conflict['selected_coverage']:.4f}% |",
        f"| Boundary selector accuracy | {conflict['selected_boundary_accuracy']:.4f}% |",
        f"| Fixed CLIP accuracy | {conflict['selected_fixed_clip_accuracy']:.4f}% |",
        f"| Higher-confidence accuracy | {conflict['selected_higher_confidence_accuracy']:.4f}% |",
        "",
        "## Predeclared gate",
        "",
        f"- Gain over fixed CLIP: `{clip_comparison['gain_pp']:.4f} pp`; "
        f"95% CI `{clip_comparison['paired_bootstrap_95_ci_pp']}`.",
        f"- Gain over higher confidence: `{confidence_comparison['gain_pp']:.4f} pp`; "
        f"95% CI `{confidence_comparison['paired_bootstrap_95_ci_pp']}`.",
        f"- Macro class delta vs fixed CLIP: `{summary['gate']['macro_class_delta_pp']:.4f} pp`.",
        f"- Car net corrections: `{summary['gate']['car_net_corrections']}`.",
        f"- Truck net corrections: `{summary['gate']['truck_net_corrections']}`.",
        "",
        "Passing this audit does not authorize adaptation training.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    _prepare_cfg()
    if cfg.SETTING.DATASET != "VISDA-C":
        raise ValueError("This predeclared audit is restricted to VISDA-C")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VisDA boundary-distance audit")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    device = torch.device("cuda")
    torch.manual_seed(cfg.SETTING.SEED)
    torch.cuda.manual_seed(cfg.SETTING.SEED)

    class_names = _load_class_names()
    test_loader = data_load(cfg)["test_aug"]
    image_records = test_loader.dataset.imgs
    net_f, net_b, net_c = _build_source_model(device)
    clip_model, _, _ = clip.load(cfg.ACTIVE.ARCH, device=device)
    clip_model.float()
    for module in (net_f, net_b, net_c, clip_model):
        _freeze(module)

    text_inputs = clip_pre_text(cfg).to(device)
    with torch.no_grad():
        text_features = F.normalize(clip_model.encode_text(text_inputs), dim=1).detach()

    full_batches: dict[str, list[np.ndarray]] = {
        "index": [],
        "label": [],
        "task_pred": [],
        "clip_pred": [],
        "arithmetic_pred": [],
        "rms_pred": [],
        "task_conf": [],
        "clip_conf": [],
    }
    signal_rows: list[dict[str, Any]] = []

    for inputs, labels, indices in test_loader:
        weak_x = inputs[1].to(device)
        labels_np = labels.long().cpu().numpy()
        indices_cpu = indices.long().cpu()

        with torch.no_grad():
            task_logits = net_c(net_b(net_f(weak_x)))
            clip_features = F.normalize(clip_model.encode_image(weak_x), dim=1)
            clip_logits = clip_model.logit_scale.exp() * clip_features @ text_features.t()
            task_prob = torch.softmax(task_logits, dim=1)
            clip_prob = torch.softmax(clip_logits, dim=1)
            task_conf, task_pred = task_prob.max(dim=1)
            clip_conf, clip_pred = clip_prob.max(dim=1)
            arithmetic_pred = arithmetic_probability_fusion(task_prob, clip_prob).argmax(dim=1)
            rms_pred = rms_probability_fusion(task_prob, clip_prob).argmax(dim=1)

        for name, value in (
            ("index", indices_cpu.numpy()),
            ("label", labels_np),
            ("task_pred", task_pred.cpu().numpy()),
            ("clip_pred", clip_pred.cpu().numpy()),
            ("arithmetic_pred", arithmetic_pred.cpu().numpy()),
            ("rms_pred", rms_pred.cpu().numpy()),
            ("task_conf", task_conf.cpu().numpy()),
            ("clip_conf", clip_conf.cpu().numpy()),
        ):
            full_batches[name].append(value)

        conflict_positions = torch.nonzero(task_pred != clip_pred, as_tuple=False).flatten()
        if conflict_positions.numel() == 0:
            continue

        conflict_task_pred = task_pred[conflict_positions]
        conflict_clip_pred = clip_pred[conflict_positions]

        task_x = weak_x[conflict_positions].detach().requires_grad_(True)
        task_boundary_logits = net_c(net_b(net_f(task_x)))
        task_radius, task_pair_margin, task_gradient_norm = pairwise_first_order_boundary(
            task_boundary_logits,
            task_x,
            conflict_task_pred,
            conflict_clip_pred,
        )
        del task_boundary_logits, task_x

        clip_x = weak_x[conflict_positions].detach().requires_grad_(True)
        clip_boundary_features = F.normalize(clip_model.encode_image(clip_x), dim=1)
        clip_boundary_logits = (
            clip_model.logit_scale.exp() * clip_boundary_features @ text_features.t()
        )
        clip_radius, clip_pair_margin, clip_gradient_norm = pairwise_first_order_boundary(
            clip_boundary_logits,
            clip_x,
            conflict_clip_pred,
            conflict_task_pred,
        )

        for local_position, batch_position in enumerate(conflict_positions.tolist()):
            index = int(indices_cpu[batch_position].item())
            task_label = int(task_pred[batch_position].item())
            clip_label = int(clip_pred[batch_position].item())
            signal_rows.append(
                {
                    "index": index,
                    "path": image_records[index][0],
                    "task_pred": task_label,
                    "task_pred_name": class_names[task_label],
                    "clip_pred": clip_label,
                    "clip_pred_name": class_names[clip_label],
                    "task_conf": float(task_conf[batch_position].item()),
                    "clip_conf": float(clip_conf[batch_position].item()),
                    "task_pair_margin": float(task_pair_margin[local_position].item()),
                    "clip_pair_margin": float(clip_pair_margin[local_position].item()),
                    "task_gradient_norm": float(task_gradient_norm[local_position].item()),
                    "clip_gradient_norm": float(clip_gradient_norm[local_position].item()),
                    "task_boundary_radius": float(task_radius[local_position].item()),
                    "clip_boundary_radius": float(clip_radius[local_position].item()),
                }
            )

    full = {name: np.concatenate(parts) for name, parts in full_batches.items()}
    full_order = np.argsort(full["index"])
    full = {name: value[full_order] for name, value in full.items()}
    signal_rows.sort(key=lambda row: row["index"])

    task_radius = torch.tensor(
        [row["task_boundary_radius"] for row in signal_rows], dtype=torch.float64
    )
    clip_radius = torch.tensor(
        [row["clip_boundary_radius"] for row in signal_rows], dtype=torch.float64
    )
    choose_task, separation = boundary_choice_and_separation(task_radius, clip_radius)
    selected = fixed_fraction_mask(separation, TOP_FRACTION)
    choose_task_np = choose_task.numpy()
    separation_np = separation.numpy()
    selected_np = selected.numpy()

    conflict_index = np.array([row["index"] for row in signal_rows], dtype=np.int64)
    if not np.array_equal(full["index"][conflict_index], conflict_index):
        raise RuntimeError("Target indices are not the expected contiguous VisDA ordering")
    label = full["label"][conflict_index]
    task_prediction = full["task_pred"][conflict_index]
    clip_prediction = full["clip_pred"][conflict_index]
    rms_prediction = full["rms_pred"][conflict_index]
    confidence_prediction = np.where(
        full["task_conf"][conflict_index] >= full["clip_conf"][conflict_index],
        task_prediction,
        clip_prediction,
    )
    boundary_prediction = np.where(choose_task_np, task_prediction, clip_prediction)

    for position, row in enumerate(signal_rows):
        row["boundary_choice"] = "task" if choose_task_np[position] else "clip"
        row["boundary_separation"] = float(separation_np[position])
        row["selected_top20"] = bool(selected_np[position])
        row["higher_confidence_choice"] = (
            "task"
            if full["task_conf"][row["index"]] >= full["clip_conf"][row["index"]]
            else "clip"
        )

    oracle_rows = []
    for position, row in enumerate(signal_rows):
        true_label = int(label[position])
        oracle_rows.append(
            {
                "index": row["index"],
                "label": true_label,
                "label_name": class_names[true_label],
                "selected_top20": bool(selected_np[position]),
                "candidate_contains_label": bool(
                    true_label in (int(task_prediction[position]), int(clip_prediction[position]))
                ),
                "task_correct": bool(task_prediction[position] == true_label),
                "clip_correct": bool(clip_prediction[position] == true_label),
                "boundary_correct": bool(boundary_prediction[position] == true_label),
                "higher_confidence_correct": bool(confidence_prediction[position] == true_label),
                "rms_correct": bool(rms_prediction[position] == true_label),
            }
        )

    all_label = full["label"]
    agreement = full["task_pred"] == full["clip_pred"]
    baseline_observed = {
        "total_samples": int(all_label.size),
        "agreement_samples": int(agreement.sum()),
        "agreement_accuracy": _accuracy(full["task_pred"], all_label, agreement),
        "task_accuracy": _accuracy(full["task_pred"], all_label, np.ones_like(agreement)),
        "clip_accuracy": _accuracy(full["clip_pred"], all_label, np.ones_like(agreement)),
        "arithmetic_mix_accuracy": _accuracy(
            full["arithmetic_pred"], all_label, np.ones_like(agreement)
        ),
        "rms_accuracy": _accuracy(full["rms_pred"], all_label, np.ones_like(agreement)),
    }
    reproduction = _baseline_reproduction(baseline_observed)

    selected_mask = selected_np.astype(bool)
    clip_comparison = _paired_comparison(
        boundary_prediction, clip_prediction, label, selected_mask
    )
    confidence_comparison = _paired_comparison(
        boundary_prediction, confidence_prediction, label, selected_mask
    )
    classwise = _classwise_oracle(
        class_names, label, selected_mask, boundary_prediction, clip_prediction
    )
    valid_class_deltas = [row["delta_pp"] for row in classwise if row["delta_pp"] is not None]
    macro_class_delta = float(np.mean(valid_class_deltas))
    by_name = {str(row["class"]).strip().lower(): row for row in classwise}
    car_net = int(by_name.get("car", {}).get("net_corrections", -10**9))
    truck_net = int(by_name.get("truck", {}).get("net_corrections", -10**9))

    gate_checks = {
        "baseline_reproduced": bool(reproduction["passed"]),
        "gain_vs_fixed_clip_at_least_2pp": clip_comparison["gain_pp"] >= MIN_GAIN_PP,
        "gain_vs_higher_confidence_at_least_2pp": (
            confidence_comparison["gain_pp"] >= MIN_GAIN_PP
        ),
        "fixed_clip_ci_lower_positive": (
            clip_comparison["paired_bootstrap_95_ci_pp"][0] > 0.0
        ),
        "higher_confidence_ci_lower_positive": (
            confidence_comparison["paired_bootstrap_95_ci_pp"][0] > 0.0
        ),
        "macro_class_delta_positive": macro_class_delta > 0.0,
        "car_net_corrections_nonnegative": car_net >= 0,
        "truck_net_corrections_nonnegative": truck_net >= 0,
    }
    decision = "PASS_OFFLINE_GATE" if all(gate_checks.values()) else "REJECT"

    summary = {
        "dataset": cfg.SETTING.DATASET,
        "task": f"{cfg.domain[cfg.SETTING.S]}->{cfg.domain[cfg.SETTING.T]}",
        "seed": int(cfg.SETTING.SEED),
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal": {
            "name": "gradient_normalized_pairwise_boundary_distance",
            "selection_fraction": TOP_FRACTION,
            "selection_rule": "top absolute log radius ratio; choose larger radius",
            "input_space": "shared normalized weak-view tensor",
        },
        "baseline_observed": baseline_observed,
        "baseline_reproduction": reproduction,
        "conflict_oracle_diagnostic": {
            "samples": int(label.size),
            "rate": _pct(label.size, all_label.size),
            "candidate_coverage": _pct(
                np.sum((label == task_prediction) | (label == clip_prediction)), label.size
            ),
            "fixed_task_accuracy": _pct(np.sum(task_prediction == label), label.size),
            "fixed_clip_accuracy": _pct(np.sum(clip_prediction == label), label.size),
            "higher_confidence_accuracy": _pct(
                np.sum(confidence_prediction == label), label.size
            ),
            "rms_accuracy": _pct(np.sum(rms_prediction == label), label.size),
            "boundary_accuracy": _pct(np.sum(boundary_prediction == label), label.size),
            "selected_samples": int(selected_mask.sum()),
            "selected_coverage": _pct(selected_mask.sum(), label.size),
            "selected_boundary_accuracy": _accuracy(
                boundary_prediction, label, selected_mask
            ),
            "selected_fixed_clip_accuracy": _accuracy(
                clip_prediction, label, selected_mask
            ),
            "selected_higher_confidence_accuracy": _accuracy(
                confidence_prediction, label, selected_mask
            ),
        },
        "gate": {
            "minimum_gain_pp": MIN_GAIN_PP,
            "versus_fixed_clip": clip_comparison,
            "versus_higher_confidence": confidence_comparison,
            "macro_class_delta_pp": macro_class_delta,
            "car_net_corrections": car_net,
            "truck_net_corrections": truck_net,
            "checks": gate_checks,
        },
        "classwise_oracle_diagnostic": classwise,
        "decision": decision,
        "training_authorized": False,
    }

    out_dir = Path(cfg.output_dir) / "boundary_distance_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "visda_conflict_boundary_distance"
    signal_path = out_dir / f"{stem}_signals.csv"
    oracle_path = out_dir / f"{stem}_oracle_diagnostic.csv"
    summary_path = out_dir / f"{stem}_summary.json"
    markdown_path = out_dir / f"{stem}_summary.md"
    _write_csv(signal_path, signal_rows)
    _write_csv(oracle_path, oracle_rows)
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_markdown(summary, markdown_path)

    print(f"Wrote label-free signals: {signal_path}")
    print(f"Wrote oracle diagnostic: {oracle_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote Markdown report: {markdown_path}")
    print(json.dumps({"decision": decision, "gate": summary["gate"]}, indent=2))


if __name__ == "__main__":
    main()
