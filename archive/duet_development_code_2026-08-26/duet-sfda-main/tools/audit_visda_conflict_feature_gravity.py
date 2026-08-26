#!/usr/bin/env python
"""Read-only VisDA preflight for feature-weighted DUET consistency.

The first phase ignores loader labels, computes all task/CLIP/augmentation
signals and logit-gradient components, writes label-free artifacts, and locks
them with SHA256 hashes. Only then does the oracle-diagnostic phase parse the
labels from the VisDA list. No optimizer is constructed and no parameter is
updated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
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
from src.utils.feature_gravity_audit import (  # noqa: E402
    binary_auroc,
    classwise_gradient_mass,
    duet_logit_descent_components,
    evaluate_preflight_gate,
    fixed_tail_masks,
    gradient_projection_summary,
    stratified_bootstrap_auc_difference,
)
from src.utils.probability_fusion import arithmetic_probability_fusion  # noqa: E402
from tools.export_conflict_diagnostics import (  # noqa: E402
    _build_source_model,
    _load_class_names,
    _prepare_cfg,
)


TAIL_FRACTION = 0.20
BOOTSTRAP_REPEATS = 1_000
WEIGHT_EFFECT_SIZE = 0.10
EXPECTED_VISDA = {
    "total_samples": 55_388,
    "agreement_samples": 27_165,
    "agreement_accuracy": 93.98,
    "task_accuracy": 51.45,
    "clip_accuracy": 82.87,
    "arithmetic_mix_accuracy": 73.83,
}
REPRODUCTION_TOLERANCES = {
    "agreement_count_fraction": 0.0025,
    "agreement_accuracy_pp": 0.25,
    "task_accuracy_pp": 0.10,
    "clip_accuracy_pp": 0.15,
    "arithmetic_mix_accuracy_pp": 0.10,
}


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_labels_after_lock(
    list_path: str | Path, ordered_image_paths: list[str]
) -> np.ndarray:
    """Parse oracle labels only after label-free artifacts have been locked."""
    labels_by_path: dict[str, int] = {}
    with Path(list_path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                image_path, label_text = stripped.rsplit(maxsplit=1)
                label = int(label_text)
            except ValueError as error:
                raise ValueError(
                    f"Malformed VisDA row {line_number} in {list_path}: {stripped}"
                ) from error
            if image_path in labels_by_path:
                raise ValueError(f"Duplicate image path in VisDA list: {image_path}")
            labels_by_path[image_path] = label

    missing = [path for path in ordered_image_paths if path not in labels_by_path]
    if missing:
        raise ValueError(f"VisDA list is missing audit path: {missing[0]}")
    if len(labels_by_path) != len(ordered_image_paths):
        raise ValueError(
            "The full-target audit requires the VisDA list and loader to contain "
            "the same number of unique rows"
        )
    return np.asarray([labels_by_path[path] for path in ordered_image_paths], dtype=np.int64)


def _accuracy(prediction: np.ndarray, label: np.ndarray, mask: np.ndarray) -> float:
    return _pct(np.sum(prediction[mask] == label[mask]), np.sum(mask))


def _baseline_reproduction(observed: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    expected_count = EXPECTED_VISDA["total_samples"]
    agreement_delta = abs(observed["agreement_samples"] - EXPECTED_VISDA["agreement_samples"])
    checks = {
        "full_target_count": observed["total_samples"] == expected_count,
        "contiguous_unique_indices": np.array_equal(indices, np.arange(indices.size)),
        "full_adaptation_list": not bool(str(cfg.ACTIVE.ADAPTATION_LIST).strip()),
        "agreement_count_within_fraction": (
            agreement_delta / expected_count
            <= REPRODUCTION_TOLERANCES["agreement_count_fraction"]
        ),
        "agreement_accuracy_within_tolerance": (
            abs(observed["agreement_accuracy"] - EXPECTED_VISDA["agreement_accuracy"])
            <= REPRODUCTION_TOLERANCES["agreement_accuracy_pp"]
        ),
        "task_accuracy_within_tolerance": (
            abs(observed["task_accuracy"] - EXPECTED_VISDA["task_accuracy"])
            <= REPRODUCTION_TOLERANCES["task_accuracy_pp"]
        ),
        "clip_accuracy_within_tolerance": (
            abs(observed["clip_accuracy"] - EXPECTED_VISDA["clip_accuracy"])
            <= REPRODUCTION_TOLERANCES["clip_accuracy_pp"]
        ),
        "arithmetic_mix_accuracy_within_tolerance": (
            abs(
                observed["arithmetic_mix_accuracy"]
                - EXPECTED_VISDA["arithmetic_mix_accuracy"]
            )
            <= REPRODUCTION_TOLERANCES["arithmetic_mix_accuracy_pp"]
        ),
    }
    return {
        "expected": EXPECTED_VISDA,
        "tolerances": REPRODUCTION_TOLERANCES,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _oracle_ce_direction(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    direction = -np.asarray(probabilities, dtype=np.float64)
    direction[np.arange(labels.size), labels] += 1.0
    return direction


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    reproduction = summary["baseline_reproduction"]
    conflict = summary["conflict_oracle_diagnostic"]
    reliability = summary["feature_reliability_oracle_diagnostic"]
    gradient = summary["gradient_oracle_diagnostic"]
    gate = summary["gate"]
    lines = [
        "# VisDA DUET Feature-Gravity Offline Audit",
        "",
        f"Decision: **{gate['decision']}**",
        "",
        "This is an oracle diagnostic. Target labels were parsed only after the",
        "label-free CSV/NPZ artifacts were written and SHA256-locked.",
        "No optimizer was constructed and no model parameter was updated.",
        "",
        "## Baseline reproduction",
        "",
        f"Passed: `{reproduction['passed']}`",
        "",
        "## Conflict coverage",
        "",
        f"- Conflict samples: `{conflict['samples']}` ({conflict['rate']:.4f}%).",
        f"- Task/CLIP candidate coverage: `{conflict['candidate_coverage']:.4f}%`.",
        f"- Fixed task accuracy: `{conflict['fixed_task_accuracy']:.4f}%`.",
        f"- Fixed CLIP accuracy: `{conflict['fixed_clip_accuracy']:.4f}%`.",
        f"- Higher-confidence accuracy: `{conflict['higher_confidence_accuracy']:.4f}%`.",
        "",
        "## Feature reliability",
        "",
        f"- Feature-cosine AUROC: `{reliability['feature_cosine_auroc']:.6f}`.",
        f"- Task-confidence AUROC: `{reliability['task_confidence_auroc']:.6f}`.",
        f"- AUROC gain: `{reliability['auc_gain']:.6f}`; 95% CI "
        f"`{reliability['auc_gain_bootstrap_95_ci']}`.",
        f"- Top-bottom cosine quintile task-accuracy gap: "
        f"`{reliability['top_bottom_task_accuracy_gap_pp']:.4f} pp`.",
        f"- Conflicts with |weight - 1| >= 0.10: "
        f"`{reliability['effective_weight_coverage']:.4f}%`.",
        "",
        "## First-order gradient diagnostic",
        "",
        f"- Harmful mass reduction: `{gradient['harmful_reduction_percent']:.4f}%`.",
        f"- Helpful mass retention: `{gradient['helpful_retention_percent']:.4f}%`.",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(f"- {name}: `{passed}`" for name, passed in gate["checks"].items())
    lines.extend(
        [
            "",
            "Passing this audit does not authorize training. It only justifies",
            "considering one separately approved matched proxy experiment.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    _prepare_cfg()
    if cfg.SETTING.DATASET != "VISDA-C" or cfg.SETTING.S != 0 or cfg.SETTING.T != 1:
        raise ValueError("This predeclared audit is restricted to VisDA train->validation")
    if str(cfg.ACTIVE.ADAPTATION_LIST).strip():
        raise ValueError("Feature-gravity audit requires the full target list")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VisDA feature-gravity audit")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    random.seed(cfg.SETTING.SEED)
    np.random.seed(cfg.SETTING.SEED)
    torch.manual_seed(cfg.SETTING.SEED)
    torch.cuda.manual_seed_all(cfg.SETTING.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    class_names = _load_class_names()
    test_loader = data_load(cfg)["test_aug"]
    image_records = test_loader.dataset.imgs
    ordered_paths = [str(record[0]) for record in image_records]
    if len(ordered_paths) != EXPECTED_VISDA["total_samples"]:
        raise ValueError(
            f"Expected {EXPECTED_VISDA['total_samples']} full-target rows, "
            f"found {len(ordered_paths)}"
        )

    net_f, net_b, net_c = _build_source_model(device)
    clip_model, _, _ = clip.load(cfg.ACTIVE.ARCH, device=device)
    clip_model.float()
    for module in (net_f, net_b, net_c, clip_model):
        _freeze(module)

    text_inputs = clip_pre_text(cfg).to(device)
    with torch.no_grad():
        text_features = F.normalize(clip_model.encode_text(text_inputs), dim=1).detach()

    full_parts: dict[str, list[np.ndarray]] = {
        "index": [],
        "task_pred": [],
        "clip_pred": [],
        "arithmetic_pred": [],
        "task_conf": [],
        "clip_conf": [],
    }
    conflict_parts: dict[str, list[np.ndarray]] = {
        "index": [],
        "task_pred": [],
        "clip_pred": [],
        "task_conf": [],
        "clip_conf": [],
        "feature_cosine": [],
        "weak_strong_kl": [],
        "clip_kl": [],
        "weak_prob": [],
        "strong_prob": [],
        "consistency_descent_weak": [],
        "consistency_descent_strong": [],
        "clip_descent_weak": [],
    }

    # Phase 1: labels returned by the loader are intentionally ignored.
    for inputs, _ignored_labels, indices in test_loader:
        del _ignored_labels
        weak_x = inputs[1].to(device)
        strong_x = inputs[2].to(device)
        indices_cpu = indices.long().cpu().numpy()
        with torch.no_grad():
            weak_features = net_b(net_f(weak_x))
            strong_features = net_b(net_f(strong_x))
            weak_logits = net_c(weak_features)
            strong_logits = net_c(strong_features)
            weak_prob = torch.softmax(weak_logits, dim=1)
            strong_prob = torch.softmax(strong_logits, dim=1)
            clip_features = F.normalize(clip_model.encode_image(weak_x), dim=1)
            clip_logits = clip_model.logit_scale.exp() * clip_features @ text_features.t()
            clip_prob = torch.softmax(clip_logits, dim=1)
            task_conf, task_pred = weak_prob.max(dim=1)
            clip_conf, clip_pred = clip_prob.max(dim=1)
            arithmetic_pred = arithmetic_probability_fusion(weak_prob, clip_prob).argmax(dim=1)
            feature_cosine = F.cosine_similarity(weak_features, strong_features, dim=1)

        for name, value in (
            ("index", indices_cpu),
            ("task_pred", task_pred.cpu().numpy()),
            ("clip_pred", clip_pred.cpu().numpy()),
            ("arithmetic_pred", arithmetic_pred.cpu().numpy()),
            ("task_conf", task_conf.cpu().numpy()),
            ("clip_conf", clip_conf.cpu().numpy()),
        ):
            full_parts[name].append(value)

        conflict_position = torch.nonzero(task_pred != clip_pred, as_tuple=False).flatten()
        if conflict_position.numel() == 0:
            continue

        # Exact output-level components of the released DUET conflict objective.
        # Logits are detached from model parameters; autograd therefore cannot
        # update or accumulate gradients in the source or CLIP networks.
        components = duet_logit_descent_components(
            weak_logits[conflict_position],
            strong_logits[conflict_position],
            clip_logits[conflict_position],
            con_weight=float(cfg.ACTIVE.CON_PAR),
            clip_weight=float(cfg.ACTIVE.KL_PAR),
            batch_size=int(weak_x.size(0)),
        )

        selected = conflict_position.cpu().numpy()
        values = {
            "index": indices_cpu[selected],
            "task_pred": task_pred[conflict_position].cpu().numpy(),
            "clip_pred": clip_pred[conflict_position].cpu().numpy(),
            "task_conf": task_conf[conflict_position].cpu().numpy(),
            "clip_conf": clip_conf[conflict_position].cpu().numpy(),
            "feature_cosine": feature_cosine[conflict_position].cpu().numpy(),
            "weak_strong_kl": components["consistency_per_sample"].cpu().numpy(),
            "clip_kl": components["clip_per_sample"].cpu().numpy(),
            "weak_prob": components["weak_prob"].cpu().numpy(),
            "strong_prob": components["strong_prob"].cpu().numpy(),
            "consistency_descent_weak": components[
                "consistency_descent_weak"
            ].cpu().numpy(),
            "consistency_descent_strong": components[
                "consistency_descent_strong"
            ].cpu().numpy(),
            "clip_descent_weak": components["clip_descent_weak"].cpu().numpy(),
        }
        for name, value in values.items():
            conflict_parts[name].append(value)

    full = {name: np.concatenate(parts) for name, parts in full_parts.items()}
    full_order = np.argsort(full["index"], kind="mergesort")
    full = {name: value[full_order] for name, value in full.items()}
    conflict = {name: np.concatenate(parts) for name, parts in conflict_parts.items()}
    conflict_order = np.argsort(conflict["index"], kind="mergesort")
    conflict = {name: value[conflict_order] for name, value in conflict.items()}
    if not np.array_equal(full["index"], np.arange(full["index"].size)):
        raise RuntimeError("Full-target indices are not unique contiguous VisDA indices")
    if not np.array_equal(
        conflict["index"],
        np.flatnonzero(full["task_pred"] != full["clip_pred"]),
    ):
        raise RuntimeError("Conflict export is not aligned with full-target predictions")

    gravity_raw = np.clip(conflict["feature_cosine"].astype(np.float64), 0.0, None)
    if gravity_raw.mean() <= 0.0:
        raise RuntimeError("Mean non-negative feature cosine is zero")
    gravity_weight = gravity_raw / gravity_raw.mean()
    bottom_tail, top_tail = fixed_tail_masks(
        conflict["feature_cosine"], fraction=TAIL_FRACTION
    )
    current_descent = np.concatenate(
        (
            conflict["consistency_descent_weak"] + conflict["clip_descent_weak"],
            conflict["consistency_descent_strong"],
        ),
        axis=1,
    )
    candidate_descent = np.concatenate(
        (
            gravity_weight[:, None] * conflict["consistency_descent_weak"]
            + conflict["clip_descent_weak"],
            gravity_weight[:, None] * conflict["consistency_descent_strong"],
        ),
        axis=1,
    )

    out_dir = Path(cfg.output_dir) / "feature_gravity_audit"
    out_dir.mkdir(parents=True, exist_ok=False)
    signal_csv = out_dir / "visda_conflict_feature_gravity_signals.csv"
    signal_npz = out_dir / "visda_conflict_feature_gravity_signals.npz"
    lock_path = out_dir / "visda_conflict_feature_gravity_signal_lock.json"
    oracle_csv = out_dir / "visda_conflict_feature_gravity_oracle_diagnostic.csv"
    class_csv = out_dir / "visda_conflict_feature_gravity_classwise.csv"
    summary_path = out_dir / "visda_conflict_feature_gravity_summary.json"
    markdown_path = out_dir / "visda_conflict_feature_gravity_summary.md"

    signal_rows: list[dict[str, Any]] = []
    for position, index in enumerate(conflict["index"]):
        signal_rows.append(
            {
                "index": int(index),
                "path": ordered_paths[int(index)],
                "task_pred": int(conflict["task_pred"][position]),
                "task_pred_name": class_names[int(conflict["task_pred"][position])],
                "clip_pred": int(conflict["clip_pred"][position]),
                "clip_pred_name": class_names[int(conflict["clip_pred"][position])],
                "task_conf": float(conflict["task_conf"][position]),
                "clip_conf": float(conflict["clip_conf"][position]),
                "feature_cosine": float(conflict["feature_cosine"][position]),
                "gravity_weight_mean_preserving": float(gravity_weight[position]),
                "weak_strong_kl": float(conflict["weak_strong_kl"][position]),
                "clip_kl": float(conflict["clip_kl"][position]),
                "bottom_cosine_quintile": bool(bottom_tail[position]),
                "top_cosine_quintile": bool(top_tail[position]),
                "current_descent_norm": float(np.linalg.norm(current_descent[position])),
                "candidate_descent_norm": float(np.linalg.norm(candidate_descent[position])),
            }
        )
    _write_csv(signal_csv, signal_rows)
    np.savez_compressed(
        signal_npz,
        index=conflict["index"],
        task_pred=conflict["task_pred"],
        clip_pred=conflict["clip_pred"],
        task_conf=conflict["task_conf"],
        clip_conf=conflict["clip_conf"],
        feature_cosine=conflict["feature_cosine"],
        gravity_weight=gravity_weight.astype(np.float32),
        weak_prob=conflict["weak_prob"],
        strong_prob=conflict["strong_prob"],
        current_descent=current_descent,
        candidate_descent=candidate_descent,
        bottom_cosine_quintile=bottom_tail,
        top_cosine_quintile=top_tail,
    )
    lock = {
        "phase": "LABEL_FREE_SIGNAL_LOCK",
        "contains_target_labels": False,
        "labels_parsed_after_this_manifest": True,
        "seed": int(cfg.SETTING.SEED),
        "full_target_required": True,
        "signal_csv": {"path": str(signal_csv), "sha256": _sha256(signal_csv)},
        "signal_npz": {"path": str(signal_npz), "sha256": _sha256(signal_npz)},
        "target_list_sha256": _sha256(cfg.t_dset_path),
        "source_checkpoint_sha256": {
            name: _sha256(Path(cfg.output_dir_src) / name)
            for name in ("source_F.pt", "source_B.pt", "source_C.pt")
        },
        "contract_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                REPO_ROOT / "cfgs/visda/plmatch.yaml",
                REPO_ROOT / "src/methods/oh/plmatch.py",
                REPO_ROOT / "src/utils/feature_gravity_audit.py",
                Path(__file__).resolve(),
            )
        },
        "signal_columns": list(signal_rows[0].keys()),
        "npz_arrays": [
            "index",
            "task_pred",
            "clip_pred",
            "task_conf",
            "clip_conf",
            "feature_cosine",
            "gravity_weight",
            "weak_prob",
            "strong_prob",
            "current_descent",
            "candidate_descent",
            "bottom_cosine_quintile",
            "top_cosine_quintile",
        ],
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: only now reveal labels for explicitly oracle diagnostics.
    all_label = _parse_labels_after_lock(cfg.t_dset_path, ordered_paths)
    conflict_label = all_label[conflict["index"]]
    agreement = full["task_pred"] == full["clip_pred"]
    all_mask = np.ones(all_label.size, dtype=bool)
    baseline_observed = {
        "total_samples": int(all_label.size),
        "agreement_samples": int(agreement.sum()),
        "agreement_accuracy": _accuracy(full["task_pred"], all_label, agreement),
        "task_accuracy": _accuracy(full["task_pred"], all_label, all_mask),
        "clip_accuracy": _accuracy(full["clip_pred"], all_label, all_mask),
        "arithmetic_mix_accuracy": _accuracy(full["arithmetic_pred"], all_label, all_mask),
    }
    reproduction = _baseline_reproduction(baseline_observed, full["index"])

    task_correct = conflict["task_pred"] == conflict_label
    clip_correct = conflict["clip_pred"] == conflict_label
    higher_confidence_prediction = np.where(
        conflict["task_conf"] >= conflict["clip_conf"],
        conflict["task_pred"],
        conflict["clip_pred"],
    )
    feature_auc = binary_auroc(conflict["feature_cosine"], task_correct)
    confidence_auc = binary_auroc(conflict["task_conf"], task_correct)
    auc_ci = stratified_bootstrap_auc_difference(
        conflict["feature_cosine"],
        conflict["task_conf"],
        task_correct,
        repeats=BOOTSTRAP_REPEATS,
        seed=int(cfg.SETTING.SEED),
    )
    bottom_accuracy = _pct(task_correct[bottom_tail].sum(), bottom_tail.sum())
    top_accuracy = _pct(task_correct[top_tail].sum(), top_tail.sum())

    oracle_direction = np.concatenate(
        (
            _oracle_ce_direction(conflict["weak_prob"], conflict_label),
            _oracle_ce_direction(conflict["strong_prob"], conflict_label),
        ),
        axis=1,
    )
    current_gradient = gradient_projection_summary(current_descent, oracle_direction)
    candidate_gradient = gradient_projection_summary(candidate_descent, oracle_direction)
    harmful_reduction = (
        100.0
        * (current_gradient["harmful_mass"] - candidate_gradient["harmful_mass"])
        / current_gradient["harmful_mass"]
        if current_gradient["harmful_mass"] > 0.0
        else 0.0
    )
    helpful_retention = (
        100.0 * candidate_gradient["helpful_mass"] / current_gradient["helpful_mass"]
        if current_gradient["helpful_mass"] > 0.0
        else 0.0
    )
    classwise = classwise_gradient_mass(
        conflict_label,
        class_names,
        current_gradient["projection"],
        candidate_gradient["projection"],
    )
    gate = evaluate_preflight_gate(
        reproduction_passed=reproduction["passed"],
        auc_gain=feature_auc - confidence_auc,
        auc_ci=auc_ci,
        quintile_accuracy_gap_pp=top_accuracy - bottom_accuracy,
        harmful_reduction_percent=harmful_reduction,
        helpful_retention_percent=helpful_retention,
        classwise=classwise,
    )

    oracle_rows: list[dict[str, Any]] = []
    for position, index in enumerate(conflict["index"]):
        label = int(conflict_label[position])
        oracle_rows.append(
            {
                "index": int(index),
                "label": label,
                "label_name": class_names[label],
                "candidate_contains_label": bool(task_correct[position] or clip_correct[position]),
                "task_correct": bool(task_correct[position]),
                "clip_correct": bool(clip_correct[position]),
                "higher_confidence_correct": bool(
                    higher_confidence_prediction[position] == label
                ),
                "feature_cosine": float(conflict["feature_cosine"][position]),
                "gravity_weight_mean_preserving": float(gravity_weight[position]),
                "current_oracle_gradient_projection": float(
                    current_gradient["projection"][position]
                ),
                "candidate_oracle_gradient_projection": float(
                    candidate_gradient["projection"][position]
                ),
            }
        )
    _write_csv(oracle_csv, oracle_rows)
    _write_csv(class_csv, classwise)

    summary = {
        "dataset": "VISDA-C",
        "task": "train->validation",
        "seed": int(cfg.SETTING.SEED),
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "model_parameters_updated": False,
        "training_authorized": False,
        "candidate_contract": {
            "scope": "current task/CLIP conflicts only",
            "feature_signal": "cosine(task weak feature, task strong feature)",
            "weight": "clamp_min(cosine, 0) / mean_conflict_weight",
            "weight_gradient": "stopped",
            "gradient_audit_precision": "float64 detached logits with log_softmax",
            "agreement_samples": "unchanged weight 1",
            "task_clip_fusion": "unchanged",
            "pseudo_label_admission": "unchanged",
            "loss_coefficients": {
                "CON_PAR": float(cfg.ACTIVE.CON_PAR),
                "KL_PAR": float(cfg.ACTIVE.KL_PAR),
            },
        },
        "baseline_observed": baseline_observed,
        "baseline_reproduction": reproduction,
        "conflict_oracle_diagnostic": {
            "samples": int(conflict_label.size),
            "rate": _pct(conflict_label.size, all_label.size),
            "candidate_coverage": _pct(np.sum(task_correct | clip_correct), conflict_label.size),
            "fixed_task_accuracy": _pct(task_correct.sum(), conflict_label.size),
            "fixed_clip_accuracy": _pct(clip_correct.sum(), conflict_label.size),
            "higher_confidence_accuracy": _pct(
                np.sum(higher_confidence_prediction == conflict_label), conflict_label.size
            ),
        },
        "feature_reliability_oracle_diagnostic": {
            "feature_cosine_auroc": feature_auc,
            "task_confidence_auroc": confidence_auc,
            "auc_gain": feature_auc - confidence_auc,
            "auc_gain_bootstrap_95_ci": list(auc_ci),
            "tail_fraction_locked_without_labels": TAIL_FRACTION,
            "bottom_quintile_task_accuracy": bottom_accuracy,
            "top_quintile_task_accuracy": top_accuracy,
            "top_bottom_task_accuracy_gap_pp": top_accuracy - bottom_accuracy,
            "effective_weight_definition": f"abs(weight - 1) >= {WEIGHT_EFFECT_SIZE}",
            "effective_weight_samples": int(
                np.sum(np.abs(gravity_weight - 1.0) >= WEIGHT_EFFECT_SIZE)
            ),
            "effective_weight_coverage": _pct(
                np.sum(np.abs(gravity_weight - 1.0) >= WEIGHT_EFFECT_SIZE),
                gravity_weight.size,
            ),
            "feature_cosine_quantiles": {
                str(quantile): float(np.quantile(conflict["feature_cosine"], quantile))
                for quantile in (0.0, 0.2, 0.5, 0.8, 1.0)
            },
        },
        "gradient_oracle_diagnostic": {
            "space": "concatenated weak/strong task logits",
            "oracle_direction": "negative supervised-CE logit gradient",
            "current_duet": {
                key: value for key, value in current_gradient.items() if key != "projection"
            },
            "feature_weighted_candidate": {
                key: value for key, value in candidate_gradient.items() if key != "projection"
            },
            "harmful_reduction_percent": harmful_reduction,
            "helpful_retention_percent": helpful_retention,
        },
        "classwise_gradient_oracle_diagnostic": classwise,
        "gate": gate,
        "decision": gate["decision"],
        "next": (
            "request explicit approval for one matched proxy experiment"
            if gate["decision"] == "PASS_OFFLINE_GATE"
            else "stop; do not run proxy or full VisDA training"
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)

    print(f"Wrote label-free signals: {signal_csv}")
    print(f"Wrote label-free tensor signals: {signal_npz}")
    print(f"Locked signals before labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_csv}")
    print(f"Wrote classwise diagnostic: {class_csv}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote Markdown report: {markdown_path}")
    print(json.dumps({"decision": gate["decision"], "checks": gate["checks"]}, indent=2))


if __name__ == "__main__":
    main()
