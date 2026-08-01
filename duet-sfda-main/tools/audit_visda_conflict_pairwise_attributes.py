#!/usr/bin/env python
"""Frozen text-attribute audit for task/CLIP conflicts on VisDA-C.

Phase 1 ignores loader labels, reproduces the released DUET task and CLIP
ViT-B/32 forward path, and locks fixed class-description signals for every
task/CLIP top-1 conflict.  Phase 2 parses target labels only for explicitly
marked oracle diagnostics and a predeclared rejection gate.  The script creates
no optimizer, performs no backward pass, and cannot update model parameters.
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
from src.utils.conflict_boundary import paired_accuracy_bootstrap_ci  # noqa: E402
from src.utils.pairwise_attribute_audit import (  # noqa: E402
    ATTRIBUTE_FAMILIES,
    PROMPT_TEMPLATES,
    build_visda_attribute_prompt_manifest,
    evaluate_pairwise_attribute_gate,
    pairwise_attribute_task_rescue,
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


BOOTSTRAP_REPEATS = 2_000
MIN_TEMPLATE_STABILITY = 0.90
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        raise ValueError("VisDA list and loader rows do not match")
    return np.asarray(
        [labels_by_path[path] for path in ordered_image_paths], dtype=np.int64
    )


def _accuracy(prediction: np.ndarray, label: np.ndarray, mask: np.ndarray) -> float:
    return _pct(np.sum(prediction[mask] == label[mask]), np.sum(mask))


def _baseline_reproduction(
    observed: dict[str, Any], ordered_indices: np.ndarray
) -> dict[str, Any]:
    expected_count = EXPECTED_VISDA["total_samples"]
    agreement_delta = abs(
        observed["agreement_samples"] - EXPECTED_VISDA["agreement_samples"]
    )
    checks = {
        "full_target_count": observed["total_samples"] == expected_count,
        "contiguous_unique_indices": np.array_equal(
            ordered_indices, np.arange(ordered_indices.size)
        ),
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


def _paired_comparison(
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    candidate_correct = candidate_prediction[mask] == labels[mask]
    baseline_correct = baseline_prediction[mask] == labels[mask]
    candidate_accuracy = _pct(candidate_correct.sum(), candidate_correct.size)
    baseline_accuracy = _pct(baseline_correct.sum(), baseline_correct.size)
    ci = paired_accuracy_bootstrap_ci(
        candidate_correct,
        baseline_correct,
        repeats=BOOTSTRAP_REPEATS,
        seed=int(cfg.SETTING.SEED),
    )
    return {
        "samples": int(mask.sum()),
        "candidate_accuracy": candidate_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "gain_pp": candidate_accuracy - baseline_accuracy,
        "paired_bootstrap_95_ci_pp": list(ci),
        "net_corrections": int(candidate_correct.sum() - baseline_correct.sum()),
    }


def _encode_attribute_text(
    clip_model: torch.nn.Module,
    prompt_manifest: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    prompts = prompt_manifest["flat_prompts"]
    tokens = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)
    parts = []
    with torch.no_grad():
        for start in range(0, tokens.size(0), 256):
            parts.append(clip_model.encode_text(tokens[start : start + 256]))
    features = F.normalize(torch.cat(parts), dim=1)
    class_count, template_count, family_count = prompt_manifest["shape"]
    return features.reshape(class_count, template_count, family_count, -1)


def _float_list(values: np.ndarray) -> str:
    return ";".join(f"{float(value):.9g}" for value in values.reshape(-1))


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    conflict = summary["full_conflict_oracle_diagnostic"]
    rescue = summary["task_rescue_oracle_diagnostic"]
    comparison = summary["routed_versus_fixed_clip"]
    lines = [
        "# VisDA Pairwise Visible-Attribute Offline Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The task model and CLIP ViT-B/32 image path are frozen and unchanged.",
        "Descriptions are fixed from class-name semantics before target labels are read.",
        "No optimizer, backward pass, parameter update, or training is present.",
        "",
        "## Full conflict oracle diagnostic",
        "",
        f"- Conflicts: `{conflict['samples']}`.",
        f"- Fixed CLIP accuracy: `{conflict['fixed_clip_accuracy']:.4f}%`.",
        f"- Task/CLIP top-1 oracle coverage: `{conflict['top1_union_coverage']:.4f}%`.",
        f"- Routed accuracy: `{comparison['candidate_accuracy']:.4f}%`.",
        f"- Gain: `{comparison['gain_pp']:.4f} pp`; 95% CI "
        f"`{comparison['paired_bootstrap_95_ci_pp']}`.",
        "",
        "## Task rescue oracle diagnostic",
        "",
        f"- Coverage: `{rescue['coverage']:.4f}%`.",
        f"- Task adjudication precision: `{rescue['adjudication_precision']:.4f}%`.",
        f"- Median routed stability: `{rescue['median_template_stability']:.6f}`.",
        f"- Car net corrections: `{rescue['car_net_corrections']}`.",
        f"- Truck net corrections: `{rescue['truck_net_corrections']}`.",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend(
        [
            "",
            "Passing authorizes only a separately approved matched proxy run; it does",
            "not authorize or start proxy or full VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    _prepare_cfg()
    if cfg.SETTING.DATASET != "VISDA-C" or cfg.SETTING.S != 0 or cfg.SETTING.T != 1:
        raise ValueError("This audit is restricted to VisDA train->validation")
    if str(cfg.ACTIVE.ADAPTATION_LIST).strip():
        raise ValueError("Pairwise attribute audit requires the full target list")
    if str(cfg.ACTIVE.ARCH) != "ViT-B/32":
        raise ValueError("Pairwise attribute audit is locked to CLIP ViT-B/32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VisDA pairwise attribute audit")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    _seed_everything(int(cfg.SETTING.SEED))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    class_names = _load_class_names()
    normalized_names = [name.strip().lower().replace("_", " ") for name in class_names]
    if "car" not in normalized_names or "truck" not in normalized_names:
        raise ValueError("VisDA class list must contain car and truck")
    car_index = normalized_names.index("car")
    truck_index = normalized_names.index("truck")
    prompt_manifest = build_visda_attribute_prompt_manifest(class_names)

    test_loader = data_load(cfg)["test_aug"]
    ordered_paths = [str(record[0]) for record in test_loader.dataset.imgs]
    if len(ordered_paths) != EXPECTED_VISDA["total_samples"]:
        raise ValueError(
            f"Expected {EXPECTED_VISDA['total_samples']} target rows, "
            f"found {len(ordered_paths)}"
        )

    net_f, net_b, net_c = _build_source_model(device)
    clip_model, _, _ = clip.load(cfg.ACTIVE.ARCH, device=device)
    clip_model.float()
    for module in (net_f, net_b, net_c, clip_model):
        _freeze(module)

    text_inputs = clip_pre_text(cfg).to(device)
    with torch.no_grad():
        class_text_features = F.normalize(
            clip_model.encode_text(text_inputs), dim=1
        ).detach()
    attribute_text_features = _encode_attribute_text(
        clip_model, prompt_manifest, device
    ).detach()

    full_parts: dict[str, list[np.ndarray]] = {
        "index": [],
        "task_probability": [],
        "clip_probability": [],
        "clip_image_feature": [],
        "task_prediction": [],
        "clip_prediction": [],
        "task_confidence": [],
        "clip_confidence": [],
        "arithmetic_prediction": [],
        "rms_prediction": [],
    }
    # Phase 1: loader labels are deliberately ignored.
    for inputs, _ignored_labels, indices in test_loader:
        del _ignored_labels
        weak_x = inputs[1].to(device)
        with torch.no_grad():
            task_logits = net_c(net_b(net_f(weak_x)))
            clip_features = F.normalize(clip_model.encode_image(weak_x), dim=1)
            clip_logits = (
                clip_model.logit_scale.exp()
                * clip_features
                @ class_text_features.t()
            )
            task_probability = torch.softmax(task_logits, dim=1)
            clip_probability = torch.softmax(clip_logits, dim=1)
            task_confidence, task_prediction = task_probability.max(dim=1)
            clip_confidence, clip_prediction = clip_probability.max(dim=1)
            arithmetic_prediction = arithmetic_probability_fusion(
                task_probability, clip_probability
            ).argmax(dim=1)
            rms_prediction = rms_probability_fusion(
                task_probability, clip_probability
            ).argmax(dim=1)
        for name, value in (
            ("index", indices.long().cpu().numpy()),
            ("task_probability", task_probability.cpu().numpy()),
            ("clip_probability", clip_probability.cpu().numpy()),
            ("clip_image_feature", clip_features.float().cpu().numpy()),
            ("task_prediction", task_prediction.cpu().numpy()),
            ("clip_prediction", clip_prediction.cpu().numpy()),
            ("task_confidence", task_confidence.cpu().numpy()),
            ("clip_confidence", clip_confidence.cpu().numpy()),
            ("arithmetic_prediction", arithmetic_prediction.cpu().numpy()),
            ("rms_prediction", rms_prediction.cpu().numpy()),
        ):
            full_parts[name].append(value)

    full = {name: np.concatenate(parts) for name, parts in full_parts.items()}
    order = np.argsort(full["index"], kind="mergesort")
    full = {name: value[order] for name, value in full.items()}
    if not np.array_equal(full["index"], np.arange(len(ordered_paths))):
        raise RuntimeError("Full-target indices are not unique contiguous VisDA indices")

    conflict = full["task_prediction"] != full["clip_prediction"]
    conflict_indices = np.flatnonzero(conflict)
    conflict_task = full["task_prediction"][conflict_indices]
    conflict_clip = full["clip_prediction"][conflict_indices]
    attribute = pairwise_attribute_task_rescue(
        full["clip_image_feature"][conflict_indices],
        attribute_text_features.float().cpu().numpy(),
        conflict_task,
        conflict_clip,
        min_template_stability=MIN_TEMPLATE_STABILITY,
    )

    out_dir = Path(cfg.output_dir) / "pairwise_attribute_audit"
    out_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_pairwise_attribute"
    prompt_path = out_dir / f"{stem}_prompt_contract.json"
    signal_csv = out_dir / f"{stem}_signals.csv"
    signal_npz = out_dir / f"{stem}_signals.npz"
    lock_path = out_dir / f"{stem}_signal_lock.json"
    oracle_csv = out_dir / f"{stem}_oracle_diagnostic.csv"
    class_csv = out_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    pair_csv = out_dir / f"{stem}_pairwise_oracle_diagnostic.csv"
    summary_path = out_dir / f"{stem}_summary.json"
    markdown_path = out_dir / f"{stem}_summary.md"

    prompt_path.write_text(json.dumps(prompt_manifest, indent=2) + "\n")
    signal_rows: list[dict[str, Any]] = []
    for row, index in enumerate(conflict_indices):
        task_index = int(conflict_task[row])
        clip_index = int(conflict_clip[row])
        routed_index = int(attribute["routed_prediction"][row])
        signal_rows.append(
            {
                "index": int(index),
                "path": ordered_paths[int(index)],
                "task_top1": task_index,
                "task_top1_name": class_names[task_index],
                "clip_top1": clip_index,
                "clip_top1_name": class_names[clip_index],
                "attribute_margin_template_family": _float_list(
                    attribute["margin"][row]
                ),
                "attribute_family_margin": _float_list(
                    attribute["family_margin"][row]
                ),
                "attribute_template_margin": _float_list(
                    attribute["template_margin"][row]
                ),
                "template_stability": float(
                    attribute["template_stability"][row]
                ),
                "all_families_support_task": bool(
                    attribute["all_families_support_task"][row]
                ),
                "both_templates_support_task": bool(
                    attribute["both_templates_support_task"][row]
                ),
                "task_rescue": bool(attribute["task_rescue"][row]),
                "descriptor_prediction": int(
                    attribute["descriptor_prediction"][row]
                ),
                "descriptor_prediction_name": class_names[
                    int(attribute["descriptor_prediction"][row])
                ],
                "routed_prediction": routed_index,
                "routed_prediction_name": class_names[routed_index],
            }
        )
    _write_csv(signal_csv, signal_rows)
    np.savez_compressed(
        signal_npz,
        index=conflict_indices,
        task_probability=full["task_probability"][conflict_indices],
        clip_probability=full["clip_probability"][conflict_indices],
        task_prediction=conflict_task,
        clip_prediction=conflict_clip,
        attribute_margin=attribute["margin"],
        attribute_family_margin=attribute["family_margin"],
        attribute_template_margin=attribute["template_margin"],
        template_stability=attribute["template_stability"],
        all_families_support_task=attribute["all_families_support_task"],
        both_templates_support_task=attribute["both_templates_support_task"],
        task_rescue=attribute["task_rescue"],
        descriptor_prediction=attribute["descriptor_prediction"],
        routed_prediction=attribute["routed_prediction"],
    )
    lock = {
        "phase": "LABEL_FREE_SIGNAL_LOCK",
        "contains_target_labels": False,
        "loader_labels_ignored": True,
        "oracle_labels_parsed_after_this_manifest": True,
        "target_images_used_to_generate_descriptions": False,
        "external_visual_models_used": False,
        "seed": int(cfg.SETTING.SEED),
        "clip_architecture": str(cfg.ACTIVE.ARCH),
        "source_architecture": str(cfg.MODEL.ARCH),
        "candidate_contract": {
            "set": "task top-1 and CLIP top-1 conflicts only",
            "default": "fixed CLIP top-1",
            "override": "task only when all four attribute families and both templates support task",
            "min_template_stability": MIN_TEMPLATE_STABILITY,
            "top2_candidates": False,
            "target_label_thresholds": False,
        },
        "description_contract": {
            "class_count": len(class_names),
            "template_count": len(PROMPT_TEMPLATES),
            "attribute_families": list(ATTRIBUTE_FAMILIES),
            "runtime_external_llm_calls": 0,
        },
        "prompt_contract": {"path": str(prompt_path), "sha256": _sha256(prompt_path)},
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
                REPO_ROOT / "clip/model.py",
                REPO_ROOT / "src/utils/pairwise_attribute_audit.py",
                Path(__file__).resolve(),
            )
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    # Phase 2: target labels are revealed only for explicit oracle diagnostics.
    all_label = _parse_labels_after_lock(cfg.t_dset_path, ordered_paths)
    all_mask = np.ones(all_label.size, dtype=bool)
    agreement = ~conflict
    baseline_observed = {
        "total_samples": int(all_label.size),
        "agreement_samples": int(agreement.sum()),
        "agreement_accuracy": _accuracy(
            full["task_prediction"], all_label, agreement
        ),
        "task_accuracy": _accuracy(full["task_prediction"], all_label, all_mask),
        "clip_accuracy": _accuracy(full["clip_prediction"], all_label, all_mask),
        "arithmetic_mix_accuracy": _accuracy(
            full["arithmetic_prediction"], all_label, all_mask
        ),
        "rms_accuracy": _accuracy(full["rms_prediction"], all_label, all_mask),
    }
    reproduction = _baseline_reproduction(baseline_observed, full["index"])

    conflict_label = all_label[conflict_indices]
    conflict_mask = np.ones(conflict_indices.size, dtype=bool)
    conflict_task_correct = conflict_task == conflict_label
    conflict_clip_correct = conflict_clip == conflict_label
    routed_prediction = attribute["routed_prediction"]
    routed_correct = routed_prediction == conflict_label
    top1_union = conflict_task_correct | conflict_clip_correct
    higher_confidence = np.where(
        full["task_confidence"][conflict_indices]
        >= full["clip_confidence"][conflict_indices],
        conflict_task,
        conflict_clip,
    )
    routed_versus_clip = _paired_comparison(
        routed_prediction,
        conflict_clip,
        conflict_label,
        conflict_mask,
    )

    rescue = attribute["task_rescue"]
    rescue_coverage = _pct(rescue.sum(), rescue.size)
    adjudication_eligible = rescue & top1_union
    adjudication_precision = _accuracy(
        conflict_task, conflict_label, adjudication_eligible
    )
    rescue_task_precision = _accuracy(conflict_task, conflict_label, rescue)
    median_routed_stability = (
        float(np.median(attribute["template_stability"][rescue]))
        if rescue.any()
        else 0.0
    )
    car_mask = conflict_label == car_index
    truck_mask = conflict_label == truck_index
    car_net = int(routed_correct[car_mask].sum() - conflict_clip_correct[car_mask].sum())
    truck_net = int(
        routed_correct[truck_mask].sum() - conflict_clip_correct[truck_mask].sum()
    )
    gate = evaluate_pairwise_attribute_gate(
        reproduction_passed=reproduction["passed"],
        conflict_gain_pp=routed_versus_clip["gain_pp"],
        conflict_gain_ci=tuple(routed_versus_clip["paired_bootstrap_95_ci_pp"]),
        rescue_coverage=rescue_coverage,
        adjudication_precision=adjudication_precision,
        median_routed_stability=median_routed_stability,
        car_net_corrections=car_net,
        truck_net_corrections=truck_net,
    )

    global_routed = full["clip_prediction"].copy()
    global_routed[conflict_indices] = routed_prediction
    global_versus_clip = _paired_comparison(
        global_routed,
        full["clip_prediction"],
        all_label,
        all_mask,
    )

    oracle_rows: list[dict[str, Any]] = []
    for row, index in enumerate(conflict_indices):
        label = int(conflict_label[row])
        oracle_rows.append(
            {
                "index": int(index),
                "label": label,
                "label_name": class_names[label],
                "task_top1": int(conflict_task[row]),
                "task_top1_name": class_names[int(conflict_task[row])],
                "clip_top1": int(conflict_clip[row]),
                "clip_top1_name": class_names[int(conflict_clip[row])],
                "top1_union_contains_label": bool(top1_union[row]),
                "task_correct": bool(conflict_task_correct[row]),
                "clip_correct": bool(conflict_clip_correct[row]),
                "higher_confidence_correct": bool(
                    higher_confidence[row] == label
                ),
                "arithmetic_correct": bool(
                    full["arithmetic_prediction"][index] == label
                ),
                "rms_correct": bool(full["rms_prediction"][index] == label),
                "descriptor_prediction_correct": bool(
                    attribute["descriptor_prediction"][row] == label
                ),
                "task_rescue": bool(rescue[row]),
                "adjudication_eligible": bool(adjudication_eligible[row]),
                "routed_correct": bool(routed_correct[row]),
                "template_stability": float(
                    attribute["template_stability"][row]
                ),
            }
        )
    _write_csv(oracle_csv, oracle_rows)

    class_rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        mask = conflict_label == class_index
        count = int(mask.sum())
        routed_count = int(routed_correct[mask].sum())
        clip_count = int(conflict_clip_correct[mask].sum())
        class_rows.append(
            {
                "class_index": class_index,
                "class": class_name,
                "conflict_samples": count,
                "task_rescue_samples": int(rescue[mask].sum()),
                "routed_accuracy": _pct(routed_count, count),
                "fixed_clip_accuracy": _pct(clip_count, count),
                "delta_pp": _pct(routed_count - clip_count, count),
                "net_corrections": routed_count - clip_count,
            }
        )
    _write_csv(class_csv, class_rows)

    pair_rows: list[dict[str, Any]] = []
    for task_index in range(len(class_names)):
        for clip_index in range(len(class_names)):
            mask = (conflict_task == task_index) & (conflict_clip == clip_index)
            count = int(mask.sum())
            if not count:
                continue
            routed_count = int(routed_correct[mask].sum())
            clip_count = int(conflict_clip_correct[mask].sum())
            pair_rows.append(
                {
                    "task_top1": task_index,
                    "task_top1_name": class_names[task_index],
                    "clip_top1": clip_index,
                    "clip_top1_name": class_names[clip_index],
                    "samples": count,
                    "task_rescue_samples": int(rescue[mask].sum()),
                    "routed_accuracy": _pct(routed_count, count),
                    "fixed_clip_accuracy": _pct(clip_count, count),
                    "delta_pp": _pct(routed_count - clip_count, count),
                    "net_corrections": routed_count - clip_count,
                }
            )
    _write_csv(pair_csv, pair_rows)

    summary = {
        "dataset": "VISDA-C",
        "task": "train->validation",
        "seed": int(cfg.SETTING.SEED),
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "optimizer_constructed": False,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "model_parameters_updated": False,
        "training_authorized": False,
        "candidate_contract": lock["candidate_contract"],
        "compute_contract": {
            "target_samples": int(all_label.size),
            "task_image_forwards": int(all_label.size),
            "clip_image_forwards": int(all_label.size),
            "attribute_text_prompts": len(prompt_manifest["flat_prompts"]),
            "masked_views": 0,
            "no_backward": True,
        },
        "baseline_observed": baseline_observed,
        "baseline_reproduction": reproduction,
        "full_conflict_oracle_diagnostic": {
            "samples": int(conflict_indices.size),
            "fixed_task_accuracy": _accuracy(
                conflict_task, conflict_label, conflict_mask
            ),
            "fixed_clip_accuracy": _accuracy(
                conflict_clip, conflict_label, conflict_mask
            ),
            "higher_confidence_accuracy": _accuracy(
                higher_confidence, conflict_label, conflict_mask
            ),
            "arithmetic_accuracy": _accuracy(
                full["arithmetic_prediction"][conflict_indices],
                conflict_label,
                conflict_mask,
            ),
            "rms_accuracy": _accuracy(
                full["rms_prediction"][conflict_indices],
                conflict_label,
                conflict_mask,
            ),
            "descriptor_prediction_accuracy": _accuracy(
                attribute["descriptor_prediction"],
                conflict_label,
                conflict_mask,
            ),
            "top1_union_coverage": _pct(top1_union.sum(), top1_union.size),
            "task_correct_clip_wrong_samples": int(conflict_task_correct.sum()),
            "task_wrong_clip_correct_samples": int(conflict_clip_correct.sum()),
            "both_wrong_samples": int((~top1_union).sum()),
        },
        "task_rescue_oracle_diagnostic": {
            "samples": int(rescue.sum()),
            "coverage": rescue_coverage,
            "adjudication_eligible_samples": int(adjudication_eligible.sum()),
            "adjudication_precision": adjudication_precision,
            "task_precision_including_both_wrong": rescue_task_precision,
            "median_template_stability": median_routed_stability,
            "car_net_corrections": car_net,
            "truck_net_corrections": truck_net,
        },
        "routed_versus_fixed_clip": routed_versus_clip,
        "global_routed_versus_fixed_clip": global_versus_clip,
        "classwise_oracle_diagnostic": class_rows,
        "gate": gate,
        "decision": gate["decision"],
        "next": (
            "eligible for separately approved matched proxy25 only; do not run full VisDA"
            if gate["decision"] == "PASS_OFFLINE_GATE"
            else "stop; do not run proxy or full VisDA training"
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)
    print(f"Wrote text-only prompt contract: {prompt_path}")
    print(f"Wrote label-free signals: {signal_csv}")
    print(f"Wrote label-free tensor signals: {signal_npz}")
    print(f"Locked signals before oracle labels: {lock_path}")
    print(f"Wrote oracle diagnostic: {oracle_csv}")
    print(f"Wrote classwise oracle diagnostic: {class_csv}")
    print(f"Wrote pairwise oracle diagnostic: {pair_csv}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps({"decision": summary["decision"], "checks": gate["checks"]}, indent=2))


if __name__ == "__main__":
    main()
