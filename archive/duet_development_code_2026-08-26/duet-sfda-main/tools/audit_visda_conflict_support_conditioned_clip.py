#!/usr/bin/env python
"""CPU-only audit of support-conditioned CLIP targets on VisDA conflicts.

The candidate keeps CLIP's relative probability inside the locked task/CLIP
top-2 union and removes only its outside tail. Candidate targets and all gates
are locked before target labels are parsed. Labels are then used exclusively
for oracle logit-direction diagnostics. No image, model, checkpoint, forward,
backward, optimizer, parameter update, or training is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.candidate_set_audit import candidate_coverage, stable_topk  # noqa: E402
from src.utils.candidate_set_gradient_audit import (  # noqa: E402
    kl_logit_descent,
    oracle_ce_logit_descent,
    paired_mean_bootstrap_ci,
    rowwise_oracle_alignment,
)
from src.utils.support_conditioned_clip_audit import (  # noqa: E402
    evaluate_support_conditioned_clip_gate,
    full_target_class_mass_shift_pp,
    negative_first_order_burden,
    normalize_probability_matrix,
    probability_entropy,
    support_conditioned_probability,
)
from tools.audit_visda_conflict_candidate_set_gradient import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    DEFAULT_CANDIDATE_DIR,
    DEFAULT_PROBABILITY_DIR,
    EXPECTED_CONFLICT_SAMPLES,
    EXPECTED_TARGET_SAMPLES,
    _load_class_names,
    _load_label_free_inputs,
    _parse_labels_after_lock,
    _pct,
    _sha256,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = Path(
    "output/uda/VISDA-C/TV/"
    "plmatch_visda_support_conditioned_clip_audit_seed2020/"
    "support_conditioned_clip_audit"
)
METHODS = ("clip", "top1_union", "clip_top2", "top2_union")
CANDIDATE = "top2_union"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-dir", type=Path, default=DEFAULT_PROBABILITY_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-list",
        type=Path,
        default=Path("data/VISDA-C/validation_list.txt"),
    )
    parser.add_argument(
        "--class-names", type=Path, default=Path("data/VISDA-C/classname.txt")
    )
    parser.add_argument("--seed", type=int, default=2020)
    return parser.parse_args()


def _clip_top2_mask(clip_probability: np.ndarray) -> np.ndarray:
    top2 = stable_topk(clip_probability, 2)
    mask = np.zeros_like(clip_probability, dtype=bool)
    mask[np.arange(clip_probability.shape[0])[:, None], top2] = True
    return mask


def _method_metrics(alignment: dict[str, np.ndarray]) -> dict[str, Any]:
    first_order = alignment["first_order"]
    return {
        "mean_cosine": float(alignment["cosine"].mean()),
        "mean_oracle_unit_projection": float(
            alignment["oracle_unit_projection"].mean()
        ),
        "mean_first_order": float(first_order.mean()),
        "mean_direction_norm": float(alignment["candidate_norm"].mean()),
        "negative_first_order_burden": negative_first_order_burden(first_order),
        "helpful_fraction_pct": _pct(first_order > 1e-15),
        "harmful_fraction_pct": _pct(first_order < -1e-15),
        "neutral_fraction_pct": _pct(np.abs(first_order) <= 1e-15),
    }


def _comparison(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    *,
    seed: int,
) -> dict[str, Any]:
    result = {}
    for offset, metric in enumerate(
        ("cosine", "oracle_unit_projection", "first_order")
    ):
        difference = candidate[metric] - baseline[metric]
        ci = paired_mean_bootstrap_ci(
            difference,
            repeats=BOOTSTRAP_REPEATS,
            seed=seed + offset,
        )
        result[metric] = {
            "mean_difference": float(difference.mean()),
            "paired_bootstrap_95_ci": [float(ci[0]), float(ci[1])],
        }
    return result


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    oracle = summary["oracle_metrics"]
    label_free = summary["label_free_metrics"]
    lines = [
        "# VisDA Conflict Support-Conditioned CLIP Audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "The candidate conditions CLIP probability on the locked task/CLIP",
        "top-2 union. It does not choose a task/CLIP winner or add a candidate",
        "loss. All targets were locked before labels were parsed.",
        "",
        "## Label-free target diagnostics",
        "",
        "| Target | Retained CLIP mass | Entropy | Full mass shift |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        metric = label_free["targets"][method]
        lines.append(
            f"| {method} | {metric['mean_retained_clip_mass']:.8f} | "
            f"{metric['mean_entropy']:.8f} | "
            f"{metric['max_abs_full_target_class_mass_shift_pp']:.6f} pp |"
        )
    lines.extend(
        [
            "",
            "## Oracle logit-direction diagnostics",
            "",
            "| Target | Cosine | Unit-oracle projection | First-order | Negative burden |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        metric = oracle["methods"][method]
        lines.append(
            f"| {method} | {metric['mean_cosine']:.8f} | "
            f"{metric['mean_oracle_unit_projection']:.8f} | "
            f"{metric['mean_first_order']:.8f} | "
            f"{metric['negative_first_order_burden']:.8f} |"
        )
    lines.extend(["", "## Gate", ""])
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in summary["gate"]["checks"].items()
    )
    lines.extend(
        [
            "",
            "Passing authorizes only one matched proxy design. It never starts",
            "or authorizes VisDA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    for path in (args.target_list, args.class_names):
        if not path.is_file():
            raise FileNotFoundError(f"Missing support-conditioned input: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")

    class_names = _load_class_names(args.class_names)
    source = _load_label_free_inputs(args.probability_dir, args.candidate_dir)
    if not all(source["checks"].values()):
        failed = [name for name, passed in source["checks"].items() if not passed]
        raise RuntimeError(f"Locked support-conditioned input failed: {failed}")

    task_probability = normalize_probability_matrix(
        source["task_probability"], name="task_probability"
    )
    clip_probability = normalize_probability_matrix(
        source["clip_probability"], name="clip_probability"
    )
    all_support = np.ones_like(clip_probability, dtype=bool)
    supports = {
        "clip": all_support,
        "top1_union": source["top1_mask"],
        "clip_top2": _clip_top2_mask(clip_probability),
        "top2_union": source["top2_mask"],
    }
    conditioned = {
        method: support_conditioned_probability(clip_probability, supports[method])
        for method in METHODS
    }
    targets = {method: conditioned[method]["probability"] for method in METHODS}
    mass_shift = {
        method: full_target_class_mass_shift_pp(
            targets[method],
            clip_probability,
            full_target_samples=EXPECTED_TARGET_SAMPLES,
        )
        for method in METHODS
    }
    target_checks = {
        f"{method}_finite": bool(np.isfinite(targets[method]).all())
        for method in METHODS
    }
    target_checks.update(
        {
            f"{method}_normalized": bool(
                np.allclose(targets[method].sum(axis=1), 1.0, atol=1e-12)
            )
            for method in METHODS
        }
    )
    target_checks.update(
        {
            f"{method}_outside_support_zero": bool(
                np.all(targets[method][~supports[method]] == 0.0)
            )
            for method in METHODS
        }
    )
    target_checks.update(
        {
            "clip_target_reproduced": np.allclose(
                targets["clip"], clip_probability, atol=1e-12, rtol=1e-12
            ),
            "top2_union_contains_clip_top2": bool(
                np.all(~supports["clip_top2"] | supports["top2_union"])
            ),
            "top2_union_contains_top1_union": bool(
                np.all(~supports["top1_union"] | supports["top2_union"])
            ),
            "candidate_top1_matches_clip": np.array_equal(
                targets[CANDIDATE].argmax(axis=1),
                clip_probability.argmax(axis=1),
            ),
        }
    )
    input_contract_valid = all({**source["checks"], **target_checks}.values())
    if not input_contract_valid:
        failed = [name for name, passed in target_checks.items() if not passed]
        raise RuntimeError(f"Support-conditioned target construction failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = "visda_conflict_support_conditioned_clip"
    target_path = args.output_dir / f"{stem}_label_free.npz"
    mass_path = args.output_dir / f"{stem}_class_mass.csv"
    lock_path = args.output_dir / f"{stem}_signal_lock.json"
    oracle_path = args.output_dir / f"{stem}_oracle_diagnostic.csv"
    class_path = args.output_dir / f"{stem}_classwise_oracle_diagnostic.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    markdown_path = args.output_dir / f"{stem}_summary.md"

    np.savez_compressed(
        target_path,
        index=source["index"],
        clip_probability=targets["clip"],
        top1_union_probability=targets["top1_union"],
        clip_top2_probability=targets["clip_top2"],
        top2_union_probability=targets["top2_union"],
        top1_union_mask=supports["top1_union"],
        clip_top2_mask=supports["clip_top2"],
        top2_union_mask=supports["top2_union"],
    )
    mass_rows = []
    for class_index, class_name in enumerate(class_names):
        row: dict[str, Any] = {"class_index": class_index, "class": class_name}
        for method in METHODS:
            row[f"{method}_conflict_mean_probability_pct"] = float(
                targets[method][:, class_index].mean() * 100.0
            )
            row[f"{method}_full_target_mass_shift_pp"] = float(
                mass_shift[method][class_index]
            )
        mass_rows.append(row)
    _write_csv(mass_path, mass_rows)

    label_free_metrics = {
        "conflict_samples": EXPECTED_CONFLICT_SAMPLES,
        "full_target_samples": EXPECTED_TARGET_SAMPLES,
        "targets": {
            method: {
                "mean_retained_clip_mass": float(
                    conditioned[method]["retained_mass"].mean()
                ),
                "mean_entropy": float(probability_entropy(targets[method]).mean()),
                "max_abs_full_target_class_mass_shift_pp": float(
                    np.abs(mass_shift[method]).max()
                ),
                "class_mass_shift_pp": [float(value) for value in mass_shift[method]],
                "top1_matches_clip": bool(
                    np.array_equal(
                        targets[method].argmax(axis=1),
                        clip_probability.argmax(axis=1),
                    )
                ),
            }
            for method in METHODS
        },
    }
    signal_lock = {
        "phase": "LABEL_FREE_SUPPORT_CONDITIONED_CLIP_LOCK",
        "contains_target_labels": False,
        "contains_target_paths": False,
        "target_list_read_before_lock": False,
        "oracle_labels_parsed_after_this_manifest": True,
        "candidate": "clip_probability_conditioned_on_task_clip_top2_union",
        "candidate_contract": {
            "support": "task top-2 union CLIP top-2",
            "target": "CLIP probability divided by its mass inside support",
            "outside_support_probability": 0.0,
            "inside_support_relative_clip_probability_preserved": True,
            "task_probability_used_as_target_weight": False,
            "hard_pseudo_label_changed": False,
            "loss_term_added": False,
            "fitted_thresholds": False,
            "target_labels": False,
        },
        "matched_controls": {
            "clip": "released DUET CLIP KL target",
            "top1_union": "CLIP conditioned on task/CLIP top-1 union",
            "clip_top2": "CLIP conditioned on its own top-2",
        },
        "predeclared_oracle_gate": {
            "candidate_minus_clip": (
                "positive mean and positive paired 95% CI lower bound for "
                "cosine, oracle-unit projection, and first-order benefit"
            ),
            "candidate_minus_top1_union_cosine": (
                "positive mean and positive paired 95% CI lower bound"
            ),
            "minimum_class_first_order_delta_vs_clip": ">= 0",
            "negative_first_order_burden": (
                "not worse than CLIP/top1 union and better than CLIP-top2"
            ),
            "max_full_target_equivalent_class_mass_shift_pp": "<= 1.0",
            "candidate_mass_shift": "below top1 union and CLIP-top2",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
        },
        "input_contract_checks": {**source["checks"], **target_checks},
        "inputs": {
            name: {"path": str(source["paths"][name]), "sha256": digest}
            for name, digest in source["hashes"].items()
        },
        "target_list_expected_opaque_sha256": source["expected_target_hash"],
        "class_names_sha256": _sha256(args.class_names),
        "label_free_metrics": label_free_metrics,
        "outputs": {
            "target_npz": {"path": str(target_path), "sha256": _sha256(target_path)},
            "class_mass_csv": {"path": str(mass_path), "sha256": _sha256(mass_path)},
        },
        "contract_sha256": {
            "src/utils/support_conditioned_clip_audit.py": _sha256(
                REPO_ROOT / "src/utils/support_conditioned_clip_audit.py"
            ),
            "tools/audit_visda_conflict_support_conditioned_clip.py": _sha256(
                Path(__file__).resolve()
            ),
            "tools/audit_visda_conflict_candidate_set_gradient.py": _sha256(
                REPO_ROOT / "tools/audit_visda_conflict_candidate_set_gradient.py"
            ),
            "src/methods/oh/plmatch.py": _sha256(
                REPO_ROOT / "src/methods/oh/plmatch.py"
            ),
        },
    }
    lock_path.write_text(json.dumps(signal_lock, indent=2) + "\n")

    # Oracle phase: target-list content is first read after the signal lock.
    target_list_hash = _sha256(args.target_list)
    target_hash_matches = target_list_hash == source["expected_target_hash"]
    if not target_hash_matches:
        raise RuntimeError("Target-list hash does not match locked probability input")
    labels = _parse_labels_after_lock(args.target_list)[source["index"]]
    oracle_direction = oracle_ce_logit_descent(task_probability, labels)
    directions = {
        method: kl_logit_descent(task_probability, targets[method])
        for method in METHODS
    }
    alignments = {
        method: rowwise_oracle_alignment(directions[method], oracle_direction)
        for method in METHODS
    }
    method_metrics = {method: _method_metrics(alignments[method]) for method in METHODS}
    comparisons = {
        "versus_clip": _comparison(
            alignments[CANDIDATE], alignments["clip"], seed=args.seed
        ),
        "versus_top1_union": _comparison(
            alignments[CANDIDATE], alignments["top1_union"], seed=args.seed + 10
        ),
        "versus_clip_top2": _comparison(
            alignments[CANDIDATE], alignments["clip_top2"], seed=args.seed + 20
        ),
    }
    top1_coverage = candidate_coverage(source["top1_mask"], labels)
    top2_coverage = candidate_coverage(source["top2_mask"], labels)

    class_rows = []
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        row: dict[str, Any] = {
            "class_index": class_index,
            "class": class_name,
            "conflict_samples": int(mask.sum()),
            "top1_union_oracle_coverage_pct": _pct(top1_coverage[mask]),
            "top2_union_oracle_coverage_pct": _pct(top2_coverage[mask]),
        }
        for method in METHODS:
            first_order = alignments[method]["first_order"][mask]
            row.update(
                {
                    f"{method}_mean_cosine": float(
                        alignments[method]["cosine"][mask].mean()
                    ),
                    f"{method}_mean_oracle_unit_projection": float(
                        alignments[method]["oracle_unit_projection"][mask].mean()
                    ),
                    f"{method}_mean_first_order": float(first_order.mean()),
                    f"{method}_negative_first_order_burden": (
                        negative_first_order_burden(first_order)
                    ),
                    f"{method}_harmful_fraction_pct": _pct(first_order < -1e-15),
                }
            )
        row.update(
            {
                "candidate_minus_clip_mean_cosine": (
                    row["top2_union_mean_cosine"] - row["clip_mean_cosine"]
                ),
                "candidate_minus_clip_mean_oracle_unit_projection": (
                    row["top2_union_mean_oracle_unit_projection"]
                    - row["clip_mean_oracle_unit_projection"]
                ),
                "candidate_minus_clip_mean_first_order": (
                    row["top2_union_mean_first_order"] - row["clip_mean_first_order"]
                ),
            }
        )
        class_rows.append(row)
    _write_csv(class_path, class_rows)

    oracle_rows = []
    for position, target_index in enumerate(source["index"]):
        label = int(labels[position])
        row: dict[str, Any] = {
            "index": int(target_index),
            "label": label,
            "label_name": class_names[label],
            "top1_union_covers_label": bool(top1_coverage[position]),
            "top2_union_covers_label": bool(top2_coverage[position]),
        }
        for method in METHODS:
            row.update(
                {
                    f"{method}_cosine": float(alignments[method]["cosine"][position]),
                    f"{method}_oracle_unit_projection": float(
                        alignments[method]["oracle_unit_projection"][position]
                    ),
                    f"{method}_first_order": float(
                        alignments[method]["first_order"][position]
                    ),
                }
            )
        row.update(
            {
                "candidate_minus_clip_cosine": (
                    row["top2_union_cosine"] - row["clip_cosine"]
                ),
                "candidate_minus_clip_oracle_unit_projection": (
                    row["top2_union_oracle_unit_projection"]
                    - row["clip_oracle_unit_projection"]
                ),
                "candidate_minus_clip_first_order": (
                    row["top2_union_first_order"] - row["clip_first_order"]
                ),
            }
        )
        oracle_rows.append(row)
    _write_csv(oracle_path, oracle_rows)

    class_deltas = [
        float(row["candidate_minus_clip_mean_first_order"]) for row in class_rows
    ]
    gate = evaluate_support_conditioned_clip_gate(
        input_contract_valid=input_contract_valid and target_hash_matches,
        versus_clip=comparisons["versus_clip"],
        versus_top1_union=comparisons["versus_top1_union"],
        minimum_class_first_order_delta_vs_clip=min(class_deltas),
        candidate_negative_burden=method_metrics[CANDIDATE][
            "negative_first_order_burden"
        ],
        clip_negative_burden=method_metrics["clip"]["negative_first_order_burden"],
        top1_union_negative_burden=method_metrics["top1_union"][
            "negative_first_order_burden"
        ],
        clip_top2_negative_burden=method_metrics["clip_top2"][
            "negative_first_order_burden"
        ],
        candidate_max_full_mass_shift_pp=label_free_metrics["targets"][CANDIDATE][
            "max_abs_full_target_class_mass_shift_pp"
        ],
        top1_union_max_full_mass_shift_pp=label_free_metrics["targets"]["top1_union"][
            "max_abs_full_target_class_mass_shift_pp"
        ],
        clip_top2_max_full_mass_shift_pp=label_free_metrics["targets"]["clip_top2"][
            "max_abs_full_target_class_mass_shift_pp"
        ],
        candidate_top1_matches_clip=target_checks["candidate_top1_matches_clip"],
    )
    summary = {
        "dataset": "VISDA-C",
        "task": "train->validation",
        "seed": args.seed,
        "oracle_diagnostic": True,
        "labels_used_only_after_signal_lock": True,
        "signal_lock": str(lock_path),
        "signal_lock_sha256": _sha256(lock_path),
        "candidate": signal_lock["candidate"],
        "input_contract": {
            "passed": input_contract_valid and target_hash_matches,
            "checks": {
                **source["checks"],
                **target_checks,
                "target_list_hash_matches_after_lock": target_hash_matches,
            },
        },
        "label_free_metrics": label_free_metrics,
        "oracle_metrics": {
            "metric_definitions": {
                "first_order": (
                    "KL logit descent dot oracle CE descent; positive predicts "
                    "an infinitesimal oracle log-probability increase"
                ),
                "negative_first_order_burden": (
                    "mean min(first_order, 0); closer to zero is safer"
                ),
            },
            "top1_union_coverage_pct": _pct(top1_coverage),
            "top2_union_coverage_pct": _pct(top2_coverage),
            "methods": method_metrics,
            "comparisons": comparisons,
            "minimum_class_first_order_delta_vs_clip": min(class_deltas),
            "class_macro_first_order_delta_vs_clip": float(np.mean(class_deltas)),
            "classwise": class_rows,
        },
        "gate": gate,
        "decision": gate["decision"],
        "next": (
            "eligible to design one matched proxy; no training is authorized"
            if gate["decision"] == "PASS_SUPPORT_CONDITIONED_CLIP_PREFLIGHT"
            else "reject support-conditioned CLIP; do not run proxy or full training"
        ),
        "historical_distinction": (
            "Unlike Stage3 candidate KL, this target does not balance task and "
            "CLIP top-1 or add candidate loss; task ranks define support only."
        ),
        "scope_limit": (
            "Exact only for task-logit directions at the locked first-cycle view; "
            "does not identify parameter-gradient interactions or later dynamics."
        ),
        "safety_contract": {
            "target_images_loaded": False,
            "model_checkpoint_loads": 0,
            "model_forward_calls": 0,
            "optimizer_constructed": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "model_parameters_updated": False,
            "training_code_modified": False,
            "training_authorized": False,
        },
        "runtime_seconds": float(time.monotonic() - started),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, markdown_path)

    for label, output in (
        ("label-free targets", target_path),
        ("label-free class mass", mass_path),
        ("signal lock", lock_path),
        ("oracle diagnostic", oracle_path),
        ("classwise oracle diagnostic", class_path),
        ("summary", summary_path),
        ("markdown summary", markdown_path),
    ):
        print(f"Wrote {label}: {output}")
    print(
        json.dumps(
            {"decision": summary["decision"], "checks": gate["checks"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
