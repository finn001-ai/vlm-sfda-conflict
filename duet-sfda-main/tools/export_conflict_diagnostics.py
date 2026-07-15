#!/usr/bin/env python
"""Export source-model vs CLIP conflict diagnostics for DUET/PLMatch.

This script does no adaptation. It loads the source checkpoint, runs the
source/task model and CLIP on the target domain, then writes per-sample CSV and
summary files. Use it before designing a method to verify whether conflict
samples contain recoverable target-domain signal.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import clip  # noqa: E402
from conf import cfg, load_cfg_from_args  # noqa: E402
from src.methods.oh.plmatch import clip_pre_text, data_load  # noqa: E402
from src.models import network  # noqa: E402


def _require_file(path: str | Path, purpose: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
    return path


def _prepare_cfg() -> None:
    load_cfg_from_args()
    cfg.type = cfg.domain
    cfg.t_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + "/" + cfg.domain[cfg.SETTING.T] + "_list.txt"
    cfg.test_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + "/" + cfg.domain[cfg.SETTING.T] + "_list.txt"
    cfg.s_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + "/" + cfg.domain[cfg.SETTING.S] + "_list.txt"
    cfg.savename = cfg.MODEL.METHOD

    _require_file(cfg.t_dset_path, "target-domain list file")
    _require_file(cfg.name_file, "class-name file")
    _require_file(Path(cfg.output_dir_src) / "source_F.pt", "source feature checkpoint")
    _require_file(Path(cfg.output_dir_src) / "source_B.pt", "source bottleneck checkpoint")
    _require_file(Path(cfg.output_dir_src) / "source_C.pt", "source classifier checkpoint")


def _build_source_model(device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module]:
    if cfg.MODEL.ARCH[:3] == "res":
        net_f = network.ResBase(res_name=cfg.MODEL.ARCH).to(device)
    elif cfg.MODEL.ARCH[:3] == "vgg":
        net_f = network.VGGBase(vgg_name=cfg.MODEL.ARCH).to(device)
    else:
        raise ValueError(f"Unsupported source architecture: {cfg.MODEL.ARCH}")

    net_b = network.feat_bottleneck(
        type="bn", feature_dim=net_f.in_features, bottleneck_dim=cfg.bottleneck
    ).to(device)
    net_c = network.feat_classifier(
        type="wn", class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck
    ).to(device)

    ckpt_dir = Path(cfg.output_dir_src)
    net_f.load_state_dict(torch.load(ckpt_dir / "source_F.pt", map_location=device))
    net_b.load_state_dict(torch.load(ckpt_dir / "source_B.pt", map_location=device))
    net_c.load_state_dict(torch.load(ckpt_dir / "source_C.pt", map_location=device))

    net_f.eval()
    net_b.eval()
    net_c.eval()
    return net_f, net_b, net_c


def _load_class_names() -> list[str]:
    with open(cfg.name_file) as f:
        names = [line.strip().replace("_", " ") for line in f if line.strip()]
    if len(names) != cfg.class_num:
        logging.warning("Class-name count %s != cfg.class_num %s", len(names), cfg.class_num)
    return names


def _case(source_correct: bool, clip_correct: bool, agree: bool) -> str:
    if agree:
        if source_correct and clip_correct:
            return "agree_both_correct"
        return "agree_both_wrong"
    if source_correct and not clip_correct:
        return "conflict_source_correct_clip_wrong"
    if clip_correct and not source_correct:
        return "conflict_source_wrong_clip_correct"
    return "conflict_both_wrong"


def _topk_string(values: torch.Tensor, class_names: list[str], k: int = 5) -> str:
    k = min(k, values.numel())
    inds = torch.topk(values, k=k).indices.tolist()
    return "|".join(f"{idx}:{class_names[idx] if idx < len(class_names) else idx}" for idx in inds)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    cases: dict[str, int] = {}
    for row in rows:
        cases[row["case"]] = cases.get(row["case"], 0) + 1

    agree = sum(1 for row in rows if row["agree"])
    conflict = total - agree
    source_correct = sum(1 for row in rows if row["source_correct"])
    clip_correct = sum(1 for row in rows if row["clip_correct"])
    source_better_conflict = cases.get("conflict_source_correct_clip_wrong", 0)
    clip_better_conflict = cases.get("conflict_source_wrong_clip_correct", 0)
    useful_conflict = source_better_conflict + clip_better_conflict

    def pct(n: int, denom: int = total) -> float:
        return round(100.0 * n / denom, 4) if denom else 0.0

    return {
        "dataset": cfg.SETTING.DATASET,
        "source_domain": cfg.domain[cfg.SETTING.S],
        "target_domain": cfg.domain[cfg.SETTING.T],
        "total_samples": total,
        "source_accuracy": pct(source_correct),
        "clip_accuracy": pct(clip_correct),
        "agreement_samples": agree,
        "agreement_rate": pct(agree),
        "conflict_samples": conflict,
        "conflict_rate": pct(conflict),
        "useful_conflict_samples": useful_conflict,
        "useful_conflict_rate_among_all": pct(useful_conflict),
        "useful_conflict_rate_among_conflicts": pct(useful_conflict, conflict),
        "cases": {key: {"count": value, "percent": pct(value)} for key, value in sorted(cases.items())},
    }


def _write_summary_md(summary: dict[str, Any], path: Path) -> None:
    cases = summary["cases"]
    lines = [
        "# Conflict Diagnostics",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Task: `{summary['source_domain']} -> {summary['target_domain']}`",
        f"Total samples: `{summary['total_samples']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Source accuracy | {summary['source_accuracy']:.2f}% |",
        f"| CLIP accuracy | {summary['clip_accuracy']:.2f}% |",
        f"| Agreement rate | {summary['agreement_rate']:.2f}% |",
        f"| Conflict rate | {summary['conflict_rate']:.2f}% |",
        f"| Useful conflict / all | {summary['useful_conflict_rate_among_all']:.2f}% |",
        f"| Useful conflict / conflicts | {summary['useful_conflict_rate_among_conflicts']:.2f}% |",
        "",
        "| Case | Count | Percent |",
        "|---|---:|---:|",
    ]
    for key, value in cases.items():
        lines.append(f"| {key} | {value['count']} | {value['percent']:.2f}% |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    _prepare_cfg()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this codebase's Office-Home pipeline.")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    device = torch.device("cuda")
    torch.manual_seed(cfg.SETTING.SEED)
    torch.cuda.manual_seed(cfg.SETTING.SEED)

    class_names = _load_class_names()
    loaders = data_load(cfg)
    test_loader = loaders["test_aug"]

    net_f, net_b, net_c = _build_source_model(device)
    clip_model, _, _ = clip.load(cfg.ACTIVE.ARCH, device=device)
    clip_model.float()
    clip_model.eval()
    text_inputs = clip_pre_text(cfg).to(device)

    rows: list[dict[str, Any]] = []
    image_records = test_loader.dataset.imgs

    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            weak_x = inputs[1].to(device)
            labels = labels.long()
            indices = indices.long()

            source_logits = net_c(net_b(net_f(weak_x)))
            clip_logits, _ = clip_model(weak_x, text_inputs)

            source_probs = torch.softmax(source_logits, dim=1).cpu()
            clip_probs = torch.softmax(clip_logits, dim=1).cpu()
            source_conf, source_pred = source_probs.max(dim=1)
            clip_conf, clip_pred = clip_probs.max(dim=1)

            for b in range(labels.size(0)):
                idx = int(indices[b].item())
                label = int(labels[b].item())
                sp = int(source_pred[b].item())
                cp = int(clip_pred[b].item())
                source_ok = sp == label
                clip_ok = cp == label
                agree = sp == cp
                image_path = image_records[idx][0]

                rows.append(
                    {
                        "index": idx,
                        "path": image_path,
                        "label": label,
                        "label_name": class_names[label] if label < len(class_names) else str(label),
                        "source_pred": sp,
                        "source_pred_name": class_names[sp] if sp < len(class_names) else str(sp),
                        "source_conf": round(float(source_conf[b].item()), 6),
                        "clip_pred": cp,
                        "clip_pred_name": class_names[cp] if cp < len(class_names) else str(cp),
                        "clip_conf": round(float(clip_conf[b].item()), 6),
                        "agree": agree,
                        "source_correct": source_ok,
                        "clip_correct": clip_ok,
                        "case": _case(source_ok, clip_ok, agree),
                        "source_top5": _topk_string(source_probs[b], class_names),
                        "clip_top5": _topk_string(clip_probs[b], class_names),
                    }
                )

    rows.sort(key=lambda row: row["index"])
    out_dir = Path(cfg.output_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.domain[cfg.SETTING.S][0].upper()}{cfg.domain[cfg.SETTING.T][0].upper()}_conflicts"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}_summary.json"
    md_path = out_dir / f"{stem}_summary.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    json_path.write_text(json.dumps(summary, indent=2))
    _write_summary_md(summary, md_path)

    print(f"Wrote per-sample CSV: {csv_path}")
    print(f"Wrote summary JSON: {json_path}")
    print(f"Wrote summary Markdown: {md_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
