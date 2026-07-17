#!/usr/bin/env python
"""Extract final task accuracy from DUET/DCCL log files."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/*/*.txt",
        help="Glob for log txt files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(p) for p in glob.glob(args.glob))
    pattern = re.compile(r"Task:\s*([A-Z]{2}),\s*Iter:\s*(\d+)/(\d+);\s*Cycle:\s*(\d+)/(\d+);\s*Accuracy\s*=\s*([0-9.]+)%")
    cfg_patterns = {
        "cand_par": re.compile(r"^\s*CAND_PAR:\s*([^\s]+)", re.MULTILINE),
        "cand_start_cycle": re.compile(r"^\s*CAND_START_CYCLE:\s*([^\s]+)", re.MULTILINE),
        "cand_tau": re.compile(r"^\s*CAND_TAU:\s*([^\s]+)", re.MULTILINE),
        "cand_weight": re.compile(r"^\s*CAND_WEIGHT:\s*([^\s]+)", re.MULTILINE),
        "kl_mode": re.compile(r"^\s*KL_MODE:\s*([^\s]+)", re.MULTILINE),
        "kl_candidate": re.compile(r"^\s*KL_CANDIDATE:\s*([^\s]+)", re.MULTILINE),
        "calib_mode": re.compile(r"^\s*CALIB_MODE:\s*([^\s]+)", re.MULTILINE),
        "calib_power": re.compile(r"^\s*CALIB_POWER:\s*([^\s]+)", re.MULTILINE),
        "calib_auto_lambda": re.compile(r"^\s*CALIB_AUTO_LAMBDA:\s*([^\s]+)", re.MULTILINE),
        "pl_expand": re.compile(r"^\s*PL_EXPAND:\s*([^\s]+)", re.MULTILINE),
        "pl_topk_per_class": re.compile(r"^\s*PL_TOPK_PER_CLASS:\s*([^\s]+)", re.MULTILINE),
        "pl_min_conf": re.compile(r"^\s*PL_MIN_CONF:\s*([^\s]+)", re.MULTILINE),
        "tau_low": re.compile(r"^\s*TAU_LOW:\s*([^\s]+)", re.MULTILINE),
        "promote_k": re.compile(r"^\s*PROMOTE_K:\s*([^\s]+)", re.MULTILINE),
    }
    print("method,task,cycle,iter,accuracy,cand_par,cand_start_cycle,cand_tau,cand_weight,kl_mode,kl_candidate,calib_mode,calib_power,calib_auto_lambda,pl_expand,pl_topk_per_class,pl_min_conf,tau_low,promote_k,log")
    for path in paths:
        text = path.read_text(errors="ignore")
        matches = pattern.findall(text)
        if not matches:
            continue
        task, iter_num, max_iter, cycle, max_cycle, acc = matches[-1]
        parts = path.parts
        method = parts[-2] if len(parts) >= 2 else "unknown"
        cfg_values = {}
        for key, cfg_pattern in cfg_patterns.items():
            match = cfg_pattern.search(text)
            cfg_values[key] = match.group(1) if match else ""
        print(
            f"{method},{task},{cycle}/{max_cycle},{iter_num}/{max_iter},{acc},"
            f"{cfg_values['cand_par']},{cfg_values['cand_start_cycle']},"
            f"{cfg_values['cand_tau']},{cfg_values['cand_weight']},"
            f"{cfg_values['kl_mode']},{cfg_values['kl_candidate']},"
            f"{cfg_values['calib_mode']},{cfg_values['calib_power']},{cfg_values['calib_auto_lambda']},"
            f"{cfg_values['pl_expand']},{cfg_values['pl_topk_per_class']},{cfg_values['pl_min_conf']},"
            f"{cfg_values['tau_low']},{cfg_values['promote_k']},{path}"
        )


if __name__ == "__main__":
    main()
