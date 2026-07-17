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
        "cand_tau": re.compile(r"^\s*CAND_TAU:\s*([^\s]+)", re.MULTILINE),
        "cand_weight": re.compile(r"^\s*CAND_WEIGHT:\s*([^\s]+)", re.MULTILINE),
        "tau_low": re.compile(r"^\s*TAU_LOW:\s*([^\s]+)", re.MULTILINE),
        "promote_k": re.compile(r"^\s*PROMOTE_K:\s*([^\s]+)", re.MULTILINE),
    }
    print("method,task,cycle,iter,accuracy,cand_par,cand_tau,cand_weight,tau_low,promote_k,log")
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
            f"{cfg_values['cand_par']},{cfg_values['cand_tau']},{cfg_values['cand_weight']},"
            f"{cfg_values['tau_low']},{cfg_values['promote_k']},{path}"
        )


if __name__ == "__main__":
    main()
