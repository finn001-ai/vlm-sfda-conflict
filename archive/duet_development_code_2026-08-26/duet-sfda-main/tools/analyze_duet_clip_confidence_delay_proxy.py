#!/usr/bin/env python3
"""Gate CLIP-confidence admission delay against matched arithmetic DUET."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_duet_support_conditioned_clip_proxy import (
    analyze as analyze_shared_proxy_gate,
)


def analyze(control: dict, candidate: dict, **kwargs) -> dict:
    """Use the same predeclared final-checkpoint gate as prior proxy candidates."""
    report = analyze_shared_proxy_gate(control, candidate, **kwargs)
    report["decision"] = (
        "PASS_CLIP_CONFIDENCE_DELAY_PROXY_GATE"
        if report["decision"] == "PASS_SUPPORT_CONDITIONED_CLIP_PROXY_GATE"
        else "REJECT_CLIP_CONFIDENCE_DELAY_PROXY"
    )
    report["candidate"] = "cycle1_class_balanced_bottom10_clip_confidence_delay"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    control = json.loads(Path(args.control_summary).read_text())
    candidate = json.loads(Path(args.candidate_summary).read_text())
    report = analyze(control, candidate)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
