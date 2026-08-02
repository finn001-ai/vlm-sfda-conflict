#!/usr/bin/env python
"""Gate the compatibility-controlled PCGrad proxy against matched DUET."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_duet_support_conditioned_clip_proxy import analyze


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument(
        "--control-provenance",
        choices=(
            "matched_current_source_and_proxy_hashes",
            "archived_control_without_source_or_proxy_hashes",
        ),
        default="matched_current_source_and_proxy_hashes",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    control = json.loads(Path(args.control_summary).read_text())
    candidate = json.loads(Path(args.candidate_summary).read_text())
    report = analyze(
        control,
        candidate,
        control_provenance=args.control_provenance,
    )
    passed = report["decision"] == "PASS_SUPPORT_CONDITIONED_CLIP_PROXY_GATE"
    report["decision"] = (
        "PASS_PCGRAD_COMPATIBILITY_PROXY_GATE"
        if passed
        else "REJECT_PCGRAD_COMPATIBILITY_PROXY"
    )
    report["candidate"] = (
        "cycle2_full_gradient_compatibility_fraction_for_conflict_pcgrad"
    )
    report["next"] = (
        "eligible for a separately reviewed matched full VisDA run"
        if passed
        else "stop; do not run full VisDA"
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
