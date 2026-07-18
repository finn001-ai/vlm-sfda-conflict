#!/usr/bin/env bash
set -euo pipefail

# This probe trains only on counterfactual conflicts synthesized from agreement
# anchors. Ground-truth labels are read solely for the reported diagnostics.

python tools/analyze_counterfactual_adjudicator.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*C_conflicts.npz'
