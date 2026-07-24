#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

bash tools/run_visda_reciprocal_boundary_proxy25.sh
bash tools/run_office_home_reciprocal_boundary_preflight.sh

python tools/summarize_reciprocal_boundary_preflight.py joint \
  --visda output/uda/VISDA-C/reciprocal_boundary_proxy25_gate.json \
  --office-home output/uda/office-home/reciprocal_boundary_preflight_gate.json \
  --out output/uda/reciprocal_boundary_preflight_gate.json

if grep -q \
  '"decision": "pass_reciprocal_boundary_preflight"' \
  output/uda/reciprocal_boundary_preflight_gate.json; then
  echo "==> Joint preflight passed. Full-data validation is authorized."
else
  echo "==> Joint preflight failed. Do not launch full-data validation." >&2
  exit 2
fi
