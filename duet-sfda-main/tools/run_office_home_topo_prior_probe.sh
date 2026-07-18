#!/usr/bin/env bash
set -euo pipefail

# No-adaptation probe for topology-prior calibration on target-Clipart tasks.
# Labels are used only by analyze_topology_prior_calibration.py for reporting.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Export topology-prior probe features: ${task_name}"
  python tools/export_conflict_diagnostics.py \
    --cfg cfgs/office-home/plmatch.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_topology_prior_calibration.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*C_conflicts.npz' \
  --out output/uda/office-home/topology_prior_probe.json
