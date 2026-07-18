#!/usr/bin/env bash
set -euo pipefail

# Diagnostic run for temporal conflict dynamics under the current both_prior path.
# This does not add a loss or change pseudo-label rules; it exports per-cycle
# prediction trajectories for later analysis.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Temporal conflict dynamics probe: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_probe.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_temporal_conflict_dynamics.py \
  --glob 'output/uda/office-home/*/temporal_probe/temporal_diagnostics/*_cycle*.npz' \
  --out output/uda/office-home/temporal_conflict_dynamics_probe.json
