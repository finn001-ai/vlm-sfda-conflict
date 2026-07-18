#!/usr/bin/env bash
set -euo pipefail

# First training run for graph-fused soft teacher with temporal diagnostics.
# Do not expand beyond target-Clipart unless at least two tasks improve over
# both_prior and at least one materially closes the DUET-paper gap.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Graph-temporal training: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/graph_temporal.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_temporal_conflict_dynamics.py \
  --glob 'output/uda/office-home/*/graph_temporal/temporal_diagnostics/*_cycle*.npz' \
  --out output/uda/office-home/graph_temporal_dynamics_probe.json
