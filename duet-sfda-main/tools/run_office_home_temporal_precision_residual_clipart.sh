#!/usr/bin/env bash
set -euo pipefail

# Temporal precision memory probe.
# The change is pseudo-label admission/memory: agreement labels must become
# temporally stable before they enter supervised CE after the warmup cycle.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Temporal-precision residual training: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_residual.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_temporal_conflict_dynamics.py \
  --glob 'output/uda/office-home/*/temporal_precision_residual/temporal_diagnostics/*_cycle*.npz' \
  --out output/uda/office-home/temporal_precision_residual_dynamics_probe.json

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_residual/*.txt' \
  > output/uda/office-home/temporal_precision_residual_clipart_accuracy.csv
