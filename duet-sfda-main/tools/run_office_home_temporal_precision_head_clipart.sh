#!/usr/bin/env bash
set -euo pipefail

# Target-head decision boundary adaptation.
# Source classifier remains frozen. A target classifier head is initialized
# from the source head, trained from temporal-precision pseudo labels, and
# blended with the source head after the warmup cycle.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Temporal-precision target-head training: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_head.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_temporal_conflict_dynamics.py \
  --glob 'output/uda/office-home/*/temporal_precision_head/temporal_diagnostics/*_cycle*.npz' \
  --out output/uda/office-home/temporal_precision_head_dynamics_probe.json

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head/*.txt' \
  > output/uda/office-home/temporal_precision_head_clipart_accuracy.csv
