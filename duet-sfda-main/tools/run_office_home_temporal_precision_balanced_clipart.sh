#!/usr/bin/env bash
set -euo pipefail

# Target-prior balanced pseudo-label admission.
# Stable temporal pseudo labels are kept, then class-balanced under the
# calibrated target teacher prior before entering supervised CE.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Temporal-precision balanced training: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_balanced.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_temporal_conflict_dynamics.py \
  --glob 'output/uda/office-home/*/temporal_precision_balanced/temporal_diagnostics/*_cycle*.npz' \
  --out output/uda/office-home/temporal_precision_balanced_dynamics_probe.json

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_balanced/*.txt' \
  > output/uda/office-home/temporal_precision_balanced_clipart_accuracy.csv
