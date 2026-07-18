#!/usr/bin/env bash
set -euo pipefail

# Full 12-task validation of temporal-precision target-head adaptation.
# The method name isolates this full validation from the initial Clipart probe.

declare -a TASKS=(
  "AC 0 1"
  "AP 0 2"
  "AR 0 3"
  "CA 1 0"
  "CP 1 2"
  "CR 1 3"
  "PA 2 0"
  "PC 2 1"
  "PR 2 3"
  "RA 3 0"
  "RC 3 1"
  "RP 3 2"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Temporal-precision target-head full Office-Home run: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_head.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD temporal_precision_head_all \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_temporal_conflict_dynamics.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_all/temporal_diagnostics/*_cycle*.npz' \
  --out output/uda/office-home/temporal_precision_head_all_dynamics_probe.json

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_all/*.txt' \
  > output/uda/office-home/temporal_precision_head_all_accuracy.csv
