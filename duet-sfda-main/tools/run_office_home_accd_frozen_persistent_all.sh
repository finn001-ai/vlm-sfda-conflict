#!/usr/bin/env bash
set -euo pipefail

# Full 12-task validation of the best fixed ACCD mechanism. The accd_fp12
# method name isolates these logs from earlier single-task ablations.

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
  echo "==> ACCD frozen+persistent full validation: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/accd_frozen_persistent.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD accd_fp12 \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/accd_fp12/*.txt'
