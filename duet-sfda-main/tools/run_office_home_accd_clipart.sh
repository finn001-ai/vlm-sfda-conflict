#!/usr/bin/env bash
set -euo pipefail

# Train ACCD only on the three target-Clipart tasks after the diffusion probe.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> ACCD target-Clipart run: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/accd.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/accd/*.txt'
