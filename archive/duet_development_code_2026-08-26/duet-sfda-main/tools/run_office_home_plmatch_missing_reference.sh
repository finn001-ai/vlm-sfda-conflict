#!/usr/bin/env bash
set -euo pipefail

# Complete the same-environment PLMatch reference without rerunning the three
# existing Art-source tasks (AC/AP/AR). Logs use an isolated method directory.

declare -a TASKS=(
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
  echo "==> PLMatch missing reference task: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/plmatch.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD plmatch_ref12 \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/plmatch_ref12/*.txt'
