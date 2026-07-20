#!/usr/bin/env bash
set -euo pipefail

# Reproduce the archived Stage14 seed-2020 run in an isolated output directory.
declare -a TASKS=(
  "AC 0 1" "AP 0 2" "AR 0 3" "CA 1 0" "CP 1 2" "CR 1 3"
  "PA 2 0" "PC 2 1" "PR 2 3" "RA 3 0" "RC 3 1" "RP 3 2"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Reproduce Stage14: seed=2020 task=${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_head.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD temporal_precision_head_seed2020_reproduce \
    SETTING.SEED 2020 SETTING.S "$s" SETTING.T "$t"
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2020_reproduce/*.txt' \
  --selection peak \
  > output/uda/office-home/temporal_precision_head_seed2020_reproduce_accuracy.csv

python tools/compare_stage14_seed2020_reproduction.py \
  --csv output/uda/office-home/temporal_precision_head_seed2020_reproduce_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_seed2020_reproduce_summary.json
