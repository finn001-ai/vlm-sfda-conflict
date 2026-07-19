#!/usr/bin/env bash
set -euo pipefail

# Cheapest robustness gate: rerun only the previously worst adaptation seed.
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
  echo "==> Target-head EMA worst-seed probe: seed=2022 task=${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_head_ema.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD temporal_precision_head_seed2022_ema_probe \
    SETTING.SEED 2022 \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_ema_probe/*.txt' \
  > output/uda/office-home/temporal_precision_head_ema_seed2022_accuracy.csv

python tools/summarize_office_home_seed_sweep.py \
  --csv output/uda/office-home/temporal_precision_head_ema_seed2022_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_ema_seed2022_summary.json
