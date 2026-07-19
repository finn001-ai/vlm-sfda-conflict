#!/usr/bin/env bash
set -euo pipefail

# Stage19-G preflight: one prior gain plus the two largest Stage19 regressions.
declare -a TASKS=("AC 0 1" "PA 2 0" "RA 3 0")

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Stage19-G GTR-only preflight: seed=2022 task=${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_head_pair_feature_gtr.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD temporal_precision_head_seed2022_pair_feature_gtr_preflight \
    SETTING.SEED 2022 \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_pair_feature_gtr_preflight/*.txt' \
  --record-type standard \
  --selection peak \
  > output/uda/office-home/temporal_precision_head_pair_feature_gtr_preflight_accuracy.csv

python tools/summarize_pair_feature_gtr.py \
  --mode preflight \
  --csv output/uda/office-home/temporal_precision_head_pair_feature_gtr_preflight_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_pair_feature_gtr_preflight_summary.json

python tools/summarize_pair_feature_flow.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_pair_feature_gtr_preflight/*.txt' \
  --out output/uda/office-home/temporal_precision_head_pair_feature_gtr_preflight_flow.json
