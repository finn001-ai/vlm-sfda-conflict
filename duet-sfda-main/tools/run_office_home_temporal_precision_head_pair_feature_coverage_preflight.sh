#!/usr/bin/env bash
set -euo pipefail

# Stage19-C preflight: verify exact Stage14 fallback on the three low-rank
# target-Art tasks before spending a complete 12-task run.
declare -a TASKS=("CA 1 0" "PA 2 0" "RA 3 0")

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Stage19-C coverage preflight: seed=2022 task=${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_head_pair_feature_coverage.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD temporal_precision_head_seed2022_pair_feature_coverage_preflight \
    SETTING.SEED 2022 \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_pair_feature_coverage_preflight/*.txt' \
  --record-type standard \
  --selection peak \
  > output/uda/office-home/temporal_precision_head_pair_feature_coverage_preflight_accuracy.csv

python tools/summarize_pair_feature_coverage_preflight.py \
  --csv output/uda/office-home/temporal_precision_head_pair_feature_coverage_preflight_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_pair_feature_coverage_preflight_summary.json

python tools/summarize_pair_feature_flow.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_pair_feature_coverage_preflight/*.txt' \
  --min-active-rank 8 \
  --allow-fallback \
  --out output/uda/office-home/temporal_precision_head_pair_feature_coverage_preflight_flow.json
