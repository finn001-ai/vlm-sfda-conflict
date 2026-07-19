#!/usr/bin/env bash
set -euo pipefail

# Complete Stage19-C gate after the target-Art coverage preflight passes.
declare -a TASKS=(
  "AC 0 1" "AP 0 2" "AR 0 3"
  "CA 1 0" "CP 1 2" "CR 1 3"
  "PA 2 0" "PC 2 1" "PR 2 3"
  "RA 3 0" "RC 3 1" "RP 3 2"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Stage19-C coverage-protected adapter: seed=2022 task=${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_head_pair_feature_coverage.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD temporal_precision_head_seed2022_pair_feature_coverage \
    SETTING.SEED 2022 \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_pair_feature_coverage/*.txt' \
  --record-type standard \
  --selection peak \
  > output/uda/office-home/temporal_precision_head_pair_feature_coverage_seed2022_accuracy.csv

python tools/summarize_office_home_pair_feature_coverage.py \
  --csv output/uda/office-home/temporal_precision_head_pair_feature_coverage_seed2022_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_pair_feature_coverage_seed2022_summary.json

python tools/summarize_pair_feature_flow.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_pair_feature_coverage/*.txt' \
  --min-active-rank 8 \
  --allow-fallback \
  --out output/uda/office-home/temporal_precision_head_pair_feature_coverage_seed2022_flow.json

python tools/summarize_office_home_peak_gap.py \
  --csv output/uda/office-home/temporal_precision_head_pair_feature_coverage_seed2022_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_pair_feature_coverage_seed2022_peak_gap.json \
  --peak-is-primary \
  --next-on-no-headroom 'move to agreement-anchor class-conditional covariance subspace transport'
