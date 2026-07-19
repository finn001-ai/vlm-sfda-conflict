#!/usr/bin/env bash
set -euo pipefail

# Cheap mechanism check before the full Stage19 seed-2022 run.
python image_target_of_oh_vs.py \
  --cfg cfgs/office-home/temporal_precision_head_pair_feature.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD temporal_precision_head_seed2022_pair_feature_preflight \
  SETTING.SEED 2022 \
  SETTING.S 0 SETTING.T 1

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/AC/temporal_precision_head_seed2022_pair_feature_preflight/*.txt' \
  --record-type standard \
  --selection peak \
  > output/uda/office-home/temporal_precision_head_pair_feature_preflight_accuracy.csv

python tools/summarize_pair_feature_preflight.py \
  --csv output/uda/office-home/temporal_precision_head_pair_feature_preflight_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_pair_feature_preflight_summary.json

python tools/summarize_pair_feature_flow.py \
  --glob 'output/uda/office-home/AC/temporal_precision_head_seed2022_pair_feature_preflight/*.txt' \
  --out output/uda/office-home/temporal_precision_head_pair_feature_preflight_flow.json
