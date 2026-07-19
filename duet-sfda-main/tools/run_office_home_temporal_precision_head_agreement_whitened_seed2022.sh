#!/usr/bin/env bash
set -euo pipefail

# Run only after the Stage21 AC/PA/RA preflight passes.
declare -a TASKS=(
  "AC 0 1" "AP 0 2" "AR 0 3"
  "CA 1 0" "CP 1 2" "CR 1 3"
  "PA 2 0" "PC 2 1" "PR 2 3"
  "RA 3 0" "RC 3 1" "RP 3 2"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Stage21 agreement-whitened transport: seed=2022 task=${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_head_agreement_whitened.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD temporal_precision_head_seed2022_agreement_whitened \
    SETTING.SEED 2022 \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_agreement_whitened/*.txt' \
  --record-type standard \
  --selection peak \
  > output/uda/office-home/temporal_precision_head_agreement_whitened_seed2022_accuracy.csv

python tools/summarize_whitened_transport_flow.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_agreement_whitened/*.txt' \
  --out output/uda/office-home/temporal_precision_head_agreement_whitened_seed2022_flow.json

python tools/summarize_office_home_whitened_transport.py \
  --csv output/uda/office-home/temporal_precision_head_agreement_whitened_seed2022_accuracy.csv \
  --flow output/uda/office-home/temporal_precision_head_agreement_whitened_seed2022_flow.json \
  --out output/uda/office-home/temporal_precision_head_agreement_whitened_seed2022_summary.json

python tools/summarize_office_home_peak_gap.py \
  --csv output/uda/office-home/temporal_precision_head_agreement_whitened_seed2022_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_agreement_whitened_seed2022_peak_gap.json \
  --peak-is-primary \
  --next-on-no-headroom 'move to three-view class-conditional noise EM consensus'
