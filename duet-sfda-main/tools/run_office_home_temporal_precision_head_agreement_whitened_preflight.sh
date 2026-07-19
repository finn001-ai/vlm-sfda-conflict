#!/usr/bin/env bash
set -euo pipefail

# Stage21 mechanism/performance preflight on one Clipart and two target-Art tasks.
declare -a TASKS=("AC 0 1" "PA 2 0" "RA 3 0")

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Stage21 agreement-whitened preflight: seed=2022 task=${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_head_agreement_whitened.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD temporal_precision_head_seed2022_agreement_whitened_preflight \
    SETTING.SEED 2022 \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_agreement_whitened_preflight/*.txt' \
  --record-type standard \
  --selection peak \
  > output/uda/office-home/temporal_precision_head_agreement_whitened_preflight_accuracy.csv

python tools/summarize_whitened_transport_flow.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_agreement_whitened_preflight/*.txt' \
  --out output/uda/office-home/temporal_precision_head_agreement_whitened_preflight_flow.json

python tools/summarize_whitened_transport_preflight.py \
  --csv output/uda/office-home/temporal_precision_head_agreement_whitened_preflight_accuracy.csv \
  --flow output/uda/office-home/temporal_precision_head_agreement_whitened_preflight_flow.json \
  --out output/uda/office-home/temporal_precision_head_agreement_whitened_preflight_summary.json
