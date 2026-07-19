#!/usr/bin/env bash
set -euo pipefail

# Run only after the complete Stage21 seed-2022 gate passes.
declare -a SEEDS=(2020 2021 2022)
declare -a TASKS=(
  "AC 0 1" "AP 0 2" "AR 0 3"
  "CA 1 0" "CP 1 2" "CR 1 3"
  "PA 2 0" "PC 2 1" "PR 2 3"
  "RA 3 0" "RC 3 1" "RP 3 2"
)

for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    read -r task_name s t <<< "$task"
    method="temporal_precision_head_seed${seed}_agreement_whitened"
    echo "==> Stage21 agreement-whitened: seed=${seed} task=${task_name}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/temporal_precision_head_agreement_whitened.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$method" \
      SETTING.SEED "$seed" \
      SETTING.S "$s" SETTING.T "$t"
  done
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed[0-9][0-9][0-9][0-9]_agreement_whitened/*.txt' \
  --record-type standard \
  --selection peak \
  > output/uda/office-home/temporal_precision_head_agreement_whitened_seed_sweep_accuracy.csv

python tools/summarize_office_home_seed_sweep.py \
  --csv output/uda/office-home/temporal_precision_head_agreement_whitened_seed_sweep_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_agreement_whitened_seed_sweep_summary.json \
  --max-seed-std 0.10 \
  --min-overall-mean 84.7825

python tools/summarize_whitened_transport_flow.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed[0-9][0-9][0-9][0-9]_agreement_whitened/*.txt' \
  --out output/uda/office-home/temporal_precision_head_agreement_whitened_seed_sweep_flow.json
