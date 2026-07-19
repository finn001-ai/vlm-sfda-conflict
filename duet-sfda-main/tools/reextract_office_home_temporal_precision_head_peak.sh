#!/usr/bin/env bash
set -euo pipefail

# No retraining. Re-evaluate the original Stage14 seed logs with peak selection.
python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed[0-9][0-9][0-9][0-9]/*.txt' \
  --selection peak \
  > output/uda/office-home/temporal_precision_head_stage14_peak_accuracy.csv

python tools/summarize_office_home_seed_sweep.py \
  --csv output/uda/office-home/temporal_precision_head_stage14_peak_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_stage14_peak_summary.json \
  --max-seed-std 0.10

python tools/summarize_office_home_peak_gap.py \
  --csv output/uda/office-home/temporal_precision_head_stage14_peak_accuracy.csv \
  --out output/uda/office-home/temporal_precision_head_stage14_final_peak_gap.json \
  --peak-is-primary \
  --next-on-no-headroom 'move to dataset-level stable class-pair conflict-flow adaptation'
