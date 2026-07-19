#!/usr/bin/env bash
set -euo pipefail

# No retraining: re-read the existing seed-2022 logs and expose oracle peak headroom.
python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed2022_ema_probe/*.txt' \
  > output/uda/office-home/temporal_precision_head_ema_seed2022_final_peak.csv

python tools/summarize_office_home_peak_gap.py \
  --csv output/uda/office-home/temporal_precision_head_ema_seed2022_final_peak.csv \
  --out output/uda/office-home/temporal_precision_head_ema_seed2022_final_peak_summary.json
