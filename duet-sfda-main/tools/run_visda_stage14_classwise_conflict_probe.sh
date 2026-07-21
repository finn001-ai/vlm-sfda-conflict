#!/usr/bin/env bash
set -euo pipefail

baseline_method="temporal_precision_head_seed2020_visda"
diagnostic_glob="output/uda/VISDA-C/TV/${baseline_method}/temporal_diagnostics/*_cycle*.npz"
result_dir="output/uda/VISDA-C"

if ! compgen -G "$diagnostic_glob" > /dev/null; then
  echo "Missing completed Stage14 temporal diagnostics: $diagnostic_glob" >&2
  exit 1
fi
if [ ! -f data/VISDA-C/classname.txt ]; then
  echo "Missing VisDA-C class names: data/VISDA-C/classname.txt" >&2
  exit 1
fi

echo "==> Analyzing existing VisDA-C Stage14 conflicts by class (no training)"
python tools/analyze_visda_classwise_temporal_conflicts.py \
  --glob "$diagnostic_glob" \
  --class-names data/VISDA-C/classname.txt \
  --out "$result_dir/temporal_precision_head_visda_seed2020_classwise_conflicts.json" \
  --csv-out "$result_dir/temporal_precision_head_visda_seed2020_classwise_conflicts.csv"
