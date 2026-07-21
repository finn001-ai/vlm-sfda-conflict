#!/usr/bin/env bash
set -euo pipefail

baseline_method="temporal_precision_head_seed2020_visda"
diagnostic_glob="output/uda/VISDA-C/TV/${baseline_method}/temporal_diagnostics/*_cycle*.npz"
result_dir="output/uda/VISDA-C"
oracle="$result_dir/temporal_precision_head_visda_seed2020_classwise_conflicts.json"

if ! compgen -G "$diagnostic_glob" > /dev/null; then
  echo "Missing completed Stage14 temporal diagnostics: $diagnostic_glob" >&2
  exit 1
fi
if [ ! -f "$oracle" ]; then
  echo "Missing classwise diagnostic: $oracle" >&2
  echo "Run: bash tools/run_visda_stage14_classwise_conflict_probe.sh" >&2
  exit 1
fi

echo "==> Testing label-free class reliability proxies (no training)"
python tools/analyze_visda_unlabeled_route_proxies.py \
  --glob "$diagnostic_glob" \
  --classwise-oracle "$oracle" \
  --out "$result_dir/temporal_precision_head_visda_seed2020_route_proxies.json" \
  --csv-out "$result_dir/temporal_precision_head_visda_seed2020_route_proxies.csv"
