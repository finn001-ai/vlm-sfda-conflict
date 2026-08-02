#!/usr/bin/env bash
set -euo pipefail

parameter_dir="output/uda/VISDA-C/TV/plmatch_pcgrad_parameter_audit_seed2020/conflict_pcgrad_parameter_audit"
output_dir="${parameter_dir}/pcgrad_compatibility_audit"

for path in \
  "${parameter_dir}/visda_conflict_pcgrad_parameter_summary.json" \
  "${parameter_dir}/visda_conflict_pcgrad_parameter_signal_lock.json" \
  "${parameter_dir}/batch_signal_locks" \
  "${parameter_dir}/visda_conflict_pcgrad_parameter_oracle_diagnostic.csv" \
  "${parameter_dir}/visda_conflict_pcgrad_parameter_groupwise_oracle_diagnostic.csv"; do
  if [ ! -e "$path" ]; then
    echo "Missing compatibility-audit input: $path" >&2
    echo "Run first: bash tools/run_visda_conflict_pcgrad_parameter_audit.sh" >&2
    exit 1
  fi
done

echo "==> CPU-only full-gradient compatibility audit for rejected raw PCGrad"
echo "==> Derives the fraction from ten prior label-free batch locks"
echo "==> Writes the new signal lock before reading oracle diagnostic CSVs"
echo "==> No image/model/checkpoint load, forward, backward, optimizer, or training"
CUDA_VISIBLE_DEVICES="" python tools/audit_visda_pcgrad_compatibility.py \
  --parameter-dir "$parameter_dir" \
  --output-dir "$output_dir"

echo "==> Audit complete: ${output_dir}/visda_conflict_pcgrad_compatibility_summary.json"
echo "==> PASS authorizes one matched proxy25 confirmation only, never full training"
