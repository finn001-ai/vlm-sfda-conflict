#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Final paper table: three core module deletions on all 12 Office-Home tasks.
# Completed runs are detected and skipped, so this command can resume safely.
experiment_seed="${1:-2020}"
tasks=(AC AP AR CA CP CR PA PC PR RA RC RP)
variants=(dcm_uniform clm_writable arg_none)

for variant in "${variants[@]}"; do
  for task in "${tasks[@]}"; do
    run_dir="output/uda/office-home/${task}/dcr_ablation_${variant}_office_home_samplewise_seed${experiment_seed}"
    logs=("$run_dir"/*.txt)
    if [ -f "${run_dir}/target_F.pt" ] \
      && [ "${#logs[@]}" -eq 1 ] \
      && [ "$(grep -c "Task: ${task}" "${logs[0]}")" -eq 16 ]; then
      echo "==> [${task}/${variant}] already complete; skipping"
      continue
    fi
    bash tools/run_office_home_dcr_ablation.sh \
      "$experiment_seed" "$task" "$variant"
  done
done

python tools/summarize_office_home_dcr_ablation.py \
  --seed "$experiment_seed" \
  --tasks AC AP AR CA CP CR PA PC PR RA RC RP
