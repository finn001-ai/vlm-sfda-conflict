#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Resolve the negative one-factor DCM/CLM ablations before any 12-task sweep.
# The same uniform DCM checkpoint is reused by the ARG-on and ARG-off runs.
experiment_seed="${1:-2020}"
tasks=(AC CP PR RA)
variants=(uniform_writable uniform_writable_arg_none)

for variant in "${variants[@]}"; do
  for task in "${tasks[@]}"; do
    run_dir="output/uda/office-home/${task}/dcr_ablation_${variant}_office_home_rankadaptive_seed${experiment_seed}"
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
  --tasks AC CP PR RA \
  --variants uniform_writable uniform_writable_arg_none

