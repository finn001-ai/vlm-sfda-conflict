#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Four-task ring: every Office-Home domain appears once as source and target.
# This is the screening set; only validated variants should be expanded to 12 tasks.
experiment_seed="${1:-2020}"
tasks=(AC CP PR RA)
variants=(dcm_uniform clm_writable arg_none)

for variant in "${variants[@]}"; do
  for task in "${tasks[@]}"; do
    run_dir="output/uda/office-home/${task}/plmatch_dac_handoff_dcr_sfda_ablation_${variant}_office_home_seed${experiment_seed}"
    logs=("$run_dir"/*.txt)
    if [ -f "${run_dir}/target_F.pt" ] \
      && [ "${#logs[@]}" -eq 1 ] \
      && [ "$(grep -c "Task: ${task}" "${logs[0]}")" -eq 16 ]; then
      echo "==> [${task}/${variant}] already complete; skipping"
      continue
    fi
    bash tools/run_office_home_dcr_sfda_ablation.sh \
      "$experiment_seed" "$task" "$variant"
  done
done

echo "==> Core DCR-SFDA ablations completed for seed ${experiment_seed}"
echo "==> Full-method reference uses the existing official DCR-SFDA runs on AC/CP/PR/RA"
echo "==> Summarize with: python tools/summarize_office_home_dcr_sfda_ablation.py --seed ${experiment_seed}"
