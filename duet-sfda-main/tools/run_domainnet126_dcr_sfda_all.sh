#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run all 12 DomainNet-126 transfers. Completed tasks are skipped.
experiment_seed="${1:-2020}"
tasks=(CP CR CS PC PR PS RC RP RS SC SP SR)
method_name="plmatch_dac_handoff_dcr_sfda_domainnet126_seed${experiment_seed}"

for task in "${tasks[@]}"; do
  run_dir="output/uda/domainnet126/${task}/${method_name}"
  logs=("$run_dir"/*.txt)
  if [ -f "${run_dir}/target_F.pt" ] \
    && [ "${#logs[@]}" -eq 1 ] \
    && [ "$(grep -c "Task: ${task}" "${logs[0]}")" -eq 16 ]; then
    echo "==> [${task}] already complete; skipping"
    continue
  fi
  bash tools/run_domainnet126_dcr_sfda.sh "$experiment_seed" "$task"
done

echo "==> DomainNet-126 DCR-SFDA completed for seed ${experiment_seed}"
