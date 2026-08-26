#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run all 12 Office-Home transfers. Completed tasks are skipped.
experiment_seed="${1:-2020}"
tasks=(AC AP AR CA CP CR PA PC PR RA RC RP)
method_name="dcr_office_home_full_seed${experiment_seed}"

for task in "${tasks[@]}"; do
  run_dir="output/uda/office-home/${task}/${method_name}"
  logs=("$run_dir"/*.txt)
  if [ -f "${run_dir}/target_F.pt" ] \
    && [ "${#logs[@]}" -eq 1 ] \
    && [ "$(grep -c "Task: ${task}" "${logs[0]}")" -eq 16 ]; then
    echo "==> [${task}] already complete; skipping"
    continue
  fi
  bash tools/run_office_home_dcr.sh "$experiment_seed" "$task"
done

echo "==> Office-Home DCR completed for seed ${experiment_seed}"
