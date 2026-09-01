#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run all 12 Office-Home transfers. Completed tasks are skipped.
experiment_seed="${1:-2020}"
profile="${2:-delayed}"
tasks=(AC AP AR CA CP CR PA PC PR RA RC RP)

case "$profile" in
  delayed)
    method_name="dcr_office_home_rankadaptive_seed${experiment_seed}"
    dcm_credit_mode="delayed"
    ;;
  uniform_locked_arg)
    method_name="dcr_office_home_uniform_locked_arg_rankadaptive_seed${experiment_seed}"
    dcm_credit_mode="uniform"
    ;;
  *)
    echo "Profile must be delayed or uniform_locked_arg" >&2
    exit 1
    ;;
esac

for task in "${tasks[@]}"; do
  run_dir="output/uda/office-home/${task}/${method_name}"
  logs=("$run_dir"/*.txt)
  if [ -f "${run_dir}/target_F.pt" ] \
    && [ "${#logs[@]}" -eq 1 ] \
    && [ "$(grep -c "Task: ${task}" "${logs[0]}")" -eq 16 ] \
    && grep -q "DCR refinement: enabled=True;.*soft_replacement_mode=task_supported;.*memory_write_mode=locked; credit_mode=${dcm_credit_mode};" "${logs[0]}"; then
    echo "==> [${task}] already complete; skipping"
    continue
  fi
  bash tools/run_office_home_dcr.sh "$experiment_seed" "$task" "$profile"
done

echo "==> Office-Home DCR completed: seed=${experiment_seed}; profile=${profile}"
