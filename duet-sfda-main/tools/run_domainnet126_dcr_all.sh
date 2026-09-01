#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run all 12 DomainNet-126 transfers. Completed tasks are skipped.
experiment_seed="${1:-2020}"
profile="${2:-delayed}"
tasks=(CP CR CS PC PR PS RC RP RS SC SP SR)

case "$profile" in
  delayed)
    method_name="dcr_domainnet126_rankadaptive_seed${experiment_seed}"
    equivalent_method_name="dcr_domainnet126_samplewise_seed${experiment_seed}"
    dcm_credit_mode="delayed"
    ;;
  uniform_locked_arg)
    method_name="dcr_domainnet126_uniform_locked_arg_rankadaptive_seed${experiment_seed}"
    equivalent_method_name=""
    dcm_credit_mode="uniform"
    ;;
  *)
    echo "Profile must be delayed or uniform_locked_arg" >&2
    exit 1
    ;;
esac

for task in "${tasks[@]}"; do
  run_dir="output/uda/domainnet126/${task}/${method_name}"
  logs=("$run_dir"/*.txt)
  if [ -f "${run_dir}/target_F.pt" ] \
    && [ "${#logs[@]}" -eq 1 ] \
    && [ "$(grep -c "Task: ${task}" "${logs[0]}")" -eq 16 ] \
    && grep -q "DCR refinement: enabled=True;.*soft_replacement_mode=task_supported;.*memory_write_mode=locked; credit_mode=${dcm_credit_mode};" "${logs[0]}"; then
    echo "==> [${task}] already complete; skipping"
    continue
  fi
  if [ -n "$equivalent_method_name" ]; then
    equivalent_run_dir="output/uda/domainnet126/${task}/${equivalent_method_name}"
    equivalent_logs=("$equivalent_run_dir"/*.txt)
    if [ -f "${equivalent_run_dir}/target_F.pt" ] \
      && [ "${#equivalent_logs[@]}" -eq 1 ] \
      && [ "$(grep -c "Task: ${task}" "${equivalent_logs[0]}")" -eq 16 ]; then
      echo "==> [${task}] equivalent samplewise branch already complete; skipping"
      continue
    fi
  fi
  bash tools/run_domainnet126_dcr.sh "$experiment_seed" "$task" "$profile"
done

echo "==> DomainNet-126 DCR-SFDA completed: seed=${experiment_seed}; profile=${profile}"
