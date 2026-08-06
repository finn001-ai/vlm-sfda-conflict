#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Ablation #2 / baseline: pure duet_first_cycle_prior with POWER=0.8.
# Runs through the same candidate entry with DUET_CONTEXT.ENABLED=False, which
# must fully degenerate to the original DUET-FCP training path.

seed=2020
method="duet_first_cycle_prior_context_transformer_office_home_full_seed${seed}"
result_dir="output/uda/benchmark_tables"
domain_keys=(A C P R)

for path in data/office-home/classname.txt; do
  if [ ! -f "$path" ]; then
    echo "Missing Office-Home input: $path" >&2
    exit 1
  fi
done
for key in "${domain_keys[@]}"; do
  for part in F B C; do
    path="source/uda/office-home/${key}/source_${part}.pt"
    if [ ! -f "$path" ]; then
      echo "Missing Office-Home source weight: $path" >&2
      exit 1
    fi
  done
done

for s in 0 1 2 3; do
  for t in 0 1 2 3; do
    if [ "$s" -eq "$t" ]; then
      continue
    fi
    task="${domain_keys[$s]}${domain_keys[$t]}"
    task_dir="output/uda/office-home/${task}/${method}"
    case "$task_dir" in
      output/uda/office-home/??/duet_first_cycle_prior_context_transformer_office_home_full_seed2020) ;;
      *)
        echo "Refusing to clear unexpected Office-Home path: $task_dir" >&2
        exit 1
        ;;
    esac
    rm -rf -- "$task_dir"

    echo "==> DUET-FCP POWER=0.8 baseline (context disabled) Office-Home: ${task}, seed=${seed}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/duet_first_cycle_prior_context_transformer.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$method" \
      SETTING.SEED "$seed" SETTING.S "$s" SETTING.T "$t" \
      ACTIVE.CYCLE 4 \
      DUET_FCP.POWER 0.8 \
      DUET_CONTEXT.ENABLED False

    logs=("$task_dir"/*.txt)
    if [ "${#logs[@]}" -ne 1 ]; then
      echo "Expected one ${task} log, found ${#logs[@]}" >&2
      exit 1
    fi
    if ! grep -q "DUET context transformer: requested=True; enabled=False" "${logs[0]}"; then
      echo "${task} did not degenerate to pure DUET-FCP (context disabled)" >&2
      exit 1
    fi
    if [ "$(grep -c "DUET first-cycle prior schedule: cycle=1; active=True" "${logs[0]}")" -ne 1 ]; then
      echo "${task} first-cycle prior activation contract failed" >&2
      exit 1
    fi
  done
done

echo "==> Office-Home DUET-FCP POWER=0.8 baseline finished: output/uda/office-home/*/${method}"
