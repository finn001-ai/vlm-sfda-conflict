#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Pure PLMatch baseline on all 12 Office-Home transfers.
experiment_seed="${1:-2020}"
domain_keys=(A C P R)
method_name="plmatch_office_home_seed${experiment_seed}"

for key in "${domain_keys[@]}"; do
  for part in F B C; do
    path="source/uda/office-home/${key}/source_${part}.pt"
    if [ ! -f "$path" ]; then
      echo "Missing source weight: $path" >&2
      echo "Train sources first: bash tools/train_office_home_sources.sh" >&2
      exit 1
    fi
  done
done

for source_index in 0 1 2 3; do
  for target_index in 0 1 2 3; do
    if [ "$source_index" -eq "$target_index" ]; then
      continue
    fi
    task="${domain_keys[$source_index]}${domain_keys[$target_index]}"
    run_dir="output/uda/office-home/${task}/${method_name}"
    logs=("$run_dir"/*.txt)
    if [ -f "${run_dir}/target_F.pt" ] \
      && [ "${#logs[@]}" -eq 1 ] \
      && [ "$(grep -c "Task: ${task}" "${logs[0]}")" -eq 16 ]; then
      echo "==> [${task}] already complete; skipping"
      continue
    fi

    echo "==> [${task}] Pure PLMatch, seed=${experiment_seed}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/plmatch.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$method_name" \
      SETTING.SEED "$experiment_seed" \
      SETTING.S "$source_index" SETTING.T "$target_index" \
      ACTIVE.CYCLE 4 ACTIVE.ADAPTATION_LIST ""
  done
done

echo "==> Office-Home PLMatch completed for seed ${experiment_seed}"
