#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Full candidate: duet_first_cycle_prior_context_transformer on Office-Home
# (Task/CLIP-consistent class-balanced anchors + Context Transformer).

seed=2020
method="duet_first_cycle_prior_context_transformer_office_home_full_seed${seed}"
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

    echo "==> DUET-FCP + Context Transformer Office-Home: ${task}, seed=${seed}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/duet_first_cycle_prior_context_transformer.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$method" \
      SETTING.SEED "$seed" SETTING.S "$s" SETTING.T "$t" \
      ACTIVE.CYCLE 4

    logs=("$task_dir"/*.txt)
    if [ "${#logs[@]}" -ne 1 ]; then
      echo "Expected one ${task} log, found ${#logs[@]}" >&2
      exit 1
    fi
    if ! grep -q "DUET context transformer: requested=True; enabled=True" "${logs[0]}"; then
      echo "${task} did not enable the Context Transformer" >&2
      exit 1
    fi
    if [ "$(grep -c "DUET context refinement: cycle=1" "${logs[0]}")" -ne 1 ]; then
      echo "${task} did not run the cycle-1 context refinement" >&2
      exit 1
    fi
    if [ "$(grep -c "Task: " "${logs[0]}")" -ne 16 ]; then
      echo "${task} did not finish 4 cycles / 16 checkpoints" >&2
      exit 1
    fi
  done
done

echo "==> Office-Home DUET-FCP + Context Transformer finished: output/uda/office-home/*/${method}"
