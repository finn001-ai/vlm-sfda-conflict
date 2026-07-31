#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Pure released-DUET/PLMatch baseline on all 12 Office-Home transfer tasks.

seed=2020
method="plmatch_office_home_full_seed${seed}"
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
      echo "Train sources first: bash tools/train_office_home_sources.sh" >&2
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
      output/uda/office-home/??/plmatch_office_home_full_seed2020) ;;
      *)
        echo "Refusing to clear unexpected Office-Home path: $task_dir" >&2
        exit 1
        ;;
    esac
    rm -rf -- "$task_dir"

    echo "==> Pure DUET Office-Home: ${task}, seed=${seed}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/plmatch.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$method" \
      SETTING.SEED "$seed" SETTING.S "$s" SETTING.T "$t" \
      ACTIVE.CYCLE 4

    logs=("$task_dir"/*.txt)
    if [ "${#logs[@]}" -ne 1 ]; then
      echo "Expected one ${task} log, found ${#logs[@]}" >&2
      exit 1
    fi
    if ! grep -q "DUET first-cycle prior: enabled=False; power=0.000" "${logs[0]}"; then
      echo "${task} did not use the pure DUET control path" >&2
      exit 1
    fi
    if [ "$(grep -c "Task: ${task}" "${logs[0]}")" -ne 16 ]; then
      echo "${task} did not finish the 4-cycle contract" >&2
      exit 1
    fi
  done
done

mkdir -p "$result_dir"
python tools/extract_final_accuracy.py \
  --glob "output/uda/office-home/*/${method}/*.txt" \
  --selection final \
  > "$result_dir/office_home_pure_duet_seed2020.csv"

visda_glob="output/uda/VISDA-C/TV/duet_first_cycle_prior_visda_full_seed2020/*.txt"
if compgen -G "$visda_glob" > /dev/null; then
  python tools/build_duet_benchmark_tables.py
  echo "==> Unified table: $result_dir/duet_fcp_visda8_office_home_duet.md"
else
  echo "==> Office-Home table: $result_dir/office_home_pure_duet_seed2020.csv"
  echo "Run VisDA full8 later, then: python tools/build_duet_benchmark_tables.py"
fi
