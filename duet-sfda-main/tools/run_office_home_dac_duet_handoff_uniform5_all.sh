#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Usage:
#   all 12 tasks: bash tools/run_office_home_dac_duet_handoff_uniform5_all.sh 2020
#   one task:     bash tools/run_office_home_dac_duet_handoff_uniform5_all.sh 2020 AC
#
# Each task uses the same schedule as VisDA:
#   DAC 15 epochs -> complete F/B/C handoff -> cyclic refinement 5 cycles x 4 epochs.
experiment_seed="${1:-2020}"
task_filter="${2:-all}"
domain_keys=(A C P R)
dac_method="duet_delayed_agreement_credit_office_home_full_seed${experiment_seed}"
duet_method="plmatch_dac_handoff_uniform5_office_home_full_seed${experiment_seed}"
result_dir="output/uda/benchmark_tables"

if [ "$task_filter" != "all" ]; then
  case "$task_filter" in
    AC|AP|AR|CA|CP|CR|PA|PC|PR|RA|RC|RP) ;;
    *)
      echo "Task must be all or one of: AC AP AR CA CP CR PA PC PR RA RC RP" >&2
      exit 1
      ;;
  esac
fi

for required_path in data/office-home/classname.txt; do
  if [ ! -f "$required_path" ]; then
    echo "Missing Office-Home input: $required_path" >&2
    exit 1
  fi
done
for key in "${domain_keys[@]}"; do
  for part in F B C; do
    required_path="source/uda/office-home/${key}/source_${part}.pt"
    if [ ! -f "$required_path" ]; then
      echo "Missing Office-Home source weight: $required_path" >&2
      echo "Train sources first: bash tools/train_office_home_sources.sh" >&2
      exit 1
    fi
  done
done

completed_tasks=0
for s in 0 1 2 3; do
  for t in 0 1 2 3; do
    if [ "$s" -eq "$t" ]; then
      continue
    fi
    task="${domain_keys[$s]}${domain_keys[$t]}"
    if [ "$task_filter" != "all" ] && [ "$task" != "$task_filter" ]; then
      continue
    fi

    target_domain=(Art Clipart Product RealWorld)
    target_list="data/office-home/${target_domain[$t]}_list.txt"
    if [ ! -f "$target_list" ]; then
      echo "Missing Office-Home target list: $target_list" >&2
      exit 1
    fi
    target_samples=$(wc -l < "$target_list" | tr -d ' ')

    dac_run_dir="output/uda/office-home/${task}/${dac_method}"
    dac_state="${dac_run_dir}/delayed_credit_state.pt"
    dac_complete=true
    for checkpoint in target_F.pt target_B.pt target_C.pt delayed_credit_state.pt; do
      if [ ! -f "${dac_run_dir}/${checkpoint}" ]; then
        dac_complete=false
      fi
    done

    if [ "$dac_complete" = false ]; then
      if [ -d "$dac_run_dir" ] && compgen -G "$dac_run_dir/*.txt" > /dev/null; then
        echo "Partial DAC directory exists for ${task}: ${dac_run_dir}" >&2
        echo "Move it aside before rerunning this task" >&2
        exit 1
      fi
      echo "==> [${task}] Stage 1/2: full-data DAC, 15 epochs"
      python image_target_of_oh_vs.py \
        --cfg cfgs/office-home/duet_delayed_agreement_credit.yaml \
        CKPT_DIR . SETTING.OUTPUT_SRC source \
        MODEL.METHOD "$dac_method" \
        SETTING.S "$s" SETTING.T "$t" SETTING.SEED "$experiment_seed" \
        ACTIVE.ADAPTATION_LIST ""
    else
      echo "==> [${task}] Reusing completed 15-epoch DAC checkpoint"
    fi

    python - "$dac_state" "$target_samples" "$task" <<'PY'
import sys

import torch

state_path = sys.argv[1]
expected_samples = int(sys.argv[2])
task = sys.argv[3]
state = torch.load(state_path, map_location="cpu", weights_only=True)
memory = state.get("memory")
if not isinstance(memory, torch.Tensor) or memory.ndim != 2:
    raise SystemExit(f"{task}: DAC state has no valid memory")
if int(memory.shape[0]) != expected_samples:
    raise SystemExit(
        f"{task}: DAC memory_rows={int(memory.shape[0])}, "
        f"expected={expected_samples}"
    )
if not torch.isfinite(memory).all():
    raise SystemExit(f"{task}: DAC memory contains non-finite values")
print(
    f"==> [{task}] Verified DAC memory: "
    f"samples={memory.shape[0]}; classes={memory.shape[1]}"
)
PY

    handoff_source="output/dac_duet_handoff_uniform5_office_home_seed${experiment_seed}_${task}"
    handoff_source_dir="${handoff_source}/uda/office-home/${domain_keys[$s]}"
    mkdir -p "$handoff_source_dir"
    cp -f "${dac_run_dir}/target_F.pt" "${handoff_source_dir}/source_F.pt"
    cp -f "${dac_run_dir}/target_B.pt" "${handoff_source_dir}/source_B.pt"
    cp -f "${dac_run_dir}/target_C.pt" "${handoff_source_dir}/source_C.pt"

    duet_run_dir="output/uda/office-home/${task}/${duet_method}"
    completed_log=""
    duet_logs=("$duet_run_dir"/*.txt)
    for candidate_log in "${duet_logs[@]}"; do
      if grep -q "handoff_target_passes=20; final_checkpoint_fixed=True" "$candidate_log" \
        && grep -q "Cycle: 5/5" "$candidate_log"; then
        completed_log="$candidate_log"
      fi
    done
    if [ -n "$completed_log" ]; then
      echo "==> [${task}] Reusing completed uniform-5 DUET handoff"
      completed_tasks=$((completed_tasks + 1))
      continue
    fi
    if [ -d "$duet_run_dir" ] && compgen -G "$duet_run_dir/*.txt" > /dev/null; then
      echo "Partial DUET handoff directory exists for ${task}: ${duet_run_dir}" >&2
      echo "Move it aside before rerunning this task" >&2
      exit 1
    fi

    echo "==> [${task}] Stage 2/2: cyclic refinement 5 cycles x 4 epochs"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/plmatch.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC "$handoff_source" \
      MODEL.METHOD "$duet_method" \
      SETTING.S "$s" SETTING.T "$t" SETTING.SEED "$experiment_seed" \
      TEST.MAX_EPOCH 4 TEST.INTERVAL 4 \
      ACTIVE.CYCLE 5 ACTIVE.ADAPTATION_LIST "" \
      DUET_HANDOFF.FINAL_EXTRA_EPOCHS 0

    duet_logs=("$duet_run_dir"/*.txt)
    if [ "${#duet_logs[@]}" -ne 1 ]; then
      echo "Expected one ${task} handoff log, found ${#duet_logs[@]}" >&2
      exit 1
    fi
    if [ "$(grep -c "Task: ${task}" "${duet_logs[0]}")" -ne 20 ]; then
      echo "${task} did not complete 5 cycles x 4 logged epochs" >&2
      exit 1
    fi
    if ! grep -q "Cycle: 5/5" "${duet_logs[0]}"; then
      echo "${task} did not reach Cycle 5/5" >&2
      exit 1
    fi
    if ! grep -q "handoff_target_passes=20; final_checkpoint_fixed=True" "${duet_logs[0]}"; then
      echo "${task} did not save the uniform-5 fixed final checkpoint" >&2
      exit 1
    fi
    completed_tasks=$((completed_tasks + 1))
  done
done

if [ "$completed_tasks" -eq 0 ]; then
  echo "No Office-Home tasks matched: $task_filter" >&2
  exit 1
fi

mkdir -p "$result_dir"
summary_path="${result_dir}/office_home_dac_duet_uniform5_seed${experiment_seed}.csv"
python tools/extract_final_accuracy.py \
  --glob "output/uda/office-home/*/${duet_method}/*.txt" \
  --selection final \
  > "$summary_path"

echo "==> Completed Office-Home tasks: ${completed_tasks}"
echo "==> Final-only summary: ${summary_path}"
