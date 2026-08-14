#!/usr/bin/env bash
set -euo pipefail

# Run one method on all 12 directed DomainNet-126 transfer tasks.
#
# Optional environment variables:
#   GPU_ID=0
#   PYTHON_BIN=python
#   METHOD=plmatch
#   SEED=2020
#   CKPT_DIR=.
#   OUTPUT_SRC=source
#   OUTPUT_ROOT=output/domainnet126_<method>_runs
#   RUN_ID=20260814_120000
#   DATA_DIR=/path/to/data
#
# Additional arguments are appended as YACS config overrides, for example:
#   bash tools/run_domainnet126_plmatch_all.sh TEST.BATCH_SIZE 32

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${script_dir}/.." && pwd)
cd "$repo_dir"

gpu_id=${GPU_ID:-0}
python_bin=${PYTHON_BIN:-python}
method=${METHOD:-plmatch}
seed=${SEED:-2020}
ckpt_dir=${CKPT_DIR:-.}
output_src=${OUTPUT_SRC:-source}
cfg_file="cfgs/domainnet126/${method}.yaml"
output_root=${OUTPUT_ROOT:-output/domainnet126_${method}_runs}
run_id=${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}
run_dir="${output_root}/${run_id}"
master_log="${run_dir}/domainnet126_${method}_${run_id}.log"
summary_csv="${run_dir}/domainnet126_${method}_${run_id}_summary.csv"
console_dir="${run_dir}/console"

domain_names=(clipart painting real sketch)
domain_keys=(C P R S)

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
fi

for required_file in image_target_in_126.py "$cfg_file"; do
  if [ ! -f "$required_file" ]; then
    echo "Missing required file: ${repo_dir}/${required_file}" >&2
    exit 1
  fi
done

if [ -e "$run_dir" ]; then
  echo "Run directory already exists: ${run_dir}" >&2
  echo "Set a different RUN_ID or OUTPUT_ROOT." >&2
  exit 1
fi

mkdir -p "$console_dir"
printf 'task,source,target,best_acc,train_seconds,train_time,status,console_log,framework_log\n' \
  > "$summary_csv"

log() {
  printf '%s\n' "$*" | tee -a "$master_log"
}

format_duration() {
  local total_seconds=$1
  printf '%02d:%02d:%02d' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

extract_best_acc() {
  local task=$1
  local task_log=$2

  sed -nE \
    "/Task:[[:space:]]*${task},.*Accuracy[[:space:]]*=/s/.*Accuracy[[:space:]]*=[[:space:]]*([0-9]+([.][0-9]+)?)%.*/\1/p" \
    "$task_log" \
    | awk 'BEGIN { best = -1 } $1 + 0 > best { best = $1 + 0 } END { if (best >= 0) printf "%.2f", best }'
}

run_started=$(date +%s)
log "DomainNet-126 ${method} 12-task run"
log "run_id=${run_id}"
log "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
log "gpu_id=${gpu_id}"
log "method=${method}"
log "seed=${seed}"
log "python=${python_bin}"
log "ckpt_dir=${ckpt_dir}"
log "output_src=${output_src}"
log "run_dir=${run_dir}"
if [ -n "${DATA_DIR:-}" ]; then
  log "data_dir=${DATA_DIR}"
fi
completed_tasks=0

for s in 0 1 2 3; do
  for t in 0 1 2 3; do
    if [ "$s" -eq "$t" ]; then
      continue
    fi

    task="${domain_keys[$s]}${domain_keys[$t]}"
    source_domain=${domain_names[$s]}
    target_domain=${domain_names[$t]}
    task_console_log="${console_dir}/${task}.log"
    task_output_dir="${run_dir}/uda/domainnet126/${task}/${method}"
    task_started=$(date +%s)

    log ""
    log "==> [${task}] ${source_domain} -> ${target_domain}"
    log "task_started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"

    command=(
      "$python_bin" image_target_in_126.py
      --cfg "$cfg_file"
      SAVE_DIR "$run_dir"
      CKPT_DIR "$ckpt_dir"
      SETTING.OUTPUT_SRC "$output_src"
      SETTING.SEED "$seed"
      SETTING.S "$s"
      SETTING.T "$t"
    )
    if [ -n "${DATA_DIR:-}" ]; then
      command+=(DATA_DIR "$DATA_DIR")
    fi
    command+=("$@")

    if CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 \
      "${command[@]}" 2>&1 | tee -a "$master_log" "$task_console_log"; then
      task_status=0
    else
      task_status=$?
    fi

    task_finished=$(date +%s)
    task_seconds=$((task_finished - task_started))
    task_time=$(format_duration "$task_seconds")
    framework_log=$(find "$task_output_dir" -maxdepth 1 -type f -name "${method}_*.txt" -print 2>/dev/null | sort | tail -n 1 || true)

    if [ "$task_status" -ne 0 ]; then
      printf '%s,%s,%s,NA,%d,%s,failed,%s,%s\n' \
        "$task" "$source_domain" "$target_domain" "$task_seconds" "$task_time" \
        "$task_console_log" "$framework_log" >> "$summary_csv"
      log "<== [${task}] FAILED (exit=${task_status}, time=${task_time})"
      log "Partial summary: ${summary_csv}"
      exit "$task_status"
    fi

    best_acc=$(extract_best_acc "$task" "$task_console_log")
    if [ -z "$best_acc" ]; then
      printf '%s,%s,%s,NA,%d,%s,no_accuracy,%s,%s\n' \
        "$task" "$source_domain" "$target_domain" "$task_seconds" "$task_time" \
        "$task_console_log" "$framework_log" >> "$summary_csv"
      log "<== [${task}] FAILED: no task accuracy was found (time=${task_time})"
      log "Partial summary: ${summary_csv}"
      exit 2
    fi

    printf '%s,%s,%s,%s,%d,%s,completed,%s,%s\n' \
      "$task" "$source_domain" "$target_domain" "$best_acc" "$task_seconds" "$task_time" \
      "$task_console_log" "$framework_log" >> "$summary_csv"
    completed_tasks=$((completed_tasks + 1))
    log "<== [${task}] best_acc=${best_acc}%, train_time=${task_time} (${task_seconds}s)"
  done
done

run_finished=$(date +%s)
total_seconds=$((run_finished - run_started))
total_time=$(format_duration "$total_seconds")
avg_acc=$(awk -F',' \
  'NR > 1 && $7 == "completed" { sum += $4; count += 1 } END { if (count > 0) printf "%.2f", sum / count }' \
  "$summary_csv")

log ""
log "========== FINAL SUMMARY =========="
while IFS=',' read -r task source target best_acc train_seconds train_time status console_log framework_log; do
  if [ "$task" = "task" ]; then
    continue
  fi
  log "${task}: best_acc=${best_acc}%, train_time=${train_time}, status=${status}"
done < "$summary_csv"
log "avg_acc=${avg_acc}% (${completed_tasks}/12 tasks)"
log "total_time=${total_time} (${total_seconds}s)"
log "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
log "master_log=${master_log}"
log "summary_csv=${summary_csv}"

if [ "$completed_tasks" -ne 12 ]; then
  log "ERROR: expected 12 completed tasks, got ${completed_tasks}"
  exit 3
fi
