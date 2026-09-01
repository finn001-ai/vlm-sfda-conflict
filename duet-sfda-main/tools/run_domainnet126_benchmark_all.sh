#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run nine SFDA baselines on all 12 directed DomainNet-126 transfer tasks.
# Each task reports target-domain accuracy; the method summary averages the 12
# task accuracies with equal weight, matching the standard DomainNet-126 table.
#
# Default:
#   bash tools/run_domainnet126_benchmark_all.sh
#
# Resume an interrupted run:
#   RUN_ID=20260902_120000 RESUME=1 \
#     bash tools/run_domainnet126_benchmark_all.sh
#
# Run a subset:
#   METHODS="shot nrc difo plmatch" \
#     bash tools/run_domainnet126_benchmark_all.sh
#
# Use fixed-final task accuracies in the primary avg_acc column:
#   REPORT_METRIC=final bash tools/run_domainnet126_benchmark_all.sh
#
# Optional environment variables:
#   PYTHON_BIN=python
#   SEED=2020
#   METHODS="shot nrc gkd adacontrast cowa sclm tpds difo plmatch"
#   OUTPUT_ROOT=output/domainnet126_benchmark_runs
#   RUN_ID=<current timestamp>
#   RESUME=0
#   CONTINUE_ON_ERROR=1
#   REPORT_METRIC=peak  # peak or final; both are always saved
#   CHECKPOINT_ROOT=<repository root containing source/uda/domainnet126>
#   FOLDER_ROOT=<repository>/data/
#   DATA_DIR=/path/to/image-data  # optional override
#
# Extra command-line arguments are appended as common YACS overrides to every
# selected method. Only pass keys shared by all selected configurations.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${script_dir}/.." && pwd)
cd "$repo_dir"

python_bin=${PYTHON_BIN:-python}
seed=${SEED:-2020}
methods_text=${METHODS:-"shot nrc gkd adacontrast cowa sclm tpds difo plmatch"}
output_root=${OUTPUT_ROOT:-output/domainnet126_benchmark_runs}
run_id=${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}
resume=${RESUME:-0}
continue_on_error=${CONTINUE_ON_ERROR:-1}
report_metric=${REPORT_METRIC:-peak}
checkpoint_root=${CHECKPOINT_ROOT:-$repo_dir}
folder_root=${FOLDER_ROOT:-${repo_dir}/data/}

case "$output_root" in
  /*) ;;
  *) output_root="${repo_dir}/${output_root}" ;;
esac
case "$checkpoint_root" in
  /*) ;;
  *) checkpoint_root="${repo_dir}/${checkpoint_root}" ;;
esac
case "$folder_root" in
  /*) ;;
  *) folder_root="${repo_dir}/${folder_root}" ;;
esac
case "$folder_root" in
  */) ;;
  *) folder_root="${folder_root}/" ;;
esac

for value in "$resume" "$continue_on_error"; do
  if [ "$value" != "0" ] && [ "$value" != "1" ]; then
    echo "RESUME and CONTINUE_ON_ERROR must be 0 or 1." >&2
    exit 1
  fi
done
if [ "$report_metric" != "peak" ] && [ "$report_metric" != "final" ]; then
  echo "REPORT_METRIC must be peak or final." >&2
  exit 1
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
fi

read -r -a raw_methods <<< "$methods_text"
methods=()
for raw_method in "${raw_methods[@]}"; do
  case "$raw_method" in
    shot|SHOT) method=shot ;;
    nrc|NRC) method=nrc ;;
    gkd|GKD) method=gkd ;;
    adacon|AdaCon|ADACON|adacontrast|AdaContrast) method=adacontrast ;;
    cowa|CoWA|COWA) method=cowa ;;
    sclm|SCLM) method=sclm ;;
    tpds|TPDS) method=tpds ;;
    difo|DIFO) method=difo ;;
    plmatch|PLMatch|PLMATCH) method=plmatch ;;
    *)
      echo "Unsupported method: ${raw_method}" >&2
      exit 1
      ;;
  esac
  for existing_method in "${methods[@]}"; do
    if [ "$existing_method" = "$method" ]; then
      echo "Duplicate method after alias normalization: ${raw_method}" >&2
      exit 1
    fi
  done
  methods+=("$method")
done
if [ "${#methods[@]}" -eq 0 ]; then
  echo "METHODS is empty." >&2
  exit 1
fi

domain_names=(clipart painting real sketch)
domain_keys=(C P R S)
tasks=(CP CR CS PC PR PS RC RP RS SC SP SR)

for method in "${methods[@]}"; do
  if [ ! -f "cfgs/domainnet126/${method}.yaml" ]; then
    echo "Missing method config: cfgs/domainnet126/${method}.yaml" >&2
    exit 1
  fi
done
for required_path in \
  image_target_in_126.py \
  data/domainnet126/classname.txt \
  "${folder_root}domainnet126/clipart_list.txt" \
  "${folder_root}domainnet126/painting_list.txt" \
  "${folder_root}domainnet126/real_list.txt" \
  "${folder_root}domainnet126/sketch_list.txt" \
  "${checkpoint_root}/source/uda/domainnet126/C/best_clipart_2020.pth" \
  "${checkpoint_root}/source/uda/domainnet126/P/best_painting_2020.pth" \
  "${checkpoint_root}/source/uda/domainnet126/R/best_real_2020.pth" \
  "${checkpoint_root}/source/uda/domainnet126/S/best_sketch_2020.pth"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing DomainNet-126 input: ${required_path}" >&2
    exit 1
  fi
done

run_dir="${output_root}/${run_id}"
framework_root="${run_dir}/framework"
console_root="${run_dir}/console"
config_root="${run_dir}/configs"
master_log="${run_dir}/domainnet126_benchmark_${run_id}.log"
attempt_csv="${run_dir}/domainnet126_task_attempts.csv"
task_summary_csv="${run_dir}/domainnet126_task_summary.csv"
method_summary_csv="${run_dir}/domainnet126_method_summary.csv"

if [ -e "$run_dir" ] && [ "$resume" != "1" ]; then
  echo "Run directory already exists: ${run_dir}" >&2
  echo "Use a different RUN_ID, or set RESUME=1 to continue it." >&2
  exit 1
fi
mkdir -p "$framework_root" "$console_root" "$config_root"

if [ ! -f "$attempt_csv" ]; then
  printf '%s\n' \
    'method,display_name,implementation,task,source,target,peak_acc,final_acc,peak_minus_final,evaluations,train_seconds,train_time,status,seed,console_log,framework_log' \
    > "$attempt_csv"
fi

log() {
  printf '%s\n' "$*" | tee -a "$master_log"
}

log_command() {
  {
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } | tee -a "$master_log"
}

format_duration() {
  local total_seconds=$1
  printf '%02d:%02d:%02d' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

display_name_for() {
  case "$1" in
    shot) printf 'SHOT' ;;
    nrc) printf 'NRC' ;;
    gkd) printf 'GKD' ;;
    adacontrast) printf 'AdaCon' ;;
    cowa) printf 'CoWA' ;;
    sclm) printf 'SCLM' ;;
    tpds) printf 'TPDS' ;;
    difo) printf 'DIFO' ;;
    plmatch) printf 'PLMatch' ;;
  esac
}

task_is_complete() {
  local method=$1
  local task=$2
  awk -F',' -v method="$method" -v task="$task" '
    NR > 1 && $1 == method && $4 == task && $13 == "completed" {
      found = 1
    }
    END { exit(found ? 0 : 1) }
  ' "$attempt_csv"
}

accuracy_values() {
  local method=$1
  local task=$2
  local console_log=$3

  case "$method" in
    adacontrast)
      sed -nE \
        's/.*EPOCH:[[:space:]]*[0-9]+\/[0-9]+[[:space:]]+ACC[[:space:]]+([0-9]+([.][0-9]+)?)%.*/\1/p' \
        "$console_log"
      ;;
    cowa)
      sed -nE \
        's/.*Model Prediction[[:space:]]*:[[:space:]]*Accuracy[[:space:]]*=[[:space:]]*([0-9]+([.][0-9]+)?)%.*/\1/p' \
        "$console_log"
      ;;
    *)
      sed -nE \
        "/Task:[[:space:]]*${task},.*Accuracy([[:space:]]+on[[:space:]]+target)?[[:space:]]*=/s/.*Accuracy([[:space:]]+on[[:space:]]+target)?[[:space:]]*=[[:space:]]*([0-9]+([.][0-9]+)?)%.*/\\2/p" \
        "$console_log"
      ;;
  esac
}

accuracy_summary() {
  local method=$1
  local task=$2
  local console_log=$3

  accuracy_values "$method" "$task" "$console_log" | awk '
    BEGIN { best = -1; count = 0 }
    {
      value = $1 + 0
      if (value > best) best = value
      final = value
      count += 1
    }
    END {
      if (count > 0) printf "%.2f %.2f %d", best, final, count
    }
  '
}

append_attempt() {
  local method=$1
  local display_name=$2
  local implementation=$3
  local task=$4
  local source_domain=$5
  local target_domain=$6
  local peak_acc=$7
  local final_acc=$8
  local evaluations=$9
  local train_seconds=${10}
  local train_time=${11}
  local status=${12}
  local console_log=${13}
  local framework_log=${14}
  local delta=NA

  if [ "$peak_acc" != "NA" ] && [ "$final_acc" != "NA" ]; then
    delta=$(awk -v peak="$peak_acc" -v final="$final_acc" \
      'BEGIN { printf "%.2f", peak - final }')
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%s,%s,%s,%s,%s\n' \
    "$method" "$display_name" "$implementation" "$task" \
    "$source_domain" "$target_domain" "$peak_acc" "$final_acc" \
    "$delta" "$evaluations" "$train_seconds" "$train_time" "$status" \
    "$seed" "$console_log" "$framework_log" >> "$attempt_csv"
}

build_summaries() {
  printf '%s\n' \
    'method,display_name,implementation,task,source,target,acc,reported_metric,peak_acc,final_acc,peak_minus_final,evaluations,train_seconds,train_time,status,seed,console_log,framework_log' \
    > "$task_summary_csv"

  local method display_name implementation task latest_line
  local source_domain target_domain peak_acc final_acc delta evaluations
  local train_seconds train_time status row_seed console_log framework_log
  local selected_acc
  for method in "${methods[@]}"; do
    display_name=$(display_name_for "$method")
    implementation=$method
    for task in "${tasks[@]}"; do
      latest_line=$(awk -F',' -v method="$method" -v task="$task" '
        NR > 1 && $1 == method && $4 == task { line = $0 }
        END { print line }
      ' "$attempt_csv")
      if [ -z "$latest_line" ]; then
        printf '%s,%s,%s,%s,,,NA,%s,NA,NA,NA,0,0,00:00:00,pending,%s,,\n' \
          "$method" "$display_name" "$implementation" "$task" \
          "$report_metric" "$seed" >> "$task_summary_csv"
        continue
      fi

      IFS=',' read -r _ _ _ _ source_domain target_domain peak_acc \
        final_acc delta evaluations train_seconds train_time status row_seed \
        console_log framework_log <<< "$latest_line"
      if [ "$status" = "completed" ]; then
        if [ "$report_metric" = "peak" ]; then
          selected_acc=$peak_acc
        else
          selected_acc=$final_acc
        fi
      else
        selected_acc=NA
      fi
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$method" "$display_name" "$implementation" "$task" \
        "$source_domain" "$target_domain" "$selected_acc" "$report_metric" \
        "$peak_acc" "$final_acc" "$delta" "$evaluations" \
        "$train_seconds" "$train_time" "$status" "$row_seed" \
        "$console_log" "$framework_log" >> "$task_summary_csv"
    done
  done

  printf '%s\n' \
    'method,display_name,implementation,completed_tasks,expected_tasks,avg_acc,reported_metric,avg_peak_acc,avg_final_acc,total_train_seconds,total_train_time,status,seed' \
    > "$method_summary_csv"

  local aggregate completed avg_acc avg_peak avg_final total_seconds
  local total_time method_status
  for method in "${methods[@]}"; do
    display_name=$(display_name_for "$method")
    aggregate=$(awk -F',' -v method="$method" '
      NR > 1 && $1 == method && $15 == "completed" {
        selected_sum += $7
        peak_sum += $9
        final_sum += $10
        total_seconds += $13
        count += 1
      }
      END {
        if (count > 0) {
          printf "%d %.2f %.2f %.2f %d", count, selected_sum / count,
            peak_sum / count, final_sum / count, total_seconds
        } else {
          printf "0 NA NA NA 0"
        }
      }
    ' "$task_summary_csv")
    read -r completed avg_acc avg_peak avg_final total_seconds <<< "$aggregate"
    total_time=$(format_duration "$total_seconds")
    if [ "$completed" -eq 12 ]; then
      method_status=completed
    else
      method_status=incomplete
    fi
    printf '%s,%s,%s,%d,12,%s,%s,%s,%s,%d,%s,%s,%s\n' \
      "$method" "$display_name" "$method" "$completed" "$avg_acc" \
      "$report_metric" "$avg_peak" "$avg_final" "$total_seconds" \
      "$total_time" "$method_status" "$seed" >> "$method_summary_csv"
  done
}

for method in "${methods[@]}"; do
  if [ ! -f "${config_root}/${method}.yaml" ]; then
    cp "cfgs/domainnet126/${method}.yaml" "${config_root}/${method}.yaml"
  fi
done

session_started_epoch=$(date +%s)
log "DomainNet-126 nine-method benchmark"
log "run_id=${run_id}"
log "started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
log "methods=${methods[*]}"
log "tasks=${tasks[*]}"
log "seed=${seed}"
log "report_metric=${report_metric}"
log "checkpoint_root=${checkpoint_root}"
log "run_dir=${run_dir}"
log "accuracy_definition=target accuracy per task; equal average over 12 tasks"
if [ -n "${DATA_DIR:-}" ]; then
  log "data_dir=${DATA_DIR}"
fi
if command -v git >/dev/null 2>&1; then
  log "git_commit=$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    log "git_worktree=dirty"
  else
    log "git_worktree=clean"
  fi
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_info=$(nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader 2>/dev/null | head -n 1 || true)
  if [ -n "$gpu_info" ]; then
    log "gpu_info=${gpu_info}"
  fi
fi

for method in "${methods[@]}"; do
  display_name=$(display_name_for "$method")
  cfg_file="${config_root}/${method}.yaml"
  log ""
  log "========== METHOD ${display_name} =========="

  for s in 0 1 2 3; do
    for t in 0 1 2 3; do
      if [ "$s" -eq "$t" ]; then
        continue
      fi

      task="${domain_keys[$s]}${domain_keys[$t]}"
      source_domain=${domain_names[$s]}
      target_domain=${domain_names[$t]}
      if task_is_complete "$method" "$task"; then
        log "SKIP [${display_name}/${task}]: already complete"
        continue
      fi

      console_log="${console_root}/${method}_${task}.log"
      framework_task_dir="${framework_root}/uda/domainnet126/${task}/${method}"
      : > "$console_log"
      started_epoch=$(date +%s)

      log ""
      log "==> [${display_name}/${task}] ${source_domain} -> ${target_domain}"
      run_command=(
        "$python_bin" image_target_in_126.py
        --cfg "$cfg_file"
        SAVE_DIR "$framework_root"
        CKPT_DIR "$checkpoint_root"
        FOLDER "$folder_root"
        MODEL.METHOD "$method"
        SETTING.OUTPUT_SRC source
        SETTING.SEED "$seed"
        SETTING.S "$s"
        SETTING.T "$t"
      )
      if [ -n "${DATA_DIR:-}" ]; then
        run_command+=(DATA_DIR "$DATA_DIR")
      fi
      run_command+=("$@")
      log_command "${run_command[@]}"

      if PYTHONUNBUFFERED=1 \
        "${run_command[@]}" 2>&1 | tee -a "$master_log" "$console_log"; then
        run_status=0
      else
        run_status=$?
      fi

      finished_epoch=$(date +%s)
      train_seconds=$((finished_epoch - started_epoch))
      train_time=$(format_duration "$train_seconds")
      framework_log=$(find "$framework_task_dir" -maxdepth 1 -type f \
        -name '*.txt' -print 2>/dev/null | sort | tail -n 1 || true)

      if [ "$run_status" -ne 0 ]; then
        append_attempt "$method" "$display_name" "$method" "$task" \
          "$source_domain" "$target_domain" NA NA 0 "$train_seconds" \
          "$train_time" failed "$console_log" "$framework_log"
        build_summaries
        log "<== [${display_name}/${task}] FAILED: exit=${run_status}, time=${train_time}"
        if [ "$continue_on_error" != "1" ]; then
          exit "$run_status"
        fi
        continue
      fi

      acc_summary=$(accuracy_summary "$method" "$task" "$console_log")
      if [ -z "$acc_summary" ]; then
        append_attempt "$method" "$display_name" "$method" "$task" \
          "$source_domain" "$target_domain" NA NA 0 "$train_seconds" \
          "$train_time" no_accuracy "$console_log" "$framework_log"
        build_summaries
        log "<== [${display_name}/${task}] FAILED: no target accuracy found"
        if [ "$continue_on_error" != "1" ]; then
          exit 2
        fi
        continue
      fi

      read -r peak_acc final_acc evaluations <<< "$acc_summary"
      append_attempt "$method" "$display_name" "$method" "$task" \
        "$source_domain" "$target_domain" "$peak_acc" "$final_acc" \
        "$evaluations" "$train_seconds" "$train_time" completed \
        "$console_log" "$framework_log"
      build_summaries
      log "<== [${display_name}/${task}] peak=${peak_acc}%, final=${final_acc}%, evaluations=${evaluations}, time=${train_time}"
    done
  done

  build_summaries
  method_line=$(awk -F',' -v method="$method" \
    'NR > 1 && $1 == method { print }' "$method_summary_csv")
  log "method_summary=${method_line}"
done

build_summaries
session_finished_epoch=$(date +%s)
session_seconds=$((session_finished_epoch - session_started_epoch))
session_time=$(format_duration "$session_seconds")

log ""
log "================ FINAL SUMMARY ================"
while IFS=',' read -r method display_name implementation completed expected \
  avg_acc metric avg_peak avg_final total_seconds total_time status row_seed; do
  if [ "$method" = "method" ]; then
    continue
  fi
  log "${display_name}: tasks=${completed}/${expected}, avg_acc=${avg_acc}% (${metric}), avg_peak=${avg_peak}%, avg_final=${avg_final}%, time=${total_time}, status=${status}"
done < "$method_summary_csv"
log "session_wall_time=${session_time} (${session_seconds}s)"
log "attempts_csv=${attempt_csv}"
log "task_summary_csv=${task_summary_csv}"
log "method_summary_csv=${method_summary_csv}"
log "master_log=${master_log}"

incomplete_methods=$(awk -F',' \
  'NR > 1 && $12 != "completed" { count += 1 } END { print count + 0 }' \
  "$method_summary_csv")
if [ "$incomplete_methods" -ne 0 ]; then
  log "ERROR: ${incomplete_methods} selected methods are incomplete."
  exit 3
fi
