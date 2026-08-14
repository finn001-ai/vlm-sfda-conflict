#!/usr/bin/env bash
set -euo pipefail

# Run Source + nine SFDA methods on all 12 directed Office-Home tasks.
#
# Default methods:
#   source shot nrc gkd adacontrast cowa sclm tpds difo plmatch
#
# Usage:
#   GPU_ID=0 bash tools/run_office_home_benchmark_all.sh
#   METHODS="shot difo plmatch" CHECKPOINT_ROOT=/path/to/checkpoints \
#     bash tools/run_office_home_benchmark_all.sh
#   RUN_ID=20260814_120000 RESUME=1 \
#     bash tools/run_office_home_benchmark_all.sh
#
# Optional environment variables:
#   GPU_ID=0
#   PYTHON_BIN=python
#   SEED=2020
#   METHODS="source shot nrc gkd adacontrast cowa sclm tpds difo plmatch"
#   OUTPUT_ROOT=output/office_home_benchmark_runs
#   RUN_ID=<current timestamp>
#   RESUME=0
#   CONTINUE_ON_ERROR=0
#   CHECKPOINT_ROOT=<run directory>/checkpoints
#   FOLDER_ROOT=<repository>/data/
#   DATA_DIR=/path/to/image-data
#
# Extra command-line arguments are appended as common YACS overrides to every
# method. Only use keys that exist in every selected method configuration.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${script_dir}/.." && pwd)
cd "$repo_dir"

gpu_id=${GPU_ID:-0}
python_bin=${PYTHON_BIN:-python}
seed=${SEED:-2020}
methods_text=${METHODS:-"source shot nrc gkd adacontrast cowa sclm tpds difo plmatch"}
output_root=${OUTPUT_ROOT:-output/office_home_benchmark_runs}
run_id=${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}
resume=${RESUME:-0}
continue_on_error=${CONTINUE_ON_ERROR:-0}

case "$output_root" in
  /*) ;;
  *) output_root="${repo_dir}/${output_root}" ;;
esac

run_dir="${output_root}/${run_id}"
framework_root="${run_dir}/framework"
console_root="${run_dir}/console"
source_staging_root="${run_dir}/source_staging"
checkpoint_root=${CHECKPOINT_ROOT:-${run_dir}/checkpoints}
folder_root=${FOLDER_ROOT:-${repo_dir}/data/}

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

master_log="${run_dir}/office_home_benchmark_${run_id}.log"
task_csv="${run_dir}/office_home_task_results.csv"
job_csv="${run_dir}/office_home_job_timing.csv"
method_csv="${run_dir}/office_home_method_summary.csv"

read -r -a methods <<< "$methods_text"
domain_names=(Art Clipart Product RealWorld)
domain_keys=(A C P R)

if [ "${#methods[@]}" -eq 0 ]; then
  echo "METHODS is empty." >&2
  exit 1
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
fi

for value in "$resume" "$continue_on_error"; do
  if [ "$value" != "0" ] && [ "$value" != "1" ]; then
    echo "RESUME and CONTINUE_ON_ERROR must be 0 or 1." >&2
    exit 1
  fi
done

for method in "${methods[@]}"; do
  case "$method" in
    source|shot|nrc|gkd|adacontrast|cowa|sclm|tpds|difo|plmatch) ;;
    *)
      echo "Unsupported method: ${method}" >&2
      exit 1
      ;;
  esac
  if [ ! -f "cfgs/office-home/${method}.yaml" ]; then
    echo "Missing method config: cfgs/office-home/${method}.yaml" >&2
    exit 1
  fi
done

for required_file in image_target_of_oh_vs.py data/office-home/classname.txt; do
  if [ ! -f "$required_file" ]; then
    echo "Missing required file: ${repo_dir}/${required_file}" >&2
    exit 1
  fi
done
for domain in "${domain_names[@]}"; do
  list_file="${folder_root}office-home/${domain}_list.txt"
  if [ ! -f "$list_file" ]; then
    echo "Missing Office-Home list: ${list_file}" >&2
    exit 1
  fi
done

if [ -e "$run_dir" ] && [ "$resume" != "1" ]; then
  echo "Run directory already exists: ${run_dir}" >&2
  echo "Use a different RUN_ID, or set RESUME=1 to continue it." >&2
  exit 1
fi

mkdir -p "$framework_root" "$console_root" "$source_staging_root" "$checkpoint_root"

if [ ! -f "$task_csv" ]; then
  printf '%s\n' \
    'method,implementation,task,source,target,best_acc,final_acc,best_minus_final,train_seconds,train_time,time_scope,status,seed,console_log,framework_log' \
    > "$task_csv"
fi
if [ ! -f "$job_csv" ]; then
  printf '%s\n' \
    'method,implementation,job_id,tasks,started_at,finished_at,train_seconds,train_time,status,exit_code,console_log,framework_log' \
    > "$job_csv"
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

implementation_for() {
  local method=$1
  if [ "$method" = "nrc" ]; then
    # nrc_vs is the repository's NRC implementation compatible with the
    # common ResBase + bottleneck + classifier source checkpoints.
    printf 'nrc_vs'
  else
    printf '%s' "$method"
  fi
}

task_is_complete() {
  local method=$1
  local task=$2
  awk -F',' -v method="$method" -v task="$task" '
    NR > 1 && $1 == method && $3 == task && $12 == "completed" { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "$task_csv"
}

accuracy_values() {
  local method=$1
  local task=$2
  local task_log=$3

  case "$method" in
    source)
      sed -nE \
        "/Task:[[:space:]]*${task},[[:space:]]*Accuracy[[:space:]]*=/s/.*Accuracy[[:space:]]*=[[:space:]]*([0-9]+([.][0-9]+)?)%.*/\1/p" \
        "$task_log"
      ;;
    adacontrast)
      sed -nE \
        's/.*EPOCH:[[:space:]]*[0-9]+\/[0-9]+[[:space:]]+ACC[[:space:]]+([0-9]+([.][0-9]+)?)%.*/\1/p' \
        "$task_log"
      ;;
    cowa)
      sed -nE \
        's/.*Model Prediction[[:space:]]*:[[:space:]]*Accuracy[[:space:]]*=[[:space:]]*([0-9]+([.][0-9]+)?)%.*/\1/p' \
        "$task_log"
      ;;
    *)
      sed -nE \
        "/Task:[[:space:]]*${task},.*Accuracy([[:space:]]+on[[:space:]]+target)?[[:space:]]*=/s/.*Accuracy([[:space:]]+on[[:space:]]+target)?[[:space:]]*=[[:space:]]*([0-9]+([.][0-9]+)?)%.*/\2/p" \
        "$task_log"
      ;;
  esac
}

accuracy_summary() {
  local method=$1
  local task=$2
  local task_log=$3

  accuracy_values "$method" "$task" "$task_log" | awk '
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

append_task_result() {
  local method=$1
  local implementation=$2
  local task=$3
  local source_domain=$4
  local target_domain=$5
  local best_acc=$6
  local final_acc=$7
  local train_seconds=$8
  local train_time=$9
  local time_scope=${10}
  local status=${11}
  local console_log=${12}
  local framework_log=${13}
  local delta=NA

  if [ "$best_acc" != "NA" ] && [ "$final_acc" != "NA" ]; then
    delta=$(awk -v best="$best_acc" -v final="$final_acc" \
      'BEGIN { printf "%.2f", best - final }')
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%d,%s,%s,%s,%s,%s,%s\n' \
    "$method" "$implementation" "$task" "$source_domain" "$target_domain" \
    "$best_acc" "$final_acc" "$delta" "$train_seconds" "$train_time" \
    "$time_scope" "$status" "$seed" "$console_log" "$framework_log" \
    >> "$task_csv"
}

append_job_result() {
  local method=$1
  local implementation=$2
  local job_id=$3
  local tasks=$4
  local started_at=$5
  local finished_at=$6
  local train_seconds=$7
  local train_time=$8
  local status=$9
  local exit_code=${10}
  local console_log=${11}
  local framework_log=${12}

  printf '%s,%s,%s,%s,%s,%s,%d,%s,%s,%s,%s,%s\n' \
    "$method" "$implementation" "$job_id" "$tasks" "$started_at" \
    "$finished_at" "$train_seconds" "$train_time" "$status" "$exit_code" \
    "$console_log" "$framework_log" >> "$job_csv"
}

build_method_summary() {
  printf '%s\n' \
    'method,implementation,completed_tasks,expected_tasks,avg_best_acc,avg_final_acc,total_train_seconds,total_train_time,completed_jobs,avg_job_seconds,avg_job_time,status' \
    > "$method_csv"

  local method implementation task_stats completed avg_best avg_final
  local job_stats completed_jobs total_seconds avg_seconds avg_seconds_rounded
  local total_time avg_time method_status

  for method in "${methods[@]}"; do
    implementation=$(implementation_for "$method")
    task_stats=$(awk -F',' -v method="$method" '
      NR > 1 && $1 == method && $12 == "completed" {
        best_sum += $6
        final_sum += $7
        count += 1
      }
      END {
        if (count > 0) {
          printf "%d %.2f %.2f", count, best_sum / count, final_sum / count
        } else {
          printf "0 NA NA"
        }
      }
    ' "$task_csv")
    read -r completed avg_best avg_final <<< "$task_stats"

    job_stats=$(awk -F',' -v method="$method" '
      NR > 1 && $1 == method && $9 == "completed" {
        total += $7
        count += 1
      }
      END {
        if (count > 0) {
          printf "%d %d %.2f", count, total, total / count
        } else {
          printf "0 0 0.00"
        }
      }
    ' "$job_csv")
    read -r completed_jobs total_seconds avg_seconds <<< "$job_stats"
    avg_seconds_rounded=$(awk -v value="$avg_seconds" 'BEGIN { printf "%.0f", value }')
    total_time=$(format_duration "$total_seconds")
    avg_time=$(format_duration "$avg_seconds_rounded")

    if [ "$completed" -eq 12 ]; then
      method_status=completed
    else
      method_status=incomplete
    fi

    printf '%s,%s,%d,12,%s,%s,%d,%s,%d,%s,%s,%s\n' \
      "$method" "$implementation" "$completed" "$avg_best" "$avg_final" \
      "$total_seconds" "$total_time" "$completed_jobs" "$avg_seconds" \
      "$avg_time" "$method_status" >> "$method_csv"
  done
}

source_weights_ready() {
  local key part path
  for key in "${domain_keys[@]}"; do
    for part in F B C; do
      path="${checkpoint_root}/source/uda/office-home/${key}/source_${part}.pt"
      if [ ! -f "$path" ]; then
        return 1
      fi
    done
  done
  return 0
}

method_selected() {
  local wanted=$1
  local method
  for method in "${methods[@]}"; do
    if [ "$method" = "$wanted" ]; then
      return 0
    fi
  done
  return 1
}

config_snapshot_dir="${run_dir}/configs"
mkdir -p "$config_snapshot_dir"
for method in "${methods[@]}"; do
  snapshot="${config_snapshot_dir}/${method}.yaml"
  if [ ! -f "$snapshot" ]; then
    cp "cfgs/office-home/${method}.yaml" "$snapshot"
  fi
done

session_started_epoch=$(date +%s)
log "Office-Home multi-method benchmark"
log "run_id=${run_id}"
log "session_started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
log "methods=${methods[*]}"
log "seed=${seed}"
log "physical_gpu=${gpu_id}; process_gpu=0"
log "python=${python_bin}"
log "checkpoint_root=${checkpoint_root}"
log "folder_root=${folder_root}"
log "run_dir=${run_dir}"
log "resume=${resume}"
log "nrc_implementation=nrc_vs (shared source checkpoint compatible)"
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
    --format=csv,noheader -i "$gpu_id" 2>/dev/null | head -n 1 || true)
  if [ -n "$gpu_info" ]; then
    log "gpu_info=${gpu_info}"
  fi
fi

if method_selected source; then
  implementation=source
  for s in 0 1 2 3; do
    source_key=${domain_keys[$s]}
    source_domain=${domain_names[$s]}
    all_tasks_complete=1
    source_tasks=""
    initial_t=$(( (s + 1) % 4 ))
    initial_task="${source_key}${domain_keys[$initial_t]}"

    for t in 0 1 2 3; do
      if [ "$s" -eq "$t" ]; then
        continue
      fi
      task="${source_key}${domain_keys[$t]}"
      if [ -n "$source_tasks" ]; then
        source_tasks="${source_tasks}|"
      fi
      source_tasks="${source_tasks}${task}"
      if ! task_is_complete source "$task"; then
        all_tasks_complete=0
      fi
    done

    source_dest="${checkpoint_root}/source/uda/office-home/${source_key}"
    weights_complete=1
    for part in F B C; do
      if [ ! -f "${source_dest}/source_${part}.pt" ]; then
        weights_complete=0
      fi
    done

    if [ "$all_tasks_complete" -eq 1 ] && [ "$weights_complete" -eq 1 ]; then
      log "SKIP source-${source_key}: results and checkpoints already complete"
      continue
    fi

    stage_dir="${source_staging_root}/${source_key}"
    task_console_log="${console_root}/source_${source_key}.log"
    mkdir -p "$stage_dir" "$source_dest"
    job_started_epoch=$(date +%s)
    job_started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')

    log ""
    log "==> [source-${source_key}] train ${source_domain}; evaluate ${source_tasks}"
    run_command=(
      "$python_bin" "${repo_dir}/image_target_of_oh_vs.py"
      --cfg "${config_snapshot_dir}/source.yaml"
      GPU_ID 0
      SAVE_DIR "$framework_root"
      CKPT_DIR "$checkpoint_root"
      FOLDER "$folder_root"
      SETTING.OUTPUT_SRC source
      SETTING.SEED "$seed"
      SETTING.S "$s"
      SETTING.T "$initial_t"
    )
    if [ -n "${DATA_DIR:-}" ]; then
      run_command+=(DATA_DIR "$DATA_DIR")
    fi
    run_command+=("$@")
    log_command "${run_command[@]}"

    if (
      cd "$stage_dir"
      CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 "${run_command[@]}"
    ) 2>&1 | tee -a "$master_log" "$task_console_log"; then
      run_status=0
    else
      run_status=$?
    fi

    job_finished_epoch=$(date +%s)
    job_finished_at=$(date '+%Y-%m-%dT%H:%M:%S%z')
    job_seconds=$((job_finished_epoch - job_started_epoch))
    job_time=$(format_duration "$job_seconds")
    framework_log=$(find "${framework_root}/uda/office-home/${initial_task}/source" \
      -maxdepth 1 -type f -name 'source_*.txt' -print 2>/dev/null \
      | sort | tail -n 1 || true)

    if [ "$run_status" -ne 0 ]; then
      append_job_result source source "source-${source_key}" "$source_tasks" \
        "$job_started_at" "$job_finished_at" "$job_seconds" "$job_time" \
        failed "$run_status" "$task_console_log" "$framework_log"
      for t in 0 1 2 3; do
        if [ "$s" -eq "$t" ]; then
          continue
        fi
        task="${source_key}${domain_keys[$t]}"
        if ! task_is_complete source "$task"; then
          append_task_result source source "$task" "$source_domain" \
            "${domain_names[$t]}" NA NA "$job_seconds" "$job_time" \
            "shared_source_${source_key}_run" failed "$task_console_log" \
            "$framework_log"
        fi
      done
      build_method_summary
      log "<== [source-${source_key}] FAILED (exit=${run_status}, time=${job_time})"
      exit "$run_status"
    fi

    missing_weight=0
    for part in F B C; do
      staged_weight="${stage_dir}/source/source_${part}.pt"
      if [ ! -f "$staged_weight" ]; then
        log "Missing staged source checkpoint: ${staged_weight}"
        missing_weight=1
      fi
    done
    if [ "$missing_weight" -ne 0 ]; then
      append_job_result source source "source-${source_key}" "$source_tasks" \
        "$job_started_at" "$job_finished_at" "$job_seconds" "$job_time" \
        failed 2 "$task_console_log" "$framework_log"
      build_method_summary
      exit 2
    fi
    for part in F B C; do
      mv -f "${stage_dir}/source/source_${part}.pt" \
        "${source_dest}/source_${part}.pt"
    done

    parse_failed=0
    for t in 0 1 2 3; do
      if [ "$s" -eq "$t" ]; then
        continue
      fi
      task="${source_key}${domain_keys[$t]}"
      if task_is_complete source "$task"; then
        continue
      fi
      acc_summary=$(accuracy_summary source "$task" "$task_console_log")
      if [ -z "$acc_summary" ]; then
        append_task_result source source "$task" "$source_domain" \
          "${domain_names[$t]}" NA NA "$job_seconds" "$job_time" \
          "shared_source_${source_key}_run" no_accuracy "$task_console_log" \
          "$framework_log"
        parse_failed=1
      else
        read -r best_acc final_acc eval_count <<< "$acc_summary"
        append_task_result source source "$task" "$source_domain" \
          "${domain_names[$t]}" "$best_acc" "$final_acc" "$job_seconds" \
          "$job_time" "shared_source_${source_key}_run" completed \
          "$task_console_log" "$framework_log"
        log "[source/${task}] best=${best_acc}%, final=${final_acc}%, evaluations=${eval_count}"
      fi
    done

    if [ "$parse_failed" -eq 0 ]; then
      job_status=completed
      job_exit=0
    else
      job_status=no_accuracy
      job_exit=2
    fi
    append_job_result source source "source-${source_key}" "$source_tasks" \
      "$job_started_at" "$job_finished_at" "$job_seconds" "$job_time" \
      "$job_status" "$job_exit" "$task_console_log" "$framework_log"
    log "<== [source-${source_key}] status=${job_status}, time=${job_time}"

    if [ "$parse_failed" -ne 0 ] && [ "$continue_on_error" != "1" ]; then
      build_method_summary
      exit 2
    fi
  done
  build_method_summary
fi

adaptation_selected=0
for method in "${methods[@]}"; do
  if [ "$method" != "source" ]; then
    adaptation_selected=1
  fi
done

if [ "$adaptation_selected" -eq 1 ] && ! source_weights_ready; then
  log "Missing common Office-Home source checkpoints under ${checkpoint_root}."
  log "Include source in METHODS, or set CHECKPOINT_ROOT to existing compatible checkpoints."
  exit 2
fi

for method in "${methods[@]}"; do
  if [ "$method" = "source" ]; then
    continue
  fi

  implementation=$(implementation_for "$method")
  cfg_file="${config_snapshot_dir}/${method}.yaml"
  log ""
  log "========== METHOD ${method} (implementation=${implementation}) =========="

  for s in 0 1 2 3; do
    for t in 0 1 2 3; do
      if [ "$s" -eq "$t" ]; then
        continue
      fi

      task="${domain_keys[$s]}${domain_keys[$t]}"
      source_domain=${domain_names[$s]}
      target_domain=${domain_names[$t]}
      if task_is_complete "$method" "$task"; then
        log "SKIP [${method}/${task}]: already complete"
        continue
      fi

      task_console_log="${console_root}/${method}_${task}.log"
      task_output_dir="${framework_root}/uda/office-home/${task}/${implementation}"
      job_started_epoch=$(date +%s)
      job_started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')

      log ""
      log "==> [${method}/${task}] ${source_domain} -> ${target_domain}"
      run_command=(
        "$python_bin" image_target_of_oh_vs.py
        --cfg "$cfg_file"
        GPU_ID 0
        SAVE_DIR "$framework_root"
        CKPT_DIR "$checkpoint_root"
        FOLDER "$folder_root"
        MODEL.METHOD "$implementation"
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

      if CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 \
        "${run_command[@]}" 2>&1 | tee -a "$master_log" "$task_console_log"; then
        run_status=0
      else
        run_status=$?
      fi

      job_finished_epoch=$(date +%s)
      job_finished_at=$(date '+%Y-%m-%dT%H:%M:%S%z')
      job_seconds=$((job_finished_epoch - job_started_epoch))
      job_time=$(format_duration "$job_seconds")
      framework_log=$(find "$task_output_dir" -maxdepth 1 -type f \
        \( -name "${method}_*.txt" -o -name "${implementation}_*.txt" \) 2>/dev/null \
        | sort | tail -n 1 || true)

      if [ "$run_status" -ne 0 ]; then
        append_job_result "$method" "$implementation" "$task" "$task" \
          "$job_started_at" "$job_finished_at" "$job_seconds" "$job_time" \
          failed "$run_status" "$task_console_log" "$framework_log"
        append_task_result "$method" "$implementation" "$task" \
          "$source_domain" "$target_domain" NA NA "$job_seconds" "$job_time" \
          task failed "$task_console_log" "$framework_log"
        log "<== [${method}/${task}] FAILED (exit=${run_status}, time=${job_time})"
        if [ "$continue_on_error" != "1" ]; then
          build_method_summary
          exit "$run_status"
        fi
        continue
      fi

      acc_summary=$(accuracy_summary "$method" "$task" "$task_console_log")
      if [ -z "$acc_summary" ]; then
        append_job_result "$method" "$implementation" "$task" "$task" \
          "$job_started_at" "$job_finished_at" "$job_seconds" "$job_time" \
          no_accuracy 2 "$task_console_log" "$framework_log"
        append_task_result "$method" "$implementation" "$task" \
          "$source_domain" "$target_domain" NA NA "$job_seconds" "$job_time" \
          task no_accuracy "$task_console_log" "$framework_log"
        log "<== [${method}/${task}] FAILED: no target accuracy found"
        if [ "$continue_on_error" != "1" ]; then
          build_method_summary
          exit 2
        fi
        continue
      fi

      read -r best_acc final_acc eval_count <<< "$acc_summary"
      append_job_result "$method" "$implementation" "$task" "$task" \
        "$job_started_at" "$job_finished_at" "$job_seconds" "$job_time" \
        completed 0 "$task_console_log" "$framework_log"
      append_task_result "$method" "$implementation" "$task" \
        "$source_domain" "$target_domain" "$best_acc" "$final_acc" \
        "$job_seconds" "$job_time" task completed "$task_console_log" \
        "$framework_log"
      log "<== [${method}/${task}] best=${best_acc}%, final=${final_acc}%, evaluations=${eval_count}, time=${job_time}"
    done
  done

  build_method_summary
  method_line=$(awk -F',' -v method="$method" 'NR > 1 && $1 == method { print }' "$method_csv")
  log "method_summary=${method_line}"
done

build_method_summary
session_finished_epoch=$(date +%s)
session_seconds=$((session_finished_epoch - session_started_epoch))
session_time=$(format_duration "$session_seconds")
global_job_stats=$(awk -F',' '
  NR > 1 && $9 == "completed" { total += $7; count += 1 }
  END {
    if (count > 0) printf "%d %d %.2f", count, total, total / count
    else printf "0 0 0.00"
  }
' "$job_csv")
read -r global_jobs global_seconds global_avg_seconds <<< "$global_job_stats"
global_avg_rounded=$(awk -v value="$global_avg_seconds" 'BEGIN { printf "%.0f", value }')
global_time=$(format_duration "$global_seconds")
global_avg_time=$(format_duration "$global_avg_rounded")

log ""
log "================ FINAL SUMMARY ================"
while IFS=',' read -r method implementation completed expected avg_best avg_final \
  total_seconds total_time completed_jobs avg_seconds avg_time status; do
  if [ "$method" = "method" ]; then
    continue
  fi
  log "${method}: tasks=${completed}/${expected}, avg_best=${avg_best}%, avg_final=${avg_final}%, total_train=${total_time}, avg_job=${avg_time}, status=${status}"
done < "$method_csv"
log "completed_jobs=${global_jobs}"
log "sum_successful_training_time=${global_time} (${global_seconds}s)"
log "average_successful_job_time=${global_avg_time} (${global_avg_seconds}s)"
log "current_session_wall_time=${session_time} (${session_seconds}s)"
log "session_finished_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
log "master_log=${master_log}"
log "task_results=${task_csv}"
log "job_timing=${job_csv}"
log "method_summary=${method_csv}"

incomplete_methods=$(awk -F',' 'NR > 1 && $12 != "completed" { count += 1 } END { print count + 0 }' "$method_csv")
if [ "$incomplete_methods" -ne 0 ]; then
  log "ERROR: ${incomplete_methods} selected methods are incomplete."
  exit 3
fi
