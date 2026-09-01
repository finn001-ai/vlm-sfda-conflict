#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run nine SFDA baselines on the full VisDA-C train -> validation task and
# summarize the macro-average class accuracy reported by each method.
#
# Default:
#   bash tools/run_visda_benchmark_all.sh
#
# Resume an interrupted run:
#   RUN_ID=20260902_120000 RESUME=1 \
#     bash tools/run_visda_benchmark_all.sh
#
# Run a subset:
#   METHODS="shot nrc difo plmatch" \
#     bash tools/run_visda_benchmark_all.sh
#
# Report the fixed final checkpoint as the primary avg_acc column:
#   REPORT_METRIC=final bash tools/run_visda_benchmark_all.sh
#
# Optional environment variables:
#   PYTHON_BIN=python
#   SEED=2020
#   GPU_ID=0
#   METHODS="shot nrc gkd adacontrast cowa sclm tpds difo plmatch"
#   OUTPUT_ROOT=output/visda_benchmark_runs
#   RUN_ID=<current timestamp>
#   RESUME=0
#   CONTINUE_ON_ERROR=1
#   REPORT_METRIC=peak  # peak or final; both are always saved
#   CHECKPOINT_ROOT=<repository root containing source/uda/VISDA-C/T>
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
gpu_id=${GPU_ID:-0}
methods_text=${METHODS:-"shot nrc gkd adacontrast cowa sclm tpds difo plmatch"}
output_root=${OUTPUT_ROOT:-output/visda_benchmark_runs}
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

for method in "${methods[@]}"; do
  if [ ! -f "cfgs/visda/${method}.yaml" ]; then
    echo "Missing method config: cfgs/visda/${method}.yaml" >&2
    exit 1
  fi
done
for required_path in \
  image_target_of_oh_vs.py \
  "${folder_root}VISDA-C/validation_list.txt" \
  data/VISDA-C/classname.txt \
  "${checkpoint_root}/source/uda/VISDA-C/T/source_F.pt" \
  "${checkpoint_root}/source/uda/VISDA-C/T/source_B.pt" \
  "${checkpoint_root}/source/uda/VISDA-C/T/source_C.pt"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing VisDA-C input: ${required_path}" >&2
    exit 1
  fi
done

run_dir="${output_root}/${run_id}"
framework_root="${run_dir}/framework"
console_root="${run_dir}/console"
config_root="${run_dir}/configs"
master_log="${run_dir}/visda_benchmark_${run_id}.log"
result_csv="${run_dir}/visda_method_attempts.csv"
summary_csv="${run_dir}/visda_method_summary.csv"

if [ -e "$run_dir" ] && [ "$resume" != "1" ]; then
  echo "Run directory already exists: ${run_dir}" >&2
  echo "Use a different RUN_ID, or set RESUME=1 to continue it." >&2
  exit 1
fi
mkdir -p "$framework_root" "$console_root" "$config_root"

if [ ! -f "$result_csv" ]; then
  printf '%s\n' \
    'method,display_name,implementation,peak_avg_acc,final_avg_acc,peak_minus_final,evaluations,train_seconds,train_time,status,seed,console_log,framework_log' \
    > "$result_csv"
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

implementation_for() {
  if [ "$1" = "nrc" ]; then
    # The nrc_vs implementation loads the same F/B/C source checkpoints used
    # by the other VisDA-C methods.
    printf 'nrc_vs'
  else
    printf '%s' "$1"
  fi
}

method_is_complete() {
  local method=$1
  awk -F',' -v method="$method" '
    NR > 1 && $1 == method && $10 == "completed" { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "$result_csv"
}

accuracy_values() {
  local method=$1
  local console_log=$2

  case "$method" in
    adacontrast|cowa)
      sed -nE \
        's/.*VISDA-C classwise accuracy[[:space:]]*:[[:space:]]*([0-9]+([.][0-9]+)?)%.*/\1/p' \
        "$console_log"
      ;;
    *)
      sed -nE \
        '/Task:[[:space:]]*TV,.*Accuracy([[:space:]]+on[[:space:]]+target)?[[:space:]]*=/s/.*Accuracy([[:space:]]+on[[:space:]]+target)?[[:space:]]*=[[:space:]]*([0-9]+([.][0-9]+)?)%.*/\2/p' \
        "$console_log"
      ;;
  esac
}

accuracy_summary() {
  local method=$1
  local console_log=$2

  accuracy_values "$method" "$console_log" | awk '
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

append_result() {
  local method=$1
  local display_name=$2
  local implementation=$3
  local peak_acc=$4
  local final_acc=$5
  local evaluations=$6
  local train_seconds=$7
  local train_time=$8
  local status=$9
  local console_log=${10}
  local framework_log=${11}
  local delta=NA

  if [ "$peak_acc" != "NA" ] && [ "$final_acc" != "NA" ]; then
    delta=$(awk -v peak="$peak_acc" -v final="$final_acc" \
      'BEGIN { printf "%.2f", peak - final }')
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%d,%s,%s,%s,%s,%s\n' \
    "$method" "$display_name" "$implementation" "$peak_acc" \
    "$final_acc" "$delta" "$evaluations" "$train_seconds" \
    "$train_time" "$status" "$seed" "$console_log" "$framework_log" \
    >> "$result_csv"
}

build_summary() {
  printf '%s\n' \
    'method,display_name,implementation,avg_acc,reported_metric,peak_avg_acc,final_avg_acc,peak_minus_final,evaluations,train_seconds,train_time,status,seed,console_log,framework_log' \
    > "$summary_csv"

  local method display_name implementation latest_line
  local peak_acc final_acc delta evaluations train_seconds train_time
  local status row_seed console_log framework_log selected_acc
  for method in "${methods[@]}"; do
    display_name=$(display_name_for "$method")
    implementation=$(implementation_for "$method")
    latest_line=$(awk -F',' -v method="$method" \
      'NR > 1 && $1 == method { line = $0 } END { print line }' \
      "$result_csv")
    if [ -z "$latest_line" ]; then
      printf '%s,%s,%s,NA,%s,NA,NA,NA,0,0,00:00:00,pending,%s,,\n' \
        "$method" "$display_name" "$implementation" "$report_metric" \
        "$seed" >> "$summary_csv"
      continue
    fi

    IFS=',' read -r _ _ _ peak_acc final_acc delta evaluations \
      train_seconds train_time status row_seed console_log framework_log \
      <<< "$latest_line"
    if [ "$status" = "completed" ]; then
      if [ "$report_metric" = "peak" ]; then
        selected_acc=$peak_acc
      else
        selected_acc=$final_acc
      fi
    else
      selected_acc=NA
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$method" "$display_name" "$implementation" "$selected_acc" \
      "$report_metric" "$peak_acc" "$final_acc" "$delta" "$evaluations" \
      "$train_seconds" "$train_time" "$status" "$row_seed" \
      "$console_log" "$framework_log" >> "$summary_csv"
  done
}

for method in "${methods[@]}"; do
  if [ ! -f "${config_root}/${method}.yaml" ]; then
    cp "cfgs/visda/${method}.yaml" "${config_root}/${method}.yaml"
  fi
done

session_started_epoch=$(date +%s)
log "VisDA-C nine-method benchmark"
log "run_id=${run_id}"
log "started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
log "methods=${methods[*]}"
log "seed=${seed}"
log "gpu_id=${gpu_id}"
log "report_metric=${report_metric}"
log "checkpoint_root=${checkpoint_root}"
log "run_dir=${run_dir}"
log "nrc_implementation=nrc_vs"
log "accuracy_definition=VisDA-C macro-average class accuracy"
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
  implementation=$(implementation_for "$method")
  if method_is_complete "$method"; then
    log "SKIP [${display_name}]: already complete"
    continue
  fi

  cfg_file="${config_root}/${method}.yaml"
  console_log="${console_root}/${method}.log"
  framework_method_dir="${framework_root}/uda/VISDA-C/TV/${implementation}"
  : > "$console_log"

  started_epoch=$(date +%s)
  log ""
  log "==> [${display_name}] implementation=${implementation}"
  run_command=(
    "$python_bin" image_target_of_oh_vs.py
    --cfg "$cfg_file"
    SAVE_DIR "$framework_root"
    CKPT_DIR "$checkpoint_root"
    FOLDER "$folder_root"
    GPU_ID "$gpu_id"
    MODEL.METHOD "$implementation"
    SETTING.OUTPUT_SRC source
    SETTING.SEED "$seed"
    SETTING.S 0
    SETTING.T 1
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
  framework_log=$(find "$framework_method_dir" -maxdepth 1 -type f \
    -name '*.txt' -print 2>/dev/null | sort | tail -n 1 || true)

  if [ "$run_status" -ne 0 ]; then
    append_result "$method" "$display_name" "$implementation" NA NA 0 \
      "$train_seconds" "$train_time" failed "$console_log" "$framework_log"
    build_summary
    log "<== [${display_name}] FAILED: exit=${run_status}, time=${train_time}"
    if [ "$continue_on_error" != "1" ]; then
      exit "$run_status"
    fi
    continue
  fi

  acc_summary=$(accuracy_summary "$method" "$console_log")
  if [ -z "$acc_summary" ]; then
    append_result "$method" "$display_name" "$implementation" NA NA 0 \
      "$train_seconds" "$train_time" no_accuracy "$console_log" \
      "$framework_log"
    build_summary
    log "<== [${display_name}] FAILED: no macro-average accuracy found"
    if [ "$continue_on_error" != "1" ]; then
      exit 2
    fi
    continue
  fi

  read -r peak_acc final_acc evaluations <<< "$acc_summary"
  append_result "$method" "$display_name" "$implementation" "$peak_acc" \
    "$final_acc" "$evaluations" "$train_seconds" "$train_time" \
    completed "$console_log" "$framework_log"
  build_summary
  log "<== [${display_name}] peak_avg=${peak_acc}%, final_avg=${final_acc}%, evaluations=${evaluations}, time=${train_time}"
done

build_summary
session_finished_epoch=$(date +%s)
session_seconds=$((session_finished_epoch - session_started_epoch))
session_time=$(format_duration "$session_seconds")

log ""
log "================ FINAL SUMMARY ================"
while IFS=',' read -r method display_name implementation avg_acc metric \
  peak_acc final_acc delta evaluations train_seconds train_time status \
  row_seed console_log framework_log; do
  if [ "$method" = "method" ]; then
    continue
  fi
  log "${display_name}: avg_acc=${avg_acc}% (${metric}), peak=${peak_acc}%, final=${final_acc}%, evaluations=${evaluations}, time=${train_time}, status=${status}"
done < "$summary_csv"
log "session_wall_time=${session_time} (${session_seconds}s)"
log "attempts_csv=${result_csv}"
log "summary_csv=${summary_csv}"
log "master_log=${master_log}"

incomplete_methods=$(awk -F',' \
  'NR > 1 && $12 != "completed" { count += 1 } END { print count + 0 }' \
  "$summary_csv")
if [ "$incomplete_methods" -ne 0 ]; then
  log "ERROR: ${incomplete_methods} selected methods are incomplete."
  exit 3
fi
