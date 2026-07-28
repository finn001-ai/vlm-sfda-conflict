#!/usr/bin/env bash
set -euo pipefail

# VisDA-C matched test：
#   control   = Stage14 temporal-precision target head
#   candidate = 同一宿主 + Boundary-Flip
# 默认使用完整 validation split 并复用已经完整结束的同名实验。

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

seed="${SEED:-2020}"
run_control="${RUN_CONTROL:-1}"
adaptation_list="${ADAPTATION_LIST:-}"
control_method="temporal_precision_head_boundary_flip_visda_control_seed${seed}"
candidate_method="boundary_flip_duet_visda_seed${seed}"
result_dir="output/uda/VISDA-C/boundary_flip_duet_seed${seed}"

if [ "$run_control" != "0" ] && [ "$run_control" != "1" ]; then
  echo "RUN_CONTROL must be 0 or 1" >&2
  exit 1
fi
case "$seed" in
  ""|*[!0-9]*)
    echo "SEED must be a non-negative integer" >&2
    exit 1
    ;;
esac

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$path" ]; then
    echo "Missing VisDA-C input: $path" >&2
    exit 1
  fi
done

evaluation_list="data/VISDA-C/validation_list.txt"
if [ -n "$adaptation_list" ]; then
  if [ ! -f "$adaptation_list" ]; then
    echo "ADAPTATION_LIST does not exist: $adaptation_list" >&2
    exit 1
  fi
  evaluation_list="$adaptation_list"
fi
expected_samples=$(wc -l < "$evaluation_list" | tr -d ' ')
if [ -z "$adaptation_list" ] && [ "$expected_samples" -ne 55388 ]; then
  echo "Expected 55388 full VisDA validation samples, found ${expected_samples}" >&2
  exit 1
fi

mkdir -p "$result_dir"
sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt \
  > "${result_dir}/source_sha256.txt"
sha256sum "$evaluation_list" \
  > "${result_dir}/adaptation_list_sha256.txt"

adaptation_opts=()
if [ -n "$adaptation_list" ]; then
  adaptation_opts+=(DCCL.ADAPTATION_LIST "$adaptation_list")
fi

reset_incomplete_run() {
  local method=$1
  local reason=$2
  local run_dir
  if [ "$method" != "$control_method" ] && \
    [ "$method" != "$candidate_method" ]; then
    echo "Refusing to reset unexpected method: $method" >&2
    exit 1
  fi
  run_dir="output/uda/VISDA-C/TV/${method}"
  echo "${method}: ${reason}; automatically overwriting incomplete output" >&2
  if [ -e "$run_dir" ]; then
    rm -rf -- "$run_dir"
  fi
  return 1
}

validate_run() {
  local method=$1
  local run_dir
  local pattern
  local path
  local checkpoints
  local refreshes
  local -a logs=()
  run_dir="output/uda/VISDA-C/TV/${method}"
  pattern="output/uda/VISDA-C/TV/${method}/*.txt"
  while IFS= read -r path; do
    logs+=("$path")
  done < <(compgen -G "$pattern" || true)
  if [ "${#logs[@]}" -eq 0 ]; then
    if [ -e "$run_dir" ]; then
      reset_incomplete_run "$method" "output exists but contains no log"
    fi
    return 1
  fi
  if [ "${#logs[@]}" -ne 1 ]; then
    reset_incomplete_run \
      "$method" "expected exactly one log, found ${#logs[@]}"
  fi
  checkpoints=$(grep -c "Task: TV" "${logs[0]}" || true)
  refreshes=$(
    grep -c "Number of valid pseudo-labeled samples" "${logs[0]}" || true
  )
  if [ "$checkpoints" -ne 32 ] || [ "$refreshes" -ne 8 ]; then
    reset_incomplete_run "$method" \
      "incomplete run (${checkpoints}/32 checkpoints, ${refreshes}/8 refreshes)"
  fi
  if ! grep -q "Cycle: 8/8" "${logs[0]}"; then
    reset_incomplete_run "$method" "log is not an eight-cycle run"
  fi
  if ! grep -q \
    "Number of valid pseudo-labeled samples: [0-9]*/${expected_samples}" \
    "${logs[0]}"; then
    reset_incomplete_run \
      "$method" "log does not match ${expected_samples} adaptation samples"
  fi
  return 0
}

control_ready=0
if validate_run "$control_method"; then
  echo "==> Reusing matched Stage14 control, seed=${seed}"
  control_ready=1
elif [ "$run_control" = "1" ]; then
  echo "==> VisDA-C matched Stage14 control, seed=${seed}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/temporal_precision_head.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source SETTING.SEED "$seed" \
    SETTING.S 0 SETTING.T 1 \
    MODEL.METHOD "$control_method" \
    "${adaptation_opts[@]}"
  validate_run "$control_method"
  control_ready=1
fi

if validate_run "$candidate_method"; then
  echo "==> Reusing VisDA-C Boundary-Flip candidate, seed=${seed}"
else
  echo "==> VisDA-C Boundary-Flip DUET, seed=${seed}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/boundary_flip_duet.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source SETTING.SEED "$seed" \
    SETTING.S 0 SETTING.T 1 \
    MODEL.METHOD "$candidate_method" \
    "${adaptation_opts[@]}"
  validate_run "$candidate_method"
fi

if [ "$control_ready" = "1" ]; then
  python tools/extract_final_accuracy.py \
    --glob "output/uda/VISDA-C/TV/${control_method}/*.txt" \
    > "${result_dir}/control_accuracy.csv"
fi

python tools/extract_final_accuracy.py \
  --glob "output/uda/VISDA-C/TV/${candidate_method}/*.txt" \
  > "${result_dir}/candidate_accuracy.csv"

python tools/analyze_visda_boundary_flip_duet.py \
  --candidate-csv "${result_dir}/candidate_accuracy.csv" \
  --control-csv "${result_dir}/control_accuracy.csv" \
  --diagnostics-glob \
    "output/uda/VISDA-C/TV/${candidate_method}/temporal_diagnostics/*_cycle*.npz" \
  --log-glob "output/uda/VISDA-C/TV/${candidate_method}/*.txt" \
  --output "${result_dir}/gate.json"

echo "Candidate accuracy: ${result_dir}/candidate_accuracy.csv"
echo "Gate: ${result_dir}/gate.json"
