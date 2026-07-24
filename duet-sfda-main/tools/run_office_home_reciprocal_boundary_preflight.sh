#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

control_method="plmatch_reciprocal_boundary_control_seed2020"
host_method="reciprocal_boundary_host_control_seed2020"
candidate_method="reciprocal_boundary_seed2020"
result_dir="output/uda/office-home"
declare -a tasks=("AC 0 1" "PC 2 1" "RC 3 1")

for list_name in Art Clipart Product RealWorld; do
  if [ ! -f "data/office-home/${list_name}_list.txt" ]; then
    echo "Missing Office-Home list: data/office-home/${list_name}_list.txt" >&2
    exit 1
  fi
done
if [ ! -f data/office-home/classname.txt ]; then
  echo "Missing Office-Home class names: data/office-home/classname.txt" >&2
  exit 1
fi
for source_name in A P R; do
  for component in F B C; do
    path="source/uda/office-home/${source_name}/source_${component}.pt"
    if [ ! -f "$path" ]; then
      echo "Missing Office-Home source checkpoint: $path" >&2
      exit 1
    fi
  done
done

validate_run() {
  method=$1
  task_name=$2
  pattern="output/uda/office-home/${task_name}/${method}/*.txt"
  logs=()
  while IFS= read -r path; do
    logs+=("$path")
  done < <(compgen -G "$pattern" || true)
  if [ "${#logs[@]}" -eq 0 ]; then
    return 1
  fi
  if [ "${#logs[@]}" -ne 1 ]; then
    echo "${method}/${task_name}: expected one log, found ${#logs[@]}" >&2
    exit 1
  fi
  checkpoints=$(grep -c "Task: ${task_name}" "${logs[0]}" || true)
  if [ "$checkpoints" -ne 16 ]; then
    echo "${method}/${task_name}: incomplete run (${checkpoints}/16)" >&2
    exit 1
  fi
  return 0
}

for task in "${tasks[@]}"; do
  read -r task_name source_index target_index <<< "$task"
  if validate_run "$control_method" "$task_name"; then
    echo "==> Reusing official-DUET control ${task_name}"
  else
    echo "==> Running official-DUET control ${task_name}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/plmatch.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$control_method" \
      SETTING.SEED 2020 \
      SETTING.S "$source_index" SETTING.T "$target_index"
    validate_run "$control_method" "$task_name"
  fi

  if validate_run "$candidate_method" "$task_name"; then
    echo "==> Reusing reciprocal-boundary candidate ${task_name}"
  else
    echo "==> Running reciprocal-boundary candidate ${task_name}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/reciprocal_boundary.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$candidate_method" \
      SETTING.SEED 2020 \
      SETTING.S "$source_index" SETTING.T "$target_index"
    validate_run "$candidate_method" "$task_name"
  fi

  if validate_run "$host_method" "$task_name"; then
    echo "==> Reusing boundary-disabled host control ${task_name}"
  else
    echo "==> Running boundary-disabled host control ${task_name}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/reciprocal_boundary.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$host_method" \
      SETTING.SEED 2020 \
      SETTING.S "$source_index" SETTING.T "$target_index" \
      DCCL.RECIPROCAL_BOUNDARY False
    validate_run "$host_method" "$task_name"
  fi
done

mkdir -p "$result_dir"
sha256sum source/uda/office-home/{A,P,R}/source_{F,B,C}.pt \
  > "$result_dir/reciprocal_boundary_preflight_source_sha256.txt"

python tools/summarize_reciprocal_boundary_preflight.py office-home \
  --control-glob "output/uda/office-home/*/${control_method}/*.txt" \
  --host-glob "output/uda/office-home/*/${host_method}/*.txt" \
  --candidate-glob "output/uda/office-home/*/${candidate_method}/*.txt" \
  --out "$result_dir/reciprocal_boundary_preflight_gate.json" \
  --csv-out "$result_dir/reciprocal_boundary_preflight_results.csv"

echo "==> Office-Home reciprocal-boundary preflight complete"
echo "Gate: $result_dir/reciprocal_boundary_preflight_gate.json"
echo "Results: $result_dir/reciprocal_boundary_preflight_results.csv"
