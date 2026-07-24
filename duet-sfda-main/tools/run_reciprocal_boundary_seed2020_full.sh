#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

preflight_gate="output/uda/reciprocal_boundary_preflight_gate.json"
if [ ! -f "$preflight_gate" ] || ! grep -q \
  '"decision": "pass_reciprocal_boundary_preflight"' "$preflight_gate"; then
  echo "Reciprocal-boundary joint preflight has not passed: $preflight_gate" >&2
  echo "Run first: bash tools/run_reciprocal_boundary_preflight.sh" >&2
  exit 1
fi

visda_control="plmatch_reciprocal_boundary_full_control_seed2020"
visda_candidate="reciprocal_boundary_full_seed2020"
office_control="plmatch_reciprocal_boundary_control_seed2020"
office_candidate="reciprocal_boundary_seed2020"

validate_visda_run() {
  method=$1
  pattern="output/uda/VISDA-C/TV/${method}/*.txt"
  logs=()
  while IFS= read -r path; do
    logs+=("$path")
  done < <(compgen -G "$pattern" || true)
  if [ "${#logs[@]}" -eq 0 ]; then
    return 1
  fi
  if [ "${#logs[@]}" -ne 1 ]; then
    echo "${method}: expected one VisDA log, found ${#logs[@]}" >&2
    exit 1
  fi
  checkpoints=$(grep -c "Task: TV" "${logs[0]}" || true)
  if [ "$checkpoints" -ne 32 ]; then
    echo "${method}: incomplete VisDA run (${checkpoints}/32)" >&2
    exit 1
  fi
  return 0
}

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

if validate_visda_run "$visda_control"; then
  echo "==> Reusing full-data official-DUET VisDA control"
else
  echo "==> Running full-data official-DUET VisDA control"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/plmatch.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$visda_control" \
    SETTING.SEED 2020 SETTING.S 0 SETTING.T 1
  validate_visda_run "$visda_control"
fi

if validate_visda_run "$visda_candidate"; then
  echo "==> Reusing full-data reciprocal-boundary VisDA candidate"
else
  echo "==> Running full-data reciprocal-boundary VisDA candidate"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/reciprocal_boundary.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$visda_candidate" \
    SETTING.SEED 2020 SETTING.S 0 SETTING.T 1
  validate_visda_run "$visda_candidate"
fi

python tools/summarize_reciprocal_boundary_preflight.py visda-full \
  --control-glob "output/uda/VISDA-C/TV/${visda_control}/*.txt" \
  --candidate-glob "output/uda/VISDA-C/TV/${visda_candidate}/*.txt" \
  --out output/uda/VISDA-C/reciprocal_boundary_full_seed2020_gate.json

declare -a tasks=(
  "AC 0 1"
  "AP 0 2"
  "AR 0 3"
  "CA 1 0"
  "CP 1 2"
  "CR 1 3"
  "PA 2 0"
  "PC 2 1"
  "PR 2 3"
  "RA 3 0"
  "RC 3 1"
  "RP 3 2"
)

for list_name in Art Clipart Product RealWorld; do
  if [ ! -f "data/office-home/${list_name}_list.txt" ]; then
    echo "Missing Office-Home list: data/office-home/${list_name}_list.txt" >&2
    exit 1
  fi
done
for source_name in A C P R; do
  for component in F B C; do
    path="source/uda/office-home/${source_name}/source_${component}.pt"
    if [ ! -f "$path" ]; then
      echo "Missing Office-Home source checkpoint: $path" >&2
      exit 1
    fi
  done
done

validate_office_run() {
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
  if validate_office_run "$office_control" "$task_name"; then
    echo "==> Reusing official-DUET Office-Home ${task_name}"
  else
    echo "==> Running official-DUET Office-Home ${task_name}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/plmatch.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$office_control" \
      SETTING.SEED 2020 \
      SETTING.S "$source_index" SETTING.T "$target_index"
    validate_office_run "$office_control" "$task_name"
  fi

  if validate_office_run "$office_candidate" "$task_name"; then
    echo "==> Reusing reciprocal-boundary Office-Home ${task_name}"
  else
    echo "==> Running reciprocal-boundary Office-Home ${task_name}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/reciprocal_boundary.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$office_candidate" \
      SETTING.SEED 2020 \
      SETTING.S "$source_index" SETTING.T "$target_index"
    validate_office_run "$office_candidate" "$task_name"
  fi
done

python tools/summarize_reciprocal_boundary_preflight.py office-home-full \
  --control-glob "output/uda/office-home/*/${office_control}/*.txt" \
  --candidate-glob "output/uda/office-home/*/${office_candidate}/*.txt" \
  --out output/uda/office-home/reciprocal_boundary_full_seed2020_gate.json \
  --csv-out output/uda/office-home/reciprocal_boundary_full_seed2020_results.csv

python tools/summarize_reciprocal_boundary_preflight.py joint-full \
  --visda output/uda/VISDA-C/reciprocal_boundary_full_seed2020_gate.json \
  --office-home output/uda/office-home/reciprocal_boundary_full_seed2020_gate.json \
  --out output/uda/reciprocal_boundary_full_seed2020_gate.json

if grep -q \
  '"decision": "pass_reciprocal_boundary_seed2020_gate"' \
  output/uda/reciprocal_boundary_full_seed2020_gate.json; then
  echo "==> Seed-2020 full gate passed. Freeze hyperparameters before seed sweep."
else
  echo "==> Seed-2020 full gate failed. Do not start a seed sweep." >&2
  exit 2
fi
