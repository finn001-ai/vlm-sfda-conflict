#!/usr/bin/env bash
set -euo pipefail

# Matched target-Clipart preflight for Boundary-Flip DUET.
# RUN_CONTROL=0 skips the Stage14 control when a matched control already exists.

seed="${SEED:-2020}"
run_control="${RUN_CONTROL:-1}"
control_method="temporal_precision_head_control_boundary_flip_seed${seed}"
candidate_method="boundary_flip_duet_preflight_seed${seed}"
result_dir="output/uda/office-home/boundary_flip_duet_preflight_seed${seed}"

declare -a tasks=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

mkdir -p "$result_dir"

for task in "${tasks[@]}"; do
  read -r task_name source_index target_index <<< "$task"
  if [ "$run_control" = "1" ]; then
    echo "==> Matched Stage14 control: ${task_name}, seed=${seed}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/temporal_precision_head.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      SETTING.SEED "$seed" SETTING.S "$source_index" SETTING.T "$target_index" \
      MODEL.METHOD "$control_method"
  fi

  echo "==> Boundary-Flip DUET: ${task_name}, seed=${seed}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/boundary_flip_duet.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.SEED "$seed" SETTING.S "$source_index" SETTING.T "$target_index" \
    MODEL.METHOD "$candidate_method"
done

if [ "$run_control" = "1" ]; then
  python tools/extract_final_accuracy.py \
    --glob "output/uda/office-home/*/${control_method}/*.txt" \
    > "${result_dir}/control_accuracy.csv"
fi

python tools/extract_final_accuracy.py \
  --glob "output/uda/office-home/*/${candidate_method}/*.txt" \
  > "${result_dir}/candidate_accuracy.csv"

analysis_args=(
  --candidate-csv "${result_dir}/candidate_accuracy.csv"
  --control-csv "${result_dir}/control_accuracy.csv"
  --diagnostics-glob
  "output/uda/office-home/*/${candidate_method}/temporal_diagnostics/*_cycle*.npz"
  --output "${result_dir}/gate.json"
)
if [ "$run_control" = "1" ]; then
  analysis_args+=(--require-control)
fi
python tools/analyze_boundary_flip_duet.py "${analysis_args[@]}"

echo "Accuracy: ${result_dir}/candidate_accuracy.csv"
echo "Gate: ${result_dir}/gate.json"
