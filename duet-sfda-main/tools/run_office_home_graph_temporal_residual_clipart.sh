#!/usr/bin/env bash
set -euo pipefail

# Graph-temporal residual regularization probe.
# Baseline both_prior CLIP/KL teachers are preserved. Graph-temporal evidence
# enters only as a low-weight residual regularizer on stable conflict samples.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Graph-temporal residual training: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/graph_temporal_residual.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_temporal_conflict_dynamics.py \
  --glob 'output/uda/office-home/*/graph_temporal_residual/temporal_diagnostics/*_cycle*.npz' \
  --out output/uda/office-home/graph_temporal_residual_dynamics_probe.json

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/graph_temporal_residual/*.txt' \
  > output/uda/office-home/graph_temporal_residual_clipart_accuracy.csv
