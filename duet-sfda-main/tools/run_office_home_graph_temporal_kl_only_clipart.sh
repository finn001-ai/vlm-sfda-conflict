#!/usr/bin/env bash
set -euo pipefail

# Weak graph-temporal injection probe.
# Graph-fused teacher is used only for the task-model KL target; CLIP visual
# update keeps the original both_prior mixed teacher.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Graph-temporal KL-only training: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/graph_temporal_kl_only.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_temporal_conflict_dynamics.py \
  --glob 'output/uda/office-home/*/graph_temporal_kl_only/temporal_diagnostics/*_cycle*.npz' \
  --out output/uda/office-home/graph_temporal_kl_only_dynamics_probe.json

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/graph_temporal_kl_only/*.txt' \
  > output/uda/office-home/graph_temporal_kl_only_clipart_accuracy.csv
