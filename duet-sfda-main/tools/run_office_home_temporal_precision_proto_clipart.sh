#!/usr/bin/env bash
set -euo pipefail

# Target prototype classifier-adapter probe.
# Stable temporal pseudo labels build target-domain feature prototypes. Source
# classifier logits are lightly adjusted by prototype logits during label
# estimation, training, and evaluation.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Temporal-precision prototype training: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/temporal_precision_proto.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_temporal_conflict_dynamics.py \
  --glob 'output/uda/office-home/*/temporal_precision_proto/temporal_diagnostics/*_cycle*.npz' \
  --out output/uda/office-home/temporal_precision_proto_dynamics_probe.json

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_proto/*.txt' \
  > output/uda/office-home/temporal_precision_proto_clipart_accuracy.csv
