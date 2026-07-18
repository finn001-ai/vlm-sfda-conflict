#!/usr/bin/env bash
set -euo pipefail

# First training gate for topology-prior calibration.
# Run only if tools/run_office_home_topo_prior_probe.sh reports pass_training_gate.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Topology-prior training: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/topo_prior.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done
