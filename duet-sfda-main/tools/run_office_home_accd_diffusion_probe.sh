#!/usr/bin/env bash
set -euo pipefail

# Fast, no-adaptation probe for the target-Clipart bottleneck tasks.
# Ground-truth labels are used only by analyze_accd_diffusion.py for reporting.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Export ACCD features: ${task_name}"
  python tools/export_conflict_diagnostics.py \
    --cfg cfgs/office-home/plmatch.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_accd_diffusion.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*C_conflicts.npz'
