#!/usr/bin/env bash
set -euo pipefail

# No-adaptation probe for continuous fusion of both_prior teacher and graph
# diffusion posterior. This is not a per-sample hard graph rule.

declare -a TASKS=(
  "AC 0 1"
  "PC 2 1"
  "RC 3 1"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> Export graph-teacher fusion probe features: ${task_name}"
  python tools/export_conflict_diagnostics.py \
    --cfg cfgs/office-home/plmatch.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"
done

python tools/analyze_graph_teacher_fusion.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*C_conflicts.npz' \
  --out output/uda/office-home/graph_teacher_fusion_probe.json
