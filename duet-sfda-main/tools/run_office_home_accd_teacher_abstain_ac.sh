#!/usr/bin/env bash
set -euo pipefail

# Topology-gated teacher abstention on A->C: stable graph-supported source
# conflicts suppress CLIP KL without becoming hard source pseudo-labels.

python image_target_of_oh_vs.py \
  --cfg cfgs/office-home/accd_teacher_abstain.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  SETTING.S 0 SETTING.T 1

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/AC/accd_ta/*.txt'
