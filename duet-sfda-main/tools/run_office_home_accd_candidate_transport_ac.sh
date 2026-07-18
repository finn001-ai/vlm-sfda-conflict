#!/usr/bin/env bash
set -euo pipefail

# Preserve CLIP's non-candidate distribution and candidate mass, then use
# stable dual-graph evidence only to redistribute source/CLIP candidate mass.

python image_target_of_oh_vs.py \
  --cfg cfgs/office-home/accd_candidate_transport.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  SETTING.S 0 SETTING.T 1

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/AC/accd_ct/*.txt'
