#!/usr/bin/env bash
set -euo pipefail

# Asymmetric conflict correction on A->C: only graph-supported source rescue
# becomes a hard pseudo-label. Graph-supported CLIP conflicts keep DUET's KL.

python image_target_of_oh_vs.py \
  --cfg cfgs/office-home/accd_source_rescue.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  SETTING.S 0 SETTING.T 1

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/AC/accd_sr/*.txt'
