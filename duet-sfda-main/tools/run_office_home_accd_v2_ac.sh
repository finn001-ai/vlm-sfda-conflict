#!/usr/bin/env bash
set -euo pipefail

# Mechanism test: frozen initial anchors plus reversible conflict promotion.
# Run A->C first because v1 is only 0.29 points below the DUET paper result.

python image_target_of_oh_vs.py \
  --cfg cfgs/office-home/accd_v2.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  SETTING.S 0 SETTING.T 1

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/AC/accd_v2/*.txt'
