#!/usr/bin/env bash
set -euo pipefail

# Isolate anchor memory from resolution memory on A->C.
# All numerical hyperparameters are identical to ACCD v1/v2.

python image_target_of_oh_vs.py \
  --cfg cfgs/office-home/accd_frozen_persistent.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  SETTING.S 0 SETTING.T 1

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/AC/accd_fp/*.txt'
