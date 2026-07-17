#!/usr/bin/env bash
set -euo pipefail

# Run the DUET/PLMatch baseline on all 12 Office-Home transfer tasks.

for s in 0 1 2 3; do
  for t in 0 1 2 3; do
    if [ "$s" -eq "$t" ]; then
      continue
    fi
    echo "==> PLMatch full run: S=${s}, T=${t}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/plmatch.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      SETTING.S "$s" SETTING.T "$t"
  done
done
