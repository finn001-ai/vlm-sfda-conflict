#!/usr/bin/env bash
set -euo pipefail

# Baseline smoke experiments matching the DCCL smoke split:
# Art -> {Clipart, Product, RealWorld}.

for t in 1 2 3; do
  echo "==> PLMatch smoke run: S=0, T=${t}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/plmatch.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S 0 SETTING.T "$t"
done
