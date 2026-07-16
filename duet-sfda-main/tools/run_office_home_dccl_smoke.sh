#!/usr/bin/env bash
set -euo pipefail

# First DCCL smoke experiments: Art -> {Clipart, Product, RealWorld}.
# Run this before launching all 12 Office-Home transfers.

for t in 1 2 3; do
  echo "==> DCCL smoke run: S=0, T=${t}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S 0 SETTING.T "$t"
done
