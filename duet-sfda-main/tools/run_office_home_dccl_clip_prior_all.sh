#!/usr/bin/env bash
set -euo pipefail

# Run the best current class-wise calibration scheme on all 12 Office-Home tasks.

for s in 0 1 2 3; do
  for t in 0 1 2 3; do
    if [ "$s" -eq "$t" ]; then
      continue
    fi
    echo "==> DCCL clip-prior full run: S=${s}, T=${t}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/dccl.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      SETTING.S "$s" SETTING.T "$t" \
      DCCL.CAND_PAR 0.0 \
      DCCL.CAND_START_CYCLE 0 \
      DCCL.KL_MODE clip \
      DCCL.CALIB_MODE clip_prior \
      DCCL.CALIB_POWER 0.5 \
      DCCL.PL_EXPAND none \
      DCCL.PROMOTE_K 999
  done
done
