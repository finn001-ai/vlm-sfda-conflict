#!/usr/bin/env bash
set -euo pipefail

# Fast validation for unsupervised per-cycle calibration selection on Art source.

for t in 1 2 3; do
  echo "==> DCCL auto-calib Art-source run: S=0, T=${t}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S 0 SETTING.T "$t" \
    DCCL.CAND_PAR 0.0 \
    DCCL.CAND_START_CYCLE 0 \
    DCCL.KL_MODE clip \
    DCCL.CALIB_MODE auto_agree \
    DCCL.CALIB_POWER 0.5 \
    DCCL.CALIB_AUTO_LAMBDA 0.2 \
    DCCL.PL_EXPAND none \
    DCCL.PROMOTE_K 999
done
