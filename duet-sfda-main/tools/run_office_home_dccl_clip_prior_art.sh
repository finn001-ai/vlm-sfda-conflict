#!/usr/bin/env bash
set -euo pipefail

# Run the best current calibration scheme on Art -> {Product, RealWorld}.
# A->C was already measured at 72.78 with this setting.

for t in 2 3; do
  echo "==> DCCL clip-prior Art-source run: S=0, T=${t}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S 0 SETTING.T "$t" \
    DCCL.CAND_PAR 0.0 \
    DCCL.CAND_START_CYCLE 0 \
    DCCL.KL_MODE clip \
    DCCL.CALIB_MODE clip_prior \
    DCCL.CALIB_POWER 0.5 \
    DCCL.PL_EXPAND none \
    DCCL.PROMOTE_K 999
done
