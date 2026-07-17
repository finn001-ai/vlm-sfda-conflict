#!/usr/bin/env bash
set -euo pipefail

# Mechanism-level A->C trials for class-wise teacher calibration before fusion.

declare -a SETTINGS=(
  "source_prior source_prior"
  "clip_prior clip_prior"
  "both_prior both_prior"
  "mix_prior mix_prior"
)

for setting in "${SETTINGS[@]}"; do
  read -r name calib_mode <<< "$setting"
  echo "==> DCCL calibration trial A->C: ${name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S 0 SETTING.T 1 \
    DCCL.CAND_PAR 0.0 \
    DCCL.CAND_START_CYCLE 0 \
    DCCL.KL_MODE clip \
    DCCL.CALIB_MODE "$calib_mode" \
    DCCL.CALIB_POWER 0.5 \
    DCCL.PL_EXPAND none \
    DCCL.PROMOTE_K 999
done
