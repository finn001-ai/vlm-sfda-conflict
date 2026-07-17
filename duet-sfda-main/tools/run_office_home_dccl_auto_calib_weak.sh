#!/usr/bin/env bash
set -euo pipefail

# Validate whether auto calibration avoids the weak global clip_prior behavior
# on Product/RealWorld source tasks.

declare -a TASKS=(
  "2 0"
  "2 1"
  "2 3"
  "3 0"
  "3 1"
  "3 2"
)

for task in "${TASKS[@]}"; do
  read -r s t <<< "$task"
  echo "==> DCCL auto-calib weak-source run: S=${s}, T=${t}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t" \
    DCCL.CAND_PAR 0.0 \
    DCCL.CAND_START_CYCLE 0 \
    DCCL.KL_MODE clip \
    DCCL.CALIB_MODE auto_agree \
    DCCL.CALIB_POWER 0.5 \
    DCCL.CALIB_AUTO_LAMBDA 0.2 \
    DCCL.PL_EXPAND none \
    DCCL.PROMOTE_K 999
done
