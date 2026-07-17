#!/usr/bin/env bash
set -euo pipefail

# Mechanism probe for weak Product/RealWorld source tasks.
# This is not a final method. It measures which class-wise calibration mode is
# actually preferred by each weak transfer before designing a selector.

declare -a TASKS=(
  "PA 2 0"
  "PC 2 1"
  "PR 2 3"
  "RA 3 0"
  "RC 3 1"
  "RP 3 2"
)

declare -a SETTINGS=(
  "none none"
  "source_prior source_prior"
  "clip_prior clip_prior"
  "both_prior both_prior"
  "mix_prior mix_prior"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  for setting in "${SETTINGS[@]}"; do
    read -r name calib_mode <<< "$setting"
    echo "==> DCCL weak calibration probe ${task_name}: ${name}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/dccl.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      SETTING.S "$s" SETTING.T "$t" \
      DCCL.CAND_PAR 0.0 \
      DCCL.CAND_START_CYCLE 0 \
      DCCL.KL_MODE clip \
      DCCL.CALIB_MODE "$calib_mode" \
      DCCL.CALIB_POWER 0.5 \
      DCCL.PL_EXPAND none \
      DCCL.PROMOTE_K 999
  done
done
