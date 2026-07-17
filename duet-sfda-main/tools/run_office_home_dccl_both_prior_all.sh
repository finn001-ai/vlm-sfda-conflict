#!/usr/bin/env bash
set -euo pipefail

# Full Office-Home validation for dual class-wise prior calibration.
# Weak-source probes indicate that calibrating both the source/task branch and
# CLIP branch is more stable than auto_agree on Product/RealWorld sources.

declare -a TASKS=(
  "AC 0 1"
  "AP 0 2"
  "AR 0 3"
  "CA 1 0"
  "CP 1 2"
  "CR 1 3"
  "PA 2 0"
  "PC 2 1"
  "PR 2 3"
  "RA 3 0"
  "RC 3 1"
  "RP 3 2"
)

for task in "${TASKS[@]}"; do
  read -r task_name s t <<< "$task"
  echo "==> DCCL both-prior full Office-Home run: ${task_name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t" \
    DCCL.CAND_PAR 0.0 \
    DCCL.CAND_START_CYCLE 0 \
    DCCL.KL_MODE clip \
    DCCL.CALIB_MODE both_prior \
    DCCL.CALIB_POWER 0.5 \
    DCCL.PL_EXPAND none \
    DCCL.PROMOTE_K 999
done
