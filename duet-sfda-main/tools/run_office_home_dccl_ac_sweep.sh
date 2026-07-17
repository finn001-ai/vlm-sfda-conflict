#!/usr/bin/env bash
set -euo pipefail

# A->C sweep for conservative conflict-candidate learning.
# Run this before spending time on all 12 tasks.

declare -a SETTINGS=(
  "0.005 0.0 none 0.4"
  "0.010 0.0 none 0.4"
  "0.020 0.0 none 0.4"
  "0.010 0.3 none 0.4"
  "0.010 0.4 none 0.4"
  "0.010 0.3 mass 0.4"
  "0.010 0.3 ramp 0.4"
  "0.010 0.4 mass 0.4"
  "0.010 0.4 ramp 0.4"
  "0.020 0.3 ramp 0.4"
)

for setting in "${SETTINGS[@]}"; do
  read -r cand_par cand_tau cand_weight tau_low <<< "$setting"
  echo "==> DCCL A->C sweep: CAND_PAR=${cand_par}, CAND_TAU=${cand_tau}, CAND_WEIGHT=${cand_weight}, TAU_LOW=${tau_low}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S 0 SETTING.T 1 \
    DCCL.CAND_PAR "$cand_par" \
    DCCL.CAND_TAU "$cand_tau" \
    DCCL.CAND_WEIGHT "$cand_weight" \
    DCCL.TAU_LOW "$tau_low" \
    DCCL.PROMOTE_K 999
done
