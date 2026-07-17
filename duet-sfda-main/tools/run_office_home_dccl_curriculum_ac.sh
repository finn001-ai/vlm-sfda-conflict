#!/usr/bin/env bash
set -euo pipefail

# Mechanism-level A->C trials for late conflict-candidate curriculum.
# The question is whether conflicts should be introduced only after PLMatch
# has stabilized the target representation.

declare -a SETTINGS=(
  "late_candidate_cycle2 0.01 2"
  "late_candidate_cycle1 0.01 1"
  "late_candidate_cycle2_stronger 0.02 2"
)

for setting in "${SETTINGS[@]}"; do
  read -r name cand_par cand_start <<< "$setting"
  echo "==> DCCL curriculum trial A->C: ${name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S 0 SETTING.T 1 \
    DCCL.CAND_PAR "$cand_par" \
    DCCL.CAND_START_CYCLE "$cand_start" \
    DCCL.CAND_TAU 0.0 \
    DCCL.CAND_WEIGHT none \
    DCCL.KL_MODE clip \
    DCCL.KL_CANDIDATE confidence \
    DCCL.TAU_LOW 0.4 \
    DCCL.PROMOTE_K 999
done
