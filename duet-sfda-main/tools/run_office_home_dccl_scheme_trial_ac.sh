#!/usr/bin/env bash
set -euo pipefail

# Mechanism-level A->C trials. These are intentionally not a parameter sweep.

declare -a SETTINGS=(
  "conflict_kl_off 0.0 non_conflict confidence"
  "conflict_candidate_kl 0.0 candidate confidence"
  "candidate_kl_plus_loss 0.01 candidate confidence"
  "candidate_kl_balanced 0.0 candidate balanced"
)

for setting in "${SETTINGS[@]}"; do
  read -r name cand_par kl_mode kl_candidate <<< "$setting"
  echo "==> DCCL scheme trial A->C: ${name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S 0 SETTING.T 1 \
    DCCL.CAND_PAR "$cand_par" \
    DCCL.CAND_TAU 0.0 \
    DCCL.CAND_WEIGHT none \
    DCCL.KL_MODE "$kl_mode" \
    DCCL.KL_CANDIDATE "$kl_candidate" \
    DCCL.TAU_LOW 0.4 \
    DCCL.PROMOTE_K 999
done
