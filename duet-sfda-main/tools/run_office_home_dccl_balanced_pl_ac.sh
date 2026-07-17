#!/usr/bin/env bash
set -euo pipefail

# Mechanism-level A->C trials for class-balanced pseudo-label expansion.
# This changes which samples enter hard pseudo-label training, rather than
# adding another conflict loss.

declare -a SETTINGS=(
  "balanced_pl_top30 30 0.0"
  "balanced_pl_top45 45 0.0"
  "balanced_pl_top45_conf02 45 0.2"
)

for setting in "${SETTINGS[@]}"; do
  read -r name topk min_conf <<< "$setting"
  echo "==> DCCL balanced PL trial A->C: ${name}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S 0 SETTING.T 1 \
    DCCL.CAND_PAR 0.0 \
    DCCL.CAND_START_CYCLE 0 \
    DCCL.KL_MODE clip \
    DCCL.PL_EXPAND balanced_topk \
    DCCL.PL_TOPK_PER_CLASS "$topk" \
    DCCL.PL_MIN_CONF "$min_conf" \
    DCCL.PROMOTE_K 999
done
