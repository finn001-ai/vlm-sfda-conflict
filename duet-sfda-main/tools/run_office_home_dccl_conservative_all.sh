#!/usr/bin/env bash
set -euo pipefail

# Run conservative DCCL on all 12 Office-Home transfer tasks.
# Arguments: CAND_PAR CAND_TAU CAND_WEIGHT TAU_LOW

CAND_PAR="${1:-0.01}"
CAND_TAU="${2:-0.0}"
CAND_WEIGHT="${3:-none}"
TAU_LOW="${4:-0.4}"

for s in 0 1 2 3; do
  for t in 0 1 2 3; do
    if [ "$s" -eq "$t" ]; then
      continue
    fi
    echo "==> DCCL conservative full: S=${s}, T=${t}, CAND_PAR=${CAND_PAR}, CAND_TAU=${CAND_TAU}, CAND_WEIGHT=${CAND_WEIGHT}, TAU_LOW=${TAU_LOW}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/dccl.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      SETTING.S "$s" SETTING.T "$t" \
      DCCL.CAND_PAR "$CAND_PAR" \
      DCCL.CAND_TAU "$CAND_TAU" \
      DCCL.CAND_WEIGHT "$CAND_WEIGHT" \
      DCCL.TAU_LOW "$TAU_LOW" \
      DCCL.PROMOTE_K 999
  done
done
