#!/usr/bin/env bash
set -euo pipefail

# Conservative DCCL smoke test: low-weight candidate-set learning with
# promotion effectively disabled. Defaults match the best setting seen so far.

CAND_PAR="${1:-0.01}"
CAND_TAU="${2:-0.0}"
CAND_WEIGHT="${3:-none}"
TAU_LOW="${4:-0.4}"

for t in 1 2 3; do
  echo "==> DCCL conservative smoke: S=0, T=${t}, CAND_PAR=${CAND_PAR}, CAND_TAU=${CAND_TAU}, CAND_WEIGHT=${CAND_WEIGHT}, TAU_LOW=${TAU_LOW}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dccl.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S 0 SETTING.T "$t" \
    DCCL.CAND_PAR "$CAND_PAR" \
    DCCL.CAND_TAU "$CAND_TAU" \
    DCCL.CAND_WEIGHT "$CAND_WEIGHT" \
    DCCL.TAU_LOW "$TAU_LOW" \
    DCCL.PROMOTE_K 999
done
