#!/usr/bin/env bash
set -euo pipefail

# Train Office-Home source checkpoints and place them where DUET target
# adaptation/diagnostics expect them:
#   source/uda/office-home/{A,C,P,R}/source_{F,B,C}.pt
#
# Usage:
#   bash tools/train_office_home_sources.sh          # train all sources
#   bash tools/train_office_home_sources.sh 0 1      # train Art and Clipart

SOURCES=("$@")
if [ "${#SOURCES[@]}" -eq 0 ]; then
  SOURCES=(0 1 2 3)
fi

DOMAIN_KEYS=(A C P R)

for s in "${SOURCES[@]}"; do
  if [ "$s" -lt 0 ] || [ "$s" -gt 3 ]; then
    echo "Invalid source index: $s. Expected one of 0 1 2 3." >&2
    exit 1
  fi

  # The source trainer also evaluates target domains after training. T only
  # needs to be different from S for config initialization.
  t=$(( (s + 1) % 4 ))
  key="${DOMAIN_KEYS[$s]}"

  echo "==> Training Office-Home source ${key} (S=${s}, initial T=${t})"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/source.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    SETTING.S "$s" SETTING.T "$t"

  dest="source/uda/office-home/${key}"
  mkdir -p "$dest"
  mv source/source_F.pt "$dest/source_F.pt"
  mv source/source_B.pt "$dest/source_B.pt"
  mv source/source_C.pt "$dest/source_C.pt"
  echo "Saved source checkpoint to ${dest}"
done
