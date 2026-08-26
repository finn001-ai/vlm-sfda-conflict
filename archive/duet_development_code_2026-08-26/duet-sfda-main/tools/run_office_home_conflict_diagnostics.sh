#!/usr/bin/env bash
set -euo pipefail

# Run source-vs-CLIP conflict diagnostics for Office-Home.
#
# Usage:
#   bash tools/run_office_home_conflict_diagnostics.sh        # all 12 pairs
#   bash tools/run_office_home_conflict_diagnostics.sh 0      # Art -> others

SOURCES=("$@")
if [ "${#SOURCES[@]}" -eq 0 ]; then
  SOURCES=(0 1 2 3)
fi

for s in "${SOURCES[@]}"; do
  for t in 0 1 2 3; do
    if [ "$s" -eq "$t" ]; then
      continue
    fi
    echo "==> Diagnostics S=${s}, T=${t}"
    python tools/export_conflict_diagnostics.py \
      --cfg cfgs/office-home/plmatch.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      SETTING.S "$s" SETTING.T "$t"
  done
done
