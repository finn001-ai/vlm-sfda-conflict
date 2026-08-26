#!/usr/bin/env bash
set -euo pipefail

# Usage: bash tools/run_officehome_duet_anchored_consensus.sh [target_index] [seed]
# Office-Home domain indices: 0=Art, 1=Clipart, 2=Product, 3=RealWorld.
# The source is fixed to Art (index 0) for the first controlled experiment.
target_index="${1:-2}"
experiment_seed="${2:-2020}"

case "$target_index" in
  1) transfer_name="AC"; target_list="Clipart_list.txt" ;;
  2) transfer_name="AP"; target_list="Product_list.txt" ;;
  3) transfer_name="AR"; target_list="RealWorld_list.txt" ;;
  *)
    echo "target_index must be 1, 2, or 3 for an Art-source transfer" >&2
    exit 1
    ;;
esac

for required_path in \
  "data/office-home/Art_list.txt" \
  "data/office-home/${target_list}" \
  "data/office-home/classname.txt" \
  "source/uda/office-home/A/source_F.pt" \
  "source/uda/office-home/A/source_B.pt" \
  "source/uda/office-home/A/source_C.pt"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing Office-Home input: $required_path" >&2
    exit 1
  fi
done

method_name="duet_anchored_consensus_${transfer_name}_seed${experiment_seed}"
echo "==> Office-Home ${transfer_name}; seed=${experiment_seed}"
echo "==> Full-distribution anchor starts before epoch 1; old DUET Cycle-1 checkpoints are intentionally not reused"

python image_target_of_oh_vs.py \
  --cfg cfgs/office-home/duet_anchored_consensus.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method_name" \
  SETTING.S 0 SETTING.T "$target_index" SETTING.SEED "$experiment_seed"

run_dir="output/uda/office-home/${transfer_name}/${method_name}"
echo "==> Result directory: ${run_dir}"
latest_log=$(find "$run_dir" -maxdepth 1 -type f -name '*.txt' -print | sort | tail -n 1)
if [ -n "$latest_log" ]; then
  echo "==> Final trajectory"
  grep "DUET anchored consensus epoch:" "$latest_log" | tail -n 5
  echo "==> Full log: ${latest_log}"
fi
