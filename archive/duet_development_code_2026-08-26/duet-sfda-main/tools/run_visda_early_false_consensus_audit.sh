#!/usr/bin/env bash
set -euo pipefail

seed=2020
method="plmatch_visda_early_false_consensus_proxy25_seed${seed}"
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
output_dir="output/uda/VISDA-C/TV/${method}"
snapshot_dir="${output_dir}/early_false_consensus"
report_dir="${output_dir}/audit_report"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$path" ]; then
    echo "Missing VisDA-C input: $path" >&2
    exit 1
  fi
done

expected_proxy=$(mktemp)
trap 'rm -f "$expected_proxy"' EXIT
python tools/prepare_visda_proxy_subset.py \
  --input data/VISDA-C/validation_list.txt \
  --output "$expected_proxy" \
  --ratio 0.25 \
  --seed "$seed" \
  --force > /dev/null

if [ ! -f "$proxy_list" ]; then
  python tools/prepare_visda_proxy_subset.py \
    --input data/VISDA-C/validation_list.txt \
    --output "$proxy_list" \
    --ratio 0.25 \
    --seed "$seed"
fi
if ! cmp -s "$expected_proxy" "$proxy_list"; then
  echo "Proxy list does not match ratio=0.25 seed=${seed}" >&2
  exit 1
fi

# This script owns exactly this diagnostic directory and intentionally
# overwrites an incomplete or previous audit run.
case "$output_dir" in
  output/uda/VISDA-C/TV/plmatch_visda_early_false_consensus_proxy25_seed2020) ;;
  *)
    echo "Refusing to overwrite unexpected path: $output_dir" >&2
    exit 1
    ;;
esac
rm -rf -- "$output_dir"

echo "==> Running original DUET for three cycles on the fixed VisDA 25% proxy"
echo "==> No model checkpoints will be saved; only compressed audit snapshots"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  TEST.INTERVAL 1 \
  ACTIVE.CYCLE 3 \
  ACTIVE.ADAPTATION_LIST "$proxy_list" \
  FAILURE_AUDIT.ENABLED True \
  FAILURE_AUDIT.DIR early_false_consensus \
  FAILURE_AUDIT.FEATURE_DTYPE float16

for snapshot in pre_cycle01.npz pre_cycle02.npz pre_cycle03.npz final_full.npz; do
  if [ ! -f "${snapshot_dir}/${snapshot}" ]; then
    echo "Missing expected audit snapshot: ${snapshot_dir}/${snapshot}" >&2
    exit 1
  fi
done

mkdir -p "$report_dir"
python tools/analyze_early_false_consensus.py \
  --early "${snapshot_dir}/pre_cycle01.npz" \
  --late "${snapshot_dir}/pre_cycle03.npz" \
  --data-list "$proxy_list" \
  --class-names data/VISDA-C/classname.txt \
  --out-json "${report_dir}/early_false_consensus.json" \
  --out-classes "${report_dir}/early_false_consensus_by_class.csv" \
  --out-samples "${report_dir}/early_false_consensus_samples.csv"

echo "==> Audit complete"
echo "Main report: ${report_dir}/early_false_consensus.json"
echo "Class report: ${report_dir}/early_false_consensus_by_class.csv"
echo "Sample report: ${report_dir}/early_false_consensus_samples.csv"
