#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# VisDA-C anchored-consensus experiment on the locked class-proportional 25%
# adaptation subset. Accuracy is always evaluated on all 55,388 target images.
experiment_seed="${1:-2020}"
proxy_list="data/VISDA-C/validation_proxy25_seed${experiment_seed}_list.txt"
method_name="duet_anchored_consensus_visda_proxy25_seed${experiment_seed}"
run_dir="output/uda/VISDA-C/TV/${method_name}"

for required_path in \
  data/VISDA-C/train_list.txt \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$required_path" ]; then
    echo "Missing VisDA-C input: $required_path" >&2
    exit 1
  fi
done

expected_proxy=$(mktemp)
trap 'rm -f "$expected_proxy"' EXIT
python tools/prepare_visda_proxy_subset.py \
  --input data/VISDA-C/validation_list.txt \
  --output "$expected_proxy" \
  --ratio 0.25 \
  --seed "$experiment_seed" \
  --force > /dev/null

if [ ! -f "$proxy_list" ]; then
  python tools/prepare_visda_proxy_subset.py \
    --input data/VISDA-C/validation_list.txt \
    --output "$proxy_list" \
    --ratio 0.25 \
    --seed "$experiment_seed"
fi
if ! cmp -s "$expected_proxy" "$proxy_list"; then
  echo "Proxy list does not match deterministic ratio=0.25 seed=${experiment_seed}" >&2
  exit 1
fi

proxy_samples=$(wc -l < "$proxy_list" | tr -d ' ')
full_samples=$(wc -l < data/VISDA-C/validation_list.txt | tr -d ' ')
if [ "$full_samples" -ne 55388 ]; then
  echo "Unexpected VisDA-C full target size: ${full_samples}; expected 55388" >&2
  exit 1
fi
echo "==> VisDA-C proxy25: adaptation=${proxy_samples}/${full_samples}; seed=${experiment_seed}"
echo "==> New consensus starts before epoch 1; old DUET Cycle-1 checkpoints are intentionally not reused"

python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_anchored_consensus.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method_name" \
  SETTING.S 0 SETTING.T 1 SETTING.SEED "$experiment_seed" \
  ACTIVE.ADAPTATION_LIST "$proxy_list"

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -lt 1 ]; then
  echo "No VisDA-C log found in ${run_dir}" >&2
  exit 1
fi
latest_log=$(printf '%s\n' "${logs[@]}" | sort | tail -n 1)
if ! grep -q "adaptation_samples=${proxy_samples}; full_evaluation_samples=${full_samples}" "$latest_log"; then
  echo "Run did not use proxy25 adaptation with full-target evaluation" >&2
  exit 1
fi
if ! grep -q "soft_coverage=100.00%; conflict_hard_coverage=100.00%" "$latest_log"; then
  echo "Run did not execute the full-coverage consensus path" >&2
  exit 1
fi

echo "==> Final five epochs"
grep "DUET anchored consensus epoch:" "$latest_log" | tail -n 5
echo "==> Full log: ${latest_log}"
