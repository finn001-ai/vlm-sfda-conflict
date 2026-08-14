#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Two-cycle, eval-only real-conflict GT feature-capacity probe on the locked
# class-proportional VisDA-C proxy25 adaptation subset. Final task accuracy is
# still evaluated on the complete validation set, matching the archive proxy.

seed=2020
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
method="duet_first_cycle_prior_context_transformer_gt_feature_probe_visda_proxy25_seed${seed}"
run_dir="output/uda/VISDA-C/TV/${method}"

for path in \
  data/VISDA-C/train_list.txt \
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

# Reuse the archived proxy25 contract exactly: deterministic SHA-256 ranking
# within every class, ratio=0.25, seed=2020, original row order preserved.
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
  echo "Proxy list is not the deterministic ratio=0.25 seed=2020 subset" >&2
  exit 1
fi

proxy_samples=$(wc -l < "$proxy_list" | tr -d ' ')
full_samples=$(wc -l < data/VISDA-C/validation_list.txt | tr -d ' ')
if [ "$proxy_samples" -ne 13847 ] || [ "$full_samples" -ne 55388 ]; then
  echo "Unexpected VisDA-C size: proxy=${proxy_samples}, full=${full_samples}; expected 13847/55388" >&2
  exit 1
fi
echo "==> Locked proxy25: adaptation=${proxy_samples}/${full_samples}; seed=${seed}"

case "$run_dir" in
  output/uda/VISDA-C/TV/duet_first_cycle_prior_context_transformer_gt_feature_probe_visda_proxy25_seed2020) ;;
  *)
    echo "Refusing to clear unexpected VisDA-C path: $run_dir" >&2
    exit 1
    ;;
esac
rm -rf -- "$run_dir"

echo "==> VisDA-C proxy25 real-conflict 16D GT feature probe, 2 cycles, seed=${seed}"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior_context_transformer.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 2 \
  ACTIVE.ADAPTATION_LIST "$proxy_list"

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected exactly one VisDA-C probe log, found ${#logs[@]}" >&2
  exit 1
fi
if ! grep -q "PLMatch adaptation proxy list: ${proxy_list}; adaptation_samples=${proxy_samples}; full_evaluation_samples=${full_samples}" "${logs[0]}"; then
  echo "VisDA-C run did not use the locked proxy25 adaptation list" >&2
  exit 1
fi
if [ "$(grep -c "DUET real-conflict GT feature probe summary eval-only: cycle=2" "${logs[0]}")" -ne 1 ]; then
  echo "VisDA-C run did not emit exactly one cycle-2 GT feature-probe summary" >&2
  exit 1
fi
if ! grep -q "DUET comparator selection: cycle=2; mode=rank_coverage;.*requested_coverage=20.00%" "${logs[0]}"; then
  echo "VisDA-C probe did not preserve fixed 20% formal admission" >&2
  exit 1
fi
if [ "$(grep -c "Task: TV" "${logs[0]}")" -ne 8 ]; then
  echo "VisDA-C probe did not finish 2 cycles / 8 checkpoints" >&2
  exit 1
fi

echo "==> Probe summary"
grep "DUET real-conflict GT feature probe summary eval-only: cycle=2" "${logs[0]}"
echo "==> Full log: ${logs[0]}"
