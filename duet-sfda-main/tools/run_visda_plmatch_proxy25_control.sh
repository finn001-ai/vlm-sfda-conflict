#!/usr/bin/env bash
set -euo pipefail

method="plmatch_visda_proxy25_seed2020"
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
result_dir="output/uda/VISDA-C"

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
  --seed 2020 \
  --force > /dev/null

if [ ! -f "$proxy_list" ]; then
  echo "==> Creating deterministic VisDA-C 25% proxy list"
  python tools/prepare_visda_proxy_subset.py \
    --input data/VISDA-C/validation_list.txt \
    --output "$proxy_list" \
    --ratio 0.25 \
    --seed 2020
fi
if ! cmp -s "$expected_proxy" "$proxy_list"; then
  echo "Proxy list does not match deterministic ratio=0.25 seed=2020 selection" >&2
  exit 1
fi

log_glob="output/uda/VISDA-C/TV/${method}/*.txt"
if compgen -G "$log_glob" > /dev/null; then
  echo "Existing ${method} log found; refusing to overwrite the control" >&2
  exit 1
fi

mkdir -p "$result_dir"
sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt \
  > "$result_dir/plmatch_visda_proxy25_seed2020_source_sha256.txt"
sha256sum "$proxy_list" \
  > "$result_dir/plmatch_visda_proxy25_seed2020_proxy_sha256.txt"

echo "==> Original PLMatch control: VisDA-C 25% adaptation, full evaluation"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 4 \
  ACTIVE.ADAPTATION_LIST "$proxy_list"

control_logs=(output/uda/VISDA-C/TV/${method}/*.txt)
if [ "${#control_logs[@]}" -ne 1 ] || ! grep -q "Task: TV" "${control_logs[0]}"; then
  echo "PLMatch control produced no VisDA-C accuracy records" >&2
  exit 1
fi
if ! grep -q \
  "PLMatch adaptation proxy list: ${proxy_list}; adaptation_samples=13847; full_evaluation_samples=55388" \
  "${control_logs[0]}"; then
  echo "PLMatch proxy/full loader counts were not verified in the log" >&2
  exit 1
fi
checkpoint_count=$(grep -c "Task: TV" "${control_logs[0]}")
if [ "$checkpoint_count" -ne 16 ]; then
  echo "Expected 16 PLMatch checkpoints, found ${checkpoint_count}" >&2
  exit 1
fi

python tools/extract_final_accuracy.py \
  --glob "$log_glob" \
  --selection peak \
  > "$result_dir/plmatch_visda_proxy25_seed2020_accuracy.csv"

python tools/summarize_visda_temporal_precision_head.py \
  --glob "$log_glob" \
  --out "$result_dir/plmatch_visda_proxy25_seed2020_summary.json" \
  --csv-out "$result_dir/plmatch_visda_proxy25_seed2020_per_class.csv" \
  --class-names data/VISDA-C/classname.txt

python tools/summarize_visda_plmatch_proxy_control.py \
  --glob "$log_glob" \
  --out "$result_dir/plmatch_visda_proxy25_seed2020_control.json"

echo "==> Control complete"
echo "Decision: $result_dir/plmatch_visda_proxy25_seed2020_control.json"
