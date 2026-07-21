#!/usr/bin/env bash
set -euo pipefail

baseline_method="temporal_precision_head_seed2020_visda"
method="temporal_precision_head_mix040_preflight_visda"
result_dir="output/uda/VISDA-C"

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

baseline_glob="output/uda/VISDA-C/TV/${baseline_method}/*.txt"
if ! compgen -G "$baseline_glob" > /dev/null; then
  echo "Missing completed Stage14 VisDA baseline log: $baseline_glob" >&2
  exit 1
fi
if compgen -G "output/uda/VISDA-C/TV/${method}/*.txt" > /dev/null; then
  echo "Existing ${method} log found; use a clean method output before rerunning" >&2
  exit 1
fi

mkdir -p "$result_dir"
echo "==> VisDA-C Stage14 preflight: target-head mix 0.4, 4 cycles, seed 2020"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/temporal_precision_head.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 4 DCCL.TARGET_HEAD_MIX 0.4

python tools/extract_final_accuracy.py \
  --glob "output/uda/VISDA-C/TV/${method}/*.txt" \
  --selection peak \
  > "$result_dir/temporal_precision_head_visda_mix040_preflight_accuracy.csv"

python tools/summarize_visda_temporal_precision_head.py \
  --glob "output/uda/VISDA-C/TV/${method}/*.txt" \
  --out "$result_dir/temporal_precision_head_visda_mix040_preflight_summary.json" \
  --csv-out "$result_dir/temporal_precision_head_visda_mix040_preflight_per_class.csv" \
  --class-names data/VISDA-C/classname.txt

python tools/analyze_temporal_conflict_dynamics.py \
  --glob "output/uda/VISDA-C/TV/${method}/temporal_diagnostics/*_cycle*.npz" \
  --out "$result_dir/temporal_precision_head_visda_mix040_preflight_dynamics.json" \
  --min-pass-tasks 1

python tools/summarize_visda_matched_preflight.py \
  --baseline-glob "$baseline_glob" \
  --candidate-glob "output/uda/VISDA-C/TV/${method}/*.txt" \
  --out "$result_dir/temporal_precision_head_visda_mix040_preflight_gate.json"
