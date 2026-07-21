#!/usr/bin/env bash
set -euo pipefail

baseline_method="temporal_precision_head_seed2020_visda"
method="temporal_precision_head_seed2020_visda_stability3_preflight"
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
echo "==> VisDA-C temporal-stability preflight: PL/GTR stable cycles 3, 5 cycles, seed 2020"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/temporal_precision_head.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 5 \
  DCCL.PL_STABLE_CYCLES 3 \
  DCCL.GTR_STABLE_CYCLES 3

candidate_logs=(output/uda/VISDA-C/TV/${method}/*.txt)
if [ "${#candidate_logs[@]}" -ne 1 ] || ! grep -q "Task: TV" "${candidate_logs[0]}"; then
  echo "Preflight training produced no VisDA-C accuracy records; refusing to summarize" >&2
  exit 1
fi

python tools/extract_final_accuracy.py \
  --glob "output/uda/VISDA-C/TV/${method}/*.txt" \
  --selection peak \
  > "$result_dir/temporal_precision_head_visda_stability3_preflight_accuracy.csv"

python tools/summarize_visda_temporal_precision_head.py \
  --glob "output/uda/VISDA-C/TV/${method}/*.txt" \
  --out "$result_dir/temporal_precision_head_visda_stability3_preflight_summary.json" \
  --csv-out "$result_dir/temporal_precision_head_visda_stability3_preflight_per_class.csv" \
  --class-names data/VISDA-C/classname.txt

python tools/analyze_temporal_conflict_dynamics.py \
  --glob "output/uda/VISDA-C/TV/${method}/temporal_diagnostics/*_cycle*.npz" \
  --out "$result_dir/temporal_precision_head_visda_stability3_preflight_dynamics.json" \
  --min-pass-tasks 1

python tools/summarize_visda_matched_preflight.py \
  --baseline-glob "$baseline_glob" \
  --candidate-glob "output/uda/VISDA-C/TV/${method}/*.txt" \
  --out "$result_dir/temporal_precision_head_visda_stability3_preflight_gate.json" \
  --matched-start-cycle 4 \
  --matched-cycles 5 \
  --min-improvement 0.20 \
  --expected-candidate-mix 0.3 \
  --expected-baseline-pl-stable-cycles 2 \
  --expected-candidate-pl-stable-cycles 3 \
  --expected-baseline-gtr-stable-cycles 2 \
  --expected-candidate-gtr-stable-cycles 3 \
  --dynamics-json "$result_dir/temporal_precision_head_visda_stability3_preflight_dynamics.json" \
  --pass-command "bash tools/run_visda_temporal_precision_head_stability3_seed2020.sh" \
  --fail-next "do not run the full stability-3 job; archive it and compare PL precision/coverage with GTR correction coverage before choosing a one-axis temporal variant"
