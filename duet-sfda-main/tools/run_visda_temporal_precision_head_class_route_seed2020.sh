#!/usr/bin/env bash
set -euo pipefail

method="temporal_precision_head_seed2020_visda_class_route"
result_dir="output/uda/VISDA-C"
gate="$result_dir/temporal_precision_head_visda_class_route_preflight_gate.json"

python - "$gate" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing preflight gate: {path}")
decision = json.loads(path.read_text()).get("decision")
if decision != "pass_full_training_gate":
    raise SystemExit(f"Preflight did not pass: {decision}")
PY

if compgen -G "output/uda/VISDA-C/TV/${method}/*.txt" > /dev/null; then
  echo "Existing ${method} log found; use a clean method output before rerunning" >&2
  exit 1
fi

echo "==> VisDA-C class-routing full run: 8 cycles, seed 2020"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/temporal_precision_head.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 \
  DCCL.GTR_CLASS_ROUTING True \
  DCCL.GTR_CLASS_ROUTE_MIN_COUNT 20 \
  DCCL.GTR_CLASS_ROUTE_FLOOR 0.25 \
  DCCL.GTR_CLASS_ROUTE_MAX_RATIO 4.0

candidate_logs=(output/uda/VISDA-C/TV/${method}/*.txt)
if [ "${#candidate_logs[@]}" -ne 1 ] || ! grep -q "Task: TV" "${candidate_logs[0]}"; then
  echo "Full training produced no VisDA-C accuracy records; refusing to summarize" >&2
  exit 1
fi
if ! grep -q "DCCL class intervention routing" "${candidate_logs[0]}"; then
  echo "Class intervention routing did not activate; refusing to summarize" >&2
  exit 1
fi

python tools/extract_final_accuracy.py \
  --glob "output/uda/VISDA-C/TV/${method}/*.txt" \
  --selection peak \
  > "$result_dir/temporal_precision_head_visda_class_route_seed2020_accuracy.csv"

python tools/summarize_visda_temporal_precision_head.py \
  --glob "output/uda/VISDA-C/TV/${method}/*.txt" \
  --out "$result_dir/temporal_precision_head_visda_class_route_seed2020_summary.json" \
  --csv-out "$result_dir/temporal_precision_head_visda_class_route_seed2020_per_class.csv" \
  --class-names data/VISDA-C/classname.txt

python tools/analyze_temporal_conflict_dynamics.py \
  --glob "output/uda/VISDA-C/TV/${method}/temporal_diagnostics/*_cycle*.npz" \
  --out "$result_dir/temporal_precision_head_visda_class_route_seed2020_dynamics.json" \
  --min-pass-tasks 1
