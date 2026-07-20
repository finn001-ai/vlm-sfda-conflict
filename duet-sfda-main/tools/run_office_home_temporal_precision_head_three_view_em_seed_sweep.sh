#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
from pathlib import Path
p = Path('output/uda/office-home/temporal_precision_head_three_view_em_seed2022_summary.json')
if not p.exists() or json.loads(p.read_text()).get('decision') != 'pass_three_view_em_seed2022_gate':
    raise SystemExit('Stage22 seed-2022 gate has not passed; refusing the seed sweep')
PY

declare -a SEEDS=(2020 2021)
declare -a TASKS=(
  "AC 0 1" "AP 0 2" "AR 0 3" "CA 1 0" "CP 1 2" "CR 1 3"
  "PA 2 0" "PC 2 1" "PR 2 3" "RA 3 0" "RC 3 1" "RP 3 2"
)

for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    read -r task_name s t <<< "$task"
    echo "==> Stage22 stability: seed=${seed} task=${task_name}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/temporal_precision_head_three_view_em.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "temporal_precision_head_seed${seed}_three_view_em" \
      SETTING.SEED "$seed" SETTING.S "$s" SETTING.T "$t"
  done
done

python tools/extract_final_accuracy.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed[0-9][0-9][0-9][0-9]_three_view_em/*.txt' \
  --selection peak \
  > output/uda/office-home/temporal_precision_head_three_view_em_seed_sweep_accuracy.csv

python tools/summarize_three_view_em_flow.py \
  --glob 'output/uda/office-home/*/temporal_precision_head_seed[0-9][0-9][0-9][0-9]_three_view_em/*.txt' \
  --out output/uda/office-home/temporal_precision_head_three_view_em_seed_sweep_flow.json

python tools/summarize_three_view_em_stability.py \
  --csv output/uda/office-home/temporal_precision_head_three_view_em_seed_sweep_accuracy.csv \
  --flow output/uda/office-home/temporal_precision_head_three_view_em_seed_sweep_flow.json \
  --out output/uda/office-home/temporal_precision_head_three_view_em_seed_sweep_summary.json
