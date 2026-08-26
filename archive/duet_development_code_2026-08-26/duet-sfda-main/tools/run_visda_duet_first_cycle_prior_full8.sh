#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

seed=2020
method="duet_first_cycle_prior_visda_full_seed${seed}"
run_dir="output/uda/VISDA-C/TV/${method}"
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

case "$run_dir" in
  output/uda/VISDA-C/TV/duet_first_cycle_prior_visda_full_seed2020) ;;
  *)
    echo "Refusing to clear unexpected VisDA-C path: $run_dir" >&2
    exit 1
    ;;
esac
rm -rf -- "$run_dir"

echo "==> DUET-FCP VisDA-C full target, 8 cycles, seed=${seed}"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 8

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected exactly one VisDA-C log, found ${#logs[@]}" >&2
  exit 1
fi
if [ "$(grep -c "DUET first-cycle prior schedule: cycle=1; active=True" "${logs[0]}")" -ne 1 ]; then
  echo "First-cycle prior activation contract failed" >&2
  exit 1
fi
if [ "$(grep -c "DUET first-cycle prior schedule: cycle=.*; active=False" "${logs[0]}")" -ne 7 ]; then
  echo "Cycles 2-8 did not preserve raw DUET probabilities" >&2
  exit 1
fi
if [ "$(grep -c "Task: TV" "${logs[0]}")" -ne 32 ]; then
  echo "VisDA-C did not finish 8 cycles / 32 checkpoints" >&2
  exit 1
fi

python tools/summarize_visda_run.py \
  --glob "$run_dir/*.txt" \
  --out "$result_dir/duet_fcp_visda_full8_seed2020_summary.json" \
  --csv-out "$result_dir/duet_fcp_visda_full8_seed2020_per_class.csv" \
  --class-names data/VISDA-C/classname.txt

office_glob="output/uda/office-home/*/plmatch_office_home_full_seed2020/*.txt"
office_count=0
if [ -d output/uda/office-home ]; then
  office_count=$(find output/uda/office-home -path "$office_glob" -type f | wc -l | tr -d ' ')
fi
if [ "$office_count" -eq 12 ]; then
  python tools/build_duet_benchmark_tables.py
  echo "==> Unified table: output/uda/benchmark_tables/duet_fcp_visda8_office_home_duet.md"
else
  echo "==> VisDA summary: $result_dir/duet_fcp_visda_full8_seed2020_summary.json"
  echo "Run Office-Home later, then: python tools/build_duet_benchmark_tables.py"
fi
