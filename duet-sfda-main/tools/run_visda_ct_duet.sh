#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

seed=2020
method="ct_duet_visda_full_seed${seed}"
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

# 用户要求可直接重跑，因此只覆盖这个脚本自己的固定输出目录。
case "$run_dir" in
  output/uda/VISDA-C/TV/ct_duet_visda_full_seed2020) ;;
  *)
    echo "Refusing to clear unexpected VisDA-C path: $run_dir" >&2
    exit 1
    ;;
esac
rm -rf -- "$run_dir"

echo "==> CT-DUET VisDA-C full target, 8 cycles, seed=${seed}"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/ct_duet.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 8

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected exactly one VisDA-C log, found ${#logs[@]}" >&2
  exit 1
fi
if [ "$(grep -c "CT-DUET complementary transition: enabled=True" "${logs[0]}")" -ne 1 ]; then
  echo "CT-DUET activation contract failed" >&2
  exit 1
fi
if [ "$(grep -c "CT-DUET cycle summary" "${logs[0]}")" -ne 8 ]; then
  echo "CT-DUET did not emit all eight cycle summaries" >&2
  exit 1
fi
if [ "$(grep -c "Task: TV" "${logs[0]}")" -ne 32 ]; then
  echo "VisDA-C did not finish 8 cycles / 32 checkpoints" >&2
  exit 1
fi

python tools/summarize_visda_run.py \
  --glob "$run_dir/*.txt" \
  --out "$result_dir/ct_duet_visda_seed2020_summary.json" \
  --csv-out "$result_dir/ct_duet_visda_seed2020_per_class.csv" \
  --class-names data/VISDA-C/classname.txt

echo "==> Baseline gate: final > 91.50 and oracle peak > 91.52"
echo "==> Summary: $result_dir/ct_duet_visda_seed2020_summary.json"
