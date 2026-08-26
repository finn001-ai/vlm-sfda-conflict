#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Pure PLMatch baseline on the full VisDA-C target set.
experiment_seed="${1:-2020}"
method_name="plmatch_visda_full_seed${experiment_seed}"
run_dir="output/uda/VISDA-C/TV/${method_name}"

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

logs=("$run_dir"/*.txt)
if [ -f "${run_dir}/target_F.pt" ] \
  && [ "${#logs[@]}" -eq 1 ] \
  && [ "$(grep -c "Task: TV" "${logs[0]}")" -eq 32 ]; then
  echo "==> VisDA-C PLMatch already complete: ${run_dir}"
  exit 0
fi

python image_target_of_oh_vs.py \
  --cfg cfgs/visda/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method_name" \
  SETTING.S 0 SETTING.T 1 SETTING.SEED "$experiment_seed" \
  TEST.MAX_EPOCH 4 TEST.INTERVAL 4 \
  ACTIVE.CYCLE 8 ACTIVE.ADAPTATION_LIST ""

echo "==> VisDA-C PLMatch completed for seed ${experiment_seed}"
