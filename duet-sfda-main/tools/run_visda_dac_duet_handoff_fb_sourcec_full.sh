#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Ablation: reuse DAC F/B, but replace DAC target_C with the frozen source C.
# This keeps 15 + 4*4 = 31 target passes. Seed 2020 finished at 91.51%.
experiment_seed="${1:-2020}"
dac_method="duet_delayed_agreement_credit_visda_full_seed${experiment_seed}"
dac_run_dir="output/uda/VISDA-C/TV/${dac_method}"
dac_state="${dac_run_dir}/delayed_credit_state.pt"
handoff_source="output/dac_duet_handoff_fb_sourcec_seed${experiment_seed}"
handoff_source_dir="${handoff_source}/uda/VISDA-C/T"
method_name="plmatch_dac_handoff_fb_sourcec_full_seed${experiment_seed}"
run_dir="output/uda/VISDA-C/TV/${method_name}"

for required_path in \
  data/VISDA-C/train_list.txt \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_C.pt \
  "${dac_run_dir}/target_F.pt" \
  "${dac_run_dir}/target_B.pt" \
  "${dac_state}"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing full-data DAC handoff input: $required_path" >&2
    echo "Finish the 15-epoch full DAC run before starting this script" >&2
    exit 1
  fi
done

full_samples=$(wc -l < data/VISDA-C/validation_list.txt | tr -d ' ')
if [ "$full_samples" -ne 55388 ]; then
  echo "Unexpected VisDA-C full target size: ${full_samples}; expected 55388" >&2
  exit 1
fi

python - "$dac_state" "$full_samples" <<'PY'
import sys

import torch

state_path = sys.argv[1]
expected_samples = int(sys.argv[2])
state = torch.load(state_path, map_location="cpu", weights_only=True)
memory = state.get("memory")
if not isinstance(memory, torch.Tensor) or memory.ndim != 2:
    raise SystemExit("DAC handoff state has no valid full-distribution memory")
if int(memory.shape[0]) != expected_samples:
    raise SystemExit(
        "DAC handoff checkpoint is not full-data: "
        f"memory_rows={int(memory.shape[0])}, expected={expected_samples}"
    )
if not torch.isfinite(memory).all():
    raise SystemExit("DAC handoff memory contains non-finite values")
print(
    "==> Verified full-data DAC checkpoint: "
    f"samples={memory.shape[0]}; classes={memory.shape[1]}"
)
PY

mkdir -p "$handoff_source_dir"
cp -f "${dac_run_dir}/target_F.pt" "${handoff_source_dir}/source_F.pt"
cp -f "${dac_run_dir}/target_B.pt" "${handoff_source_dir}/source_B.pt"
cp -f source/uda/VISDA-C/T/source_C.pt "${handoff_source_dir}/source_C.pt"
if ! cmp -s source/uda/VISDA-C/T/source_C.pt "${handoff_source_dir}/source_C.pt"; then
  echo "Handoff classifier is not the untouched source classifier" >&2
  exit 1
fi

echo "==> Ablation: DAC F/B + frozen source C"
echo "==> Stage 2 starting: released DUET, 4 cycles x 4 passes"
echo "==> Total target passes: 31; target GT affects training: False"

python image_target_of_oh_vs.py \
  --cfg cfgs/visda/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC "$handoff_source" \
  MODEL.METHOD "$method_name" \
  SETTING.S 0 SETTING.T 1 SETTING.SEED "$experiment_seed" \
  ACTIVE.CYCLE 4 ACTIVE.ADAPTATION_LIST ""

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -lt 1 ]; then
  echo "No handoff log found in ${run_dir}" >&2
  exit 1
fi
latest_log=$(printf '%s\n' "${logs[@]}" | sort | tail -n 1)
if ! grep -q "Iter:3464/3464; Cycle: 4/4" "$latest_log"; then
  echo "Handoff run did not complete all four unchanged DUET cycles" >&2
  exit 1
fi
for checkpoint in target_F.pt target_B.pt target_C.pt; do
  if [ ! -f "${run_dir}/${checkpoint}" ]; then
    echo "Missing final handoff checkpoint: ${run_dir}/${checkpoint}" >&2
    exit 1
  fi
done
if ! grep -q "handoff_target_passes=16; final_checkpoint_fixed=True" "$latest_log"; then
  echo "Handoff run did not save the fixed final checkpoint contract" >&2
  exit 1
fi

echo "==> Final classifier-preserving DUET-handoff checkpoint"
grep "Cycle: 4/4" "$latest_log" | tail -n 1
echo "==> Archived seed-2020 result: 91.51%"
echo "==> Full log: ${latest_log}"
