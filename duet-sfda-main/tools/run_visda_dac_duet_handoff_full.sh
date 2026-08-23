#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Two-stage full-data rescue:
#   1. reuse the fixed final checkpoint of the completed 15-epoch DAC run;
#   2. hand its Task F/B/C state to the released DUET path for four cycles.
# This gives 15 + 4*4 = 31 target passes, close to DUET's 8*4 = 32.
experiment_seed="${1:-2020}"
dac_method="duet_delayed_agreement_credit_visda_full_seed${experiment_seed}"
dac_run_dir="output/uda/VISDA-C/TV/${dac_method}"
dac_state="${dac_run_dir}/delayed_credit_state.pt"
handoff_source="output/dac_duet_handoff_source_seed${experiment_seed}"
handoff_source_dir="${handoff_source}/uda/VISDA-C/T"
method_name="plmatch_dac_handoff_full_seed${experiment_seed}"
run_dir="output/uda/VISDA-C/TV/${method_name}"

for required_path in \
  data/VISDA-C/train_list.txt \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  "${dac_run_dir}/target_F.pt" \
  "${dac_run_dir}/target_B.pt" \
  "${dac_run_dir}/target_C.pt" \
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

# A proxy25 checkpoint has 13,847 memory rows and must never enter this run.
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
cp -f "${dac_run_dir}/target_C.pt" "${handoff_source_dir}/source_C.pt"

echo "==> Stage 1 reused: DAC final checkpoint, 15 target passes"
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
if ! grep -q "Cycle: 4/4" "$latest_log"; then
  echo "Handoff run did not complete all four DUET cycles" >&2
  exit 1
fi

echo "==> Final DUET-handoff checkpoint"
grep "Cycle: 4/4" "$latest_log" | tail -n 1
echo "==> Reference full-data DUET final: 91.50%"
echo "==> Full log: ${latest_log}"
