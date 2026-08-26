#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Uniform-budget VisDA experiment:
#   Stage 1: train DAC for 15 passes, or reuse its fixed final checkpoint.
#   Stage 2: released DUET for 4 cycles x 5 epochs (20 passes).
# Every DUET cycle has the same length; total target passes = 35.
experiment_seed="${1:-2020}"
dac_method="duet_delayed_agreement_credit_visda_full_seed${experiment_seed}"
dac_run_dir="output/uda/VISDA-C/TV/${dac_method}"
dac_state="${dac_run_dir}/delayed_credit_state.pt"
handoff_source="output/dac_duet_handoff_uniform5_source_seed${experiment_seed}"
handoff_source_dir="${handoff_source}/uda/VISDA-C/T"
method_name="plmatch_dac_handoff_uniform5_visda_full_seed${experiment_seed}"
run_dir="output/uda/VISDA-C/TV/${method_name}"

for required_path in \
  data/VISDA-C/train_list.txt \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$required_path" ]; then
    echo "Missing VisDA-C input: $required_path" >&2
    exit 1
  fi
done

full_samples=$(wc -l < data/VISDA-C/validation_list.txt | tr -d ' ')
if [ "$full_samples" -ne 55388 ]; then
  echo "Unexpected VisDA-C full target size: ${full_samples}; expected 55388" >&2
  exit 1
fi

dac_complete=true
for checkpoint in target_F.pt target_B.pt target_C.pt delayed_credit_state.pt; do
  if [ ! -f "${dac_run_dir}/${checkpoint}" ]; then
    dac_complete=false
  fi
done

if [ "$dac_complete" = false ]; then
  if [ -d "$dac_run_dir" ] && compgen -G "$dac_run_dir/*.txt" > /dev/null; then
    echo "Partial DAC directory exists: $dac_run_dir" >&2
    echo "Move it aside before rerunning the complete two-stage experiment" >&2
    exit 1
  fi
  echo "==> Stage 1/2: full-data DAC, 15 epochs"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/duet_delayed_agreement_credit.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$dac_method" \
    SETTING.S 0 SETTING.T 1 SETTING.SEED "$experiment_seed" \
    ACTIVE.ADAPTATION_LIST ""
else
  echo "==> Stage 1/2: reusing completed 15-epoch DAC checkpoint"
fi

for checkpoint in target_F.pt target_B.pt target_C.pt delayed_credit_state.pt; do
  if [ ! -f "${dac_run_dir}/${checkpoint}" ]; then
    echo "DAC stage did not produce ${dac_run_dir}/${checkpoint}" >&2
    exit 1
  fi
done

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

echo "==> Stage 1 complete: DAC F/B/C final checkpoint, 15 target passes"
echo "==> Stage 2/2: released DUET, 4 cycles x 5 epochs"
echo "==> Uniform DUET schedule: 5/5/5/5; total target passes: 35"
echo "==> Target GT affects training: False"

if [ -d "$run_dir" ] && compgen -G "$run_dir/*.txt" > /dev/null; then
  echo "Refusing to mix logs with an existing run: $run_dir" >&2
  echo "Move the existing directory aside before rerunning" >&2
  exit 1
fi

python image_target_of_oh_vs.py \
  --cfg cfgs/visda/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC "$handoff_source" \
  MODEL.METHOD "$method_name" \
  SETTING.S 0 SETTING.T 1 SETTING.SEED "$experiment_seed" \
  TEST.MAX_EPOCH 5 TEST.INTERVAL 5 \
  ACTIVE.CYCLE 4 ACTIVE.ADAPTATION_LIST "" \
  DUET_HANDOFF.FINAL_EXTRA_EPOCHS 0

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected one handoff log in ${run_dir}, found ${#logs[@]}" >&2
  exit 1
fi
latest_log="${logs[0]}"
if [ "$(grep -c "Task: TV" "$latest_log")" -ne 20 ]; then
  echo "Run did not complete 4 cycles x 5 logged epochs" >&2
  exit 1
fi
if ! grep -q "handoff_target_passes=20; final_checkpoint_fixed=True" "$latest_log"; then
  echo "Run did not save the uniform 20-pass DUET checkpoint" >&2
  exit 1
fi
for checkpoint in target_F.pt target_B.pt target_C.pt; do
  if [ ! -f "${run_dir}/${checkpoint}" ]; then
    echo "Missing final handoff checkpoint: ${run_dir}/${checkpoint}" >&2
    exit 1
  fi
done

echo "==> Final uniform-5 VisDA checkpoint"
grep "Cycle: 4/4" "$latest_log" | tail -n 1
echo "==> Full log: ${latest_log}"
