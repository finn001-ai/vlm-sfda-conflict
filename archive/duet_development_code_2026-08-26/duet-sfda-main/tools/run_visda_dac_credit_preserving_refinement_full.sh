#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Reuse the completed 15-epoch full-data DAC run, then replace released DUET's
# history-erasing feedback loop with four cycles of credit-preserving
# refinement.  The second stage is uniform 4/4/4/4 (31 total target passes).
experiment_seed="${1:-2020}"
dac_method="duet_delayed_agreement_credit_visda_full_seed${experiment_seed}"
dac_run_dir="output/uda/VISDA-C/TV/${dac_method}"
dac_state="${dac_run_dir}/delayed_credit_state.pt"
handoff_source="output/dac_credit_preserving_source_seed${experiment_seed}"
handoff_source_dir="${handoff_source}/uda/VISDA-C/T"
method_name="plmatch_dac_handoff_credit_preserving_visda_full_seed${experiment_seed}"
run_dir="output/uda/VISDA-C/TV/${method_name}"

for required_path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  "${dac_run_dir}/target_F.pt" \
  "${dac_run_dir}/target_B.pt" \
  "${dac_run_dir}/target_C.pt" \
  "${dac_state}"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing full-data DAC input: $required_path" >&2
    echo "This experiment is designed to reuse the completed DAC-15 checkpoint" >&2
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

state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
expected_samples = int(sys.argv[2])
memory = state.get("memory")
if not isinstance(memory, torch.Tensor) or tuple(memory.shape) != (expected_samples, 12):
    raise SystemExit(
        "Invalid full-data VisDA DAC memory: "
        f"shape={getattr(memory, 'shape', None)}, expected=({expected_samples}, 12)"
    )
if not torch.isfinite(memory).all():
    raise SystemExit("DAC memory contains non-finite values")
print(f"==> Verified DAC state: samples={memory.shape[0]}; classes={memory.shape[1]}")
PY

if [ -d "$run_dir" ] && compgen -G "$run_dir/*.txt" > /dev/null; then
  echo "Refusing to mix logs with an existing run: $run_dir" >&2
  echo "Move the existing directory aside before rerunning" >&2
  exit 1
fi

mkdir -p "$handoff_source_dir"
cp -f "${dac_run_dir}/target_F.pt" "${handoff_source_dir}/source_F.pt"
cp -f "${dac_run_dir}/target_B.pt" "${handoff_source_dir}/source_B.pt"
cp -f "${dac_run_dir}/target_C.pt" "${handoff_source_dir}/source_C.pt"

echo "==> Stage 1 reused: full-data DAC, 15 epochs"
echo "==> Stage 2: credit-preserving refinement, 4 cycles x 4 epochs"
echo "==> Conflict soft coverage: 100%; hard rank coverage: 80%"
echo "==> Total target passes: 31; target GT affects training: False"

python image_target_of_oh_vs.py \
  --cfg cfgs/visda/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC "$handoff_source" \
  MODEL.METHOD "$method_name" \
  SETTING.S 0 SETTING.T 1 SETTING.SEED "$experiment_seed" \
  TEST.MAX_EPOCH 4 TEST.INTERVAL 4 \
  ACTIVE.CYCLE 4 ACTIVE.ADAPTATION_LIST "" \
  DUET_HANDOFF.FINAL_EXTRA_EPOCHS 0 \
  DUET_HANDOFF.CREDIT_PRESERVING True \
  DUET_HANDOFF.STATE_PATH "$dac_state" \
  DUET_HANDOFF.CONFLICT_HARD_FRACTION 0.8 \
  DUET_HANDOFF.FREEZE_CLIP True

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected one refinement log in ${run_dir}, found ${#logs[@]}" >&2
  exit 1
fi
latest_log="${logs[0]}"
if [ "$(grep -c "Task: TV" "$latest_log")" -ne 16 ]; then
  echo "Run did not complete 4 cycles x 4 logged epochs" >&2
  exit 1
fi
if ! grep -q "DAC credit-preserving teacher: cycle=4" "$latest_log"; then
  echo "Credit-preserving teacher was not active through cycle 4" >&2
  exit 1
fi
for checkpoint in target_F.pt target_B.pt target_C.pt refined_credit_state.pt; do
  if [ ! -f "${run_dir}/${checkpoint}" ]; then
    echo "Missing final refinement checkpoint: ${run_dir}/${checkpoint}" >&2
    exit 1
  fi
done

echo "==> Final fixed checkpoint"
grep "Cycle: 4/4" "$latest_log" | tail -n 1
echo "==> Full log: ${latest_log}"
