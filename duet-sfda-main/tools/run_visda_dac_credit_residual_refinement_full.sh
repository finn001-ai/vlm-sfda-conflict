#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run the same DAC-state-preserving residual refinement used by the final
# Office-Home experiment: no conflict hard admission, cumulative agreement
# admission, adaptive CLIP, and DAC-memory KL replacement only when the
# retained history supports the current Task candidate.
experiment_seed="${1:-2020}"
dac_method="duet_delayed_agreement_credit_visda_full_seed${experiment_seed}"
dac_run_dir="output/uda/VISDA-C/TV/${dac_method}"
dac_state="${dac_run_dir}/delayed_credit_state.pt"
handoff_source="output/dac_credit_residual_visda_seed${experiment_seed}"
handoff_source_dir="${handoff_source}/uda/VISDA-C/T"
method_name="plmatch_dac_handoff_credit_residual_visda_full_seed${experiment_seed}"
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
    echo "Complete the 15-epoch full-data DAC stage before refinement" >&2
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
required = {
    "memory",
    "previous_task",
    "previous_clip",
    "task_loss_sum",
    "clip_loss_sum",
    "feedback_mass",
    "task_weight",
    "clip_weight",
}
missing = sorted(required.difference(state))
if missing:
    raise SystemExit(f"DAC state is missing keys: {missing}")
memory = state["memory"]
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
echo "==> Stage 2: state-preserving residual refinement, 4 cycles x 4 epochs"
echo "==> Conflict hard admission: 0%"
echo "==> Residual soft target: unresolved conflicts supported by Task history"
echo "==> CLIP update: enabled; cumulative agreement admission: enabled"
echo "==> Total target passes: 31; target GT affects training: False"

python image_target_of_oh_vs.py \
  --cfg cfgs/visda/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC "$handoff_source" \
  MODEL.METHOD "$method_name" \
  SETTING.S 0 SETTING.T 1 SETTING.SEED "$experiment_seed" \
  TEST.MAX_EPOCH 4 TEST.INTERVAL 4 \
  ACTIVE.CYCLE 8 ACTIVE.ADAPTATION_LIST "" \
  DUET_HANDOFF.FINAL_EXTRA_EPOCHS 0 \
  DUET_HANDOFF.CREDIT_PRESERVING True \
  DUET_HANDOFF.STATE_PATH "$dac_state" \
  DUET_HANDOFF.CONFLICT_HARD_FRACTION 0.0 \
  DUET_HANDOFF.FREEZE_CLIP False \
  DUET_HANDOFF.SOFT_REPLACEMENT_MODE task_supported \
  DUET_HANDOFF.CUMULATIVE_AGREEMENT_MASK True

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected one residual log in ${run_dir}, found ${#logs[@]}" >&2
  exit 1
fi
latest_log="${logs[0]}"
if [ "$(grep -c "Task: TV" "$latest_log")" -ne 16 ]; then
  echo "Run did not complete 4 cycles x 4 logged epochs" >&2
  exit 1
fi
if ! grep -q "DAC credit residual KL: cycle=4" "$latest_log"; then
  echo "DAC residual was not active through cycle 4" >&2
  exit 1
fi
for checkpoint in target_F.pt target_B.pt target_C.pt refined_credit_state.pt; do
  if [ ! -f "${run_dir}/${checkpoint}" ]; then
    echo "Missing final refinement checkpoint: ${run_dir}/${checkpoint}" >&2
    exit 1
  fi
done

best_acc=$(sed -nE \
  '/Task:[[:space:]]*TV,.*Accuracy[[:space:]]*=/s/.*Accuracy[[:space:]]*=[[:space:]]*([0-9]+([.][0-9]+)?)%.*/\1/p' \
  "$latest_log" \
  | awk 'BEGIN { best = -1 } $1 + 0 > best { best = $1 + 0 } END { if (best >= 0) printf "%.2f", best }')

echo "==> Best accuracy over the fixed 16-point trajectory: ${best_acc}%"
echo "==> Final fixed checkpoint"
grep "Cycle: 4/4" "$latest_log" | tail -n 1
echo "==> Full log: ${latest_log}"
