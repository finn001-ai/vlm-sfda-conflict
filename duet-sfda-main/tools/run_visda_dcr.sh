#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Stable DCR-SFDA protocol for VisDA-C.
# DCM uses 15 epochs; CLM+ARG then use 8 cycles x 4 epochs.
experiment_seed="${1:-2020}"
dcm_method="dcr_memory_visda_full_seed${experiment_seed}"
dcm_run_dir="output/uda/VISDA-C/TV/${dcm_method}"
dcm_state="${dcm_run_dir}/dcr_memory_state.pt"
legacy_dcm_run_dir="output/uda/VISDA-C/TV/duet_delayed_agreement_credit_visda_full_seed${experiment_seed}"
legacy_dcm_state="${legacy_dcm_run_dir}/delayed_credit_state.pt"
handoff_source="output/dcr_visda_seed${experiment_seed}"
handoff_source_dir="${handoff_source}/uda/VISDA-C/T"
method_name="dcr_visda_full_seed${experiment_seed}"
run_dir="output/uda/VISDA-C/TV/${method_name}"

for required_path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt; do
  if [ ! -f "$required_path" ]; then
    echo "Missing VisDA-C input: $required_path" >&2
    exit 1
  fi
done

if [ -f "$dcm_state" ]; then
  echo "==> Reusing DCR memory: ${dcm_state}"
elif [ -f "$legacy_dcm_state" ]; then
  dcm_run_dir="$legacy_dcm_run_dir"
  dcm_state="$legacy_dcm_state"
  echo "==> Reusing legacy memory artifact: ${dcm_state}"
else
  if [ -d "$dcm_run_dir" ] && compgen -G "$dcm_run_dir/*.txt" > /dev/null; then
    echo "Partial DCM run exists but dcr_memory_state.pt is missing: $dcm_run_dir" >&2
    echo "Move that partial directory before rebuilding VisDA-C DCM" >&2
    exit 1
  fi
  echo "==> Building VisDA-C DCM for 15 epochs"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/dcr.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$dcm_method" \
    SETTING.S 0 SETTING.T 1 SETTING.SEED "$experiment_seed" \
    ACTIVE.ADAPTATION_LIST ""
fi

for artifact in "$dcm_state" \
  "${dcm_run_dir}/target_F.pt" \
  "${dcm_run_dir}/target_B.pt" \
  "${dcm_run_dir}/target_C.pt"; do
  if [ ! -f "$artifact" ]; then
    echo "Missing completed VisDA-C DCM artifact: $artifact" >&2
    exit 1
  fi
done

full_samples=$(awk 'END {print NR}' data/VISDA-C/validation_list.txt)
if [ "$full_samples" -ne 55388 ]; then
  echo "Unexpected VisDA-C full target size: ${full_samples}; expected 55388" >&2
  exit 1
fi

python - "$dcm_state" "$full_samples" <<'PY'
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
    raise SystemExit(f"DCR memory state is missing keys: {missing}")
memory = state["memory"]
if not isinstance(memory, torch.Tensor) or tuple(memory.shape) != (expected_samples, 12):
    raise SystemExit(
        "Invalid full-data VisDA DCR memory: "
        f"shape={getattr(memory, 'shape', None)}, expected=({expected_samples}, 12)"
    )
if not torch.isfinite(memory).all():
    raise SystemExit("DCR memory contains non-finite values")
print(f"==> Verified DCM: samples={memory.shape[0]}; classes={memory.shape[1]}")
PY

if [ -d "$run_dir" ] && compgen -G "$run_dir/*.txt" > /dev/null; then
  echo "Refusing to mix logs with an existing run: $run_dir" >&2
  echo "Move the existing directory aside before rerunning" >&2
  exit 1
fi

mkdir -p "$handoff_source_dir"
cp -f "${dcm_run_dir}/target_F.pt" "${handoff_source_dir}/source_F.pt"
cp -f "${dcm_run_dir}/target_B.pt" "${handoff_source_dir}/source_B.pt"
cp -f "${dcm_run_dir}/target_C.pt" "${handoff_source_dir}/source_C.pt"

echo "==> DCM ready: 15 epochs"
echo "==> CLM+ARG: 8 cycles x 4 epochs"
echo "==> Conflict hard admission: 0%"
echo "==> Residual soft target: unresolved conflicts supported by Task history"
echo "==> CLIP update: enabled; cumulative agreement admission: enabled"
echo "==> Total target passes: 47; target GT affects training: False"

python image_target_of_oh_vs.py \
  --cfg cfgs/visda/dcr.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC "$handoff_source" \
  MODEL.METHOD "$method_name" \
  SETTING.S 0 SETTING.T 1 SETTING.SEED "$experiment_seed" \
  TEST.MAX_EPOCH 4 TEST.INTERVAL 4 \
  ACTIVE.CYCLE 8 ACTIVE.ADAPTATION_LIST "" \
  DCR.FINAL_EXTRA_EPOCHS 0 \
  DCR.CREDIT_PRESERVING True \
  DCR.STATE_PATH "$dcm_state" \
  DCR.CONFLICT_HARD_FRACTION 0.0 \
  DCR.FREEZE_CLIP False \
  DCR.SOFT_REPLACEMENT_MODE task_supported \
  DCR.MEMORY_WRITE_MODE locked \
  DCR.CUMULATIVE_AGREEMENT_MASK True \
  DCR.CREDIT_DECAY 0.9 \
  DCR.CREDIT_ETA 4.0 \
  DCR.MEMORY_UPDATE_RATE 0.5 \
  DCR.CREDIT_MODE delayed \
  DCR.FEEDBACK_MODE agreement_temporal

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected one residual log in ${run_dir}, found ${#logs[@]}" >&2
  exit 1
fi
latest_log="${logs[0]}"
if [ "$(grep -c "Task: TV" "$latest_log")" -ne 32 ]; then
  echo "Run did not complete 8 cycles x 4 logged epochs" >&2
  exit 1
fi
if ! grep -q "DCR asymmetric residual guidance: cycle=8" "$latest_log"; then
  echo "DCR residual guidance was not active through cycle 8" >&2
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

echo "==> Best accuracy over the fixed 32-point trajectory: ${best_acc}%"
echo "==> Final fixed checkpoint"
grep "Cycle: 8/8" "$latest_log" | tail -n 1
echo "==> Full log: ${latest_log}"
