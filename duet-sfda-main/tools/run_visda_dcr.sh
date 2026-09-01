#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
source tools/lib/dcr_timing.sh

# Stable DCR-SFDA protocol for VisDA-C.
# DCM uses 15 epochs; CLM+ARG then use 8 cycles x 4 epochs.
experiment_seed="${1:-2020}"
profile="${2:-delayed}"

case "$profile" in
  delayed)
    dcm_credit_mode="delayed"
    dcm_method="dcr_memory_visda_rankadaptive_seed${experiment_seed}"
    handoff_source="output/dcr_visda_rankadaptive_seed${experiment_seed}"
    method_name="dcr_visda_rankadaptive_seed${experiment_seed}"
    ;;
  uniform_locked_arg)
    dcm_credit_mode="uniform"
    dcm_method="dcr_memory_uniform_visda_rankadaptive_seed${experiment_seed}"
    handoff_source="output/dcr_visda_uniform_locked_arg_rankadaptive_seed${experiment_seed}"
    method_name="dcr_visda_uniform_locked_arg_rankadaptive_seed${experiment_seed}"
    ;;
  *)
    echo "Profile must be delayed or uniform_locked_arg" >&2
    exit 1
    ;;
esac

dcm_run_dir="output/uda/VISDA-C/TV/${dcm_method}"
dcm_state="${dcm_run_dir}/dcr_memory_state.pt"
handoff_source_dir="${handoff_source}/uda/VISDA-C/T"
run_dir="output/uda/VISDA-C/TV/${method_name}"
timing_file="${run_dir}/stage_timing.csv"

for required_path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt; do
  if [ ! -f "$required_path" ]; then
    echo "Missing VisDA-C input: $required_path" >&2
    exit 1
  fi
done

if [ -d "$run_dir" ] && compgen -G "$run_dir/*.txt" > /dev/null; then
  echo "Refusing to mix logs with an existing run: $run_dir" >&2
  echo "Move the existing directory aside before rerunning" >&2
  exit 1
fi
dcr_timing_init "$timing_file"

if [ -f "$dcm_state" ]; then
  echo "==> Reusing ${dcm_credit_mode} rank-adaptive DCR memory: ${dcm_state}"
  if ! dcr_timing_has_stage "$timing_file" stage1; then
    dcr_timing_record "$timing_file" stage1 NA true NA NA
  fi
else
  if [ -d "$dcm_run_dir" ] && compgen -G "$dcm_run_dir/*.txt" > /dev/null; then
    echo "Partial DCM run exists but dcr_memory_state.pt is missing: $dcm_run_dir" >&2
    echo "Move that partial directory before rebuilding VisDA-C DCM" >&2
    exit 1
  fi
  echo "==> Stage 1/2: building VisDA-C ${dcm_credit_mode} rank-adaptive DCM for 15 epochs"
  stage1_started="$(date +%s)"
  stage1_started_iso="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/dcr.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$dcm_method" \
    SETTING.S 0 SETTING.T 1 SETTING.SEED "$experiment_seed" \
    ACTIVE.ADAPTATION_LIST "" \
    DCR_MEMORY.CREDIT_MODE "$dcm_credit_mode"
  stage1_finished="$(date +%s)"
  stage1_finished_iso="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  dcr_timing_record "$timing_file" stage1 \
    "$((stage1_finished - stage1_started))" false \
    "$stage1_started_iso" "$stage1_finished_iso"
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
dcm_logs=("$dcm_run_dir"/*.txt)
if [ "${#dcm_logs[@]}" -ne 1 ]; then
  echo "Expected one VisDA-C DCM log in ${dcm_run_dir}, found ${#dcm_logs[@]}" >&2
  exit 1
fi
if ! grep -q "alignment_mode=rank_adaptive" "${dcm_logs[0]}" \
  || ! grep -q "credit_mode=${dcm_credit_mode}" "${dcm_logs[0]}"; then
  echo "VisDA-C DCM log does not match profile=${profile}" >&2
  exit 1
fi

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

mkdir -p "$handoff_source_dir"
cp -f "${dcm_run_dir}/target_F.pt" "${handoff_source_dir}/source_F.pt"
cp -f "${dcm_run_dir}/target_B.pt" "${handoff_source_dir}/source_B.pt"
cp -f "${dcm_run_dir}/target_C.pt" "${handoff_source_dir}/source_C.pt"

echo "==> Profile=${profile}; DCM credit=${dcm_credit_mode}"
echo "==> Rank-adaptive DCM ready: 15 epochs"
echo "==> CLM+ARG: 8 cycles x 4 epochs"
echo "==> Conflict hard admission: 0%"
echo "==> Residual soft target: unresolved conflicts supported by Task history"
echo "==> CLIP update: enabled; cumulative agreement admission: enabled"
echo "==> Total target passes: 47; target GT affects training: False"

stage2_started="$(date +%s)"
stage2_started_iso="$(date '+%Y-%m-%dT%H:%M:%S%z')"
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
  DCR.CREDIT_MODE "$dcm_credit_mode" \
  DCR.FEEDBACK_MODE agreement_temporal
stage2_finished="$(date +%s)"
stage2_finished_iso="$(date '+%Y-%m-%dT%H:%M:%S%z')"
dcr_timing_record "$timing_file" stage2 \
  "$((stage2_finished - stage2_started))" false \
  "$stage2_started_iso" "$stage2_finished_iso"

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
if ! grep -q "DCR refinement: enabled=True;.*soft_replacement_mode=task_supported;.*memory_write_mode=locked; credit_mode=${dcm_credit_mode};" "$latest_log"; then
  echo "VisDA-C Stage-2 log does not match profile=${profile}" >&2
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
dcr_timing_record_total "$timing_file"
echo "==> Stage timing: ${timing_file}"
column -s, -t "$timing_file" 2>/dev/null || cat "$timing_file"
