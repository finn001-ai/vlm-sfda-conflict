#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
source tools/lib/dcr_timing.sh

# Stable DCR-SFDA protocol for one Office-Home task.
# Profiles keep the historical delayed run reproducible while exposing the
# validated uniform-credit + locked-memory + ARG candidate without mixing
# checkpoints or logs.
experiment_seed="${1:-2020}"
task="${2:-AC}"
profile="${3:-delayed}"
domain_keys=(A C P R)
domain_names=(Art Clipart Product RealWorld)

case "$task" in
  AC) source_index=0; target_index=1 ;;
  AP) source_index=0; target_index=2 ;;
  AR) source_index=0; target_index=3 ;;
  CA) source_index=1; target_index=0 ;;
  CP) source_index=1; target_index=2 ;;
  CR) source_index=1; target_index=3 ;;
  PA) source_index=2; target_index=0 ;;
  PC) source_index=2; target_index=1 ;;
  PR) source_index=2; target_index=3 ;;
  RA) source_index=3; target_index=0 ;;
  RC) source_index=3; target_index=1 ;;
  RP) source_index=3; target_index=2 ;;
  *)
    echo "Task must be one of: AC AP AR CA CP CR PA PC PR RA RC RP" >&2
    exit 1
    ;;
esac

case "$profile" in
  delayed)
    dcm_credit_mode="delayed"
    dcm_method="dcr_memory_office_home_rankadaptive_seed${experiment_seed}"
    handoff_source="output/dcr_office_home_rankadaptive_seed${experiment_seed}_${task}"
    method_name="dcr_office_home_rankadaptive_seed${experiment_seed}"
    ;;
  uniform_locked_arg)
    dcm_credit_mode="uniform"
    dcm_method="dcr_memory_uniform_office_home_rankadaptive_seed${experiment_seed}"
    handoff_source="output/dcr_office_home_uniform_locked_arg_rankadaptive_seed${experiment_seed}_${task}"
    method_name="dcr_office_home_uniform_locked_arg_rankadaptive_seed${experiment_seed}"
    ;;
  *)
    echo "Profile must be delayed or uniform_locked_arg" >&2
    exit 1
    ;;
esac

dcm_run_dir="output/uda/office-home/${task}/${dcm_method}"
dcm_state="${dcm_run_dir}/dcr_memory_state.pt"
handoff_source_dir="${handoff_source}/uda/office-home/${domain_keys[$source_index]}"
run_dir="output/uda/office-home/${task}/${method_name}"
timing_file="${run_dir}/stage_timing.csv"
target_list="data/office-home/${domain_names[$target_index]}_list.txt"

for required_path in \
  "$target_list" \
  data/office-home/classname.txt \
  "source/uda/office-home/${domain_keys[$source_index]}/source_F.pt" \
  "source/uda/office-home/${domain_keys[$source_index]}/source_B.pt" \
  "source/uda/office-home/${domain_keys[$source_index]}/source_C.pt"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing ${task} base input: $required_path" >&2
    exit 1
  fi
done

if [ -d "$run_dir" ] && compgen -G "$run_dir/*.txt" > /dev/null; then
  echo "Refusing to mix logs with an existing run: $run_dir" >&2
  echo "Move the existing directory aside before rerunning" >&2
  exit 1
fi
dcr_timing_init "$timing_file"

# Reuse the already validated four-task uniform DCM screen when available.
# The log contract prevents a delayed checkpoint from being accepted merely
# because it has a compatible tensor shape.
if [ "$profile" = "uniform_locked_arg" ] && [ ! -f "$dcm_state" ]; then
  legacy_dcm_run_dir="output/uda/office-home/${task}/dcr_memory_ablation_uniform_office_home_rankadaptive_seed${experiment_seed}"
  legacy_dcm_state="${legacy_dcm_run_dir}/dcr_memory_state.pt"
  legacy_logs=("$legacy_dcm_run_dir"/*.txt)
  if [ -f "$legacy_dcm_state" ] \
    && [ -f "${legacy_dcm_run_dir}/target_F.pt" ] \
    && [ -f "${legacy_dcm_run_dir}/target_B.pt" ] \
    && [ -f "${legacy_dcm_run_dir}/target_C.pt" ] \
    && [ "${#legacy_logs[@]}" -eq 1 ] \
    && grep -q "alignment_mode=rank_adaptive" "${legacy_logs[0]}" \
    && grep -q "credit_mode=uniform" "${legacy_logs[0]}"; then
    dcm_run_dir="$legacy_dcm_run_dir"
    dcm_state="$legacy_dcm_state"
    echo "==> [${task}] Reusing validated uniform DCM screen: ${dcm_state}"
  fi
fi

if [ -f "$dcm_state" ]; then
  echo "==> [${task}] Reusing ${dcm_credit_mode} rank-adaptive DCR memory: ${dcm_state}"
  if ! dcr_timing_has_stage "$timing_file" stage1; then
    dcr_timing_record "$timing_file" stage1 NA true NA NA
  fi
else
  if [ -d "$dcm_run_dir" ] && compgen -G "$dcm_run_dir/*.txt" > /dev/null; then
    echo "Partial DCM run exists but dcr_memory_state.pt is missing: $dcm_run_dir" >&2
    echo "Move that partial directory before rebuilding ${task} DCM" >&2
    exit 1
  fi
  echo "==> [${task}] Stage 1/2: building ${dcm_credit_mode} rank-adaptive DCM for 15 epochs"
  stage1_started="$(date +%s)"
  stage1_started_iso="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dcr.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$dcm_method" \
    SETTING.S "$source_index" SETTING.T "$target_index" \
    SETTING.SEED "$experiment_seed" \
    ACTIVE.ADAPTATION_LIST "" \
    DCR_MEMORY.CREDIT_MODE "$dcm_credit_mode"
  stage1_finished="$(date +%s)"
  stage1_finished_iso="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  dcr_timing_record "$timing_file" stage1 \
    "$((stage1_finished - stage1_started))" false \
    "$stage1_started_iso" "$stage1_finished_iso"
fi

if [ ! -f "$dcm_state" ]; then
  echo "${task} DCM did not produce: $dcm_state" >&2
  exit 1
fi

for artifact in \
  "${dcm_run_dir}/target_F.pt" \
  "${dcm_run_dir}/target_B.pt" \
  "${dcm_run_dir}/target_C.pt"; do
  if [ ! -f "$artifact" ]; then
    echo "Missing completed ${task} rank-adaptive DCM artifact: $artifact" >&2
    exit 1
  fi
done
dcm_logs=("$dcm_run_dir"/*.txt)
if [ "${#dcm_logs[@]}" -ne 1 ]; then
  echo "Expected one ${task} DCM log in ${dcm_run_dir}, found ${#dcm_logs[@]}" >&2
  exit 1
fi
if ! grep -q "alignment_mode=rank_adaptive" "${dcm_logs[0]}" \
  || ! grep -q "credit_mode=${dcm_credit_mode}" "${dcm_logs[0]}"; then
  echo "${task} DCM log does not match profile=${profile}: ${dcm_run_dir}" >&2
  exit 1
fi
dcm_weight_f="${dcm_run_dir}/target_F.pt"
dcm_weight_b="${dcm_run_dir}/target_B.pt"
dcm_weight_c="${dcm_run_dir}/target_C.pt"

target_samples=$(awk 'END {print NR}' "$target_list")
python - "$dcm_state" "$target_samples" "$task" <<'PY'
import sys

import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
expected_samples = int(sys.argv[2])
task = sys.argv[3]
memory = state.get("memory")
if not isinstance(memory, torch.Tensor) or tuple(memory.shape) != (expected_samples, 65):
    raise SystemExit(
        f"{task}: invalid DCR memory: shape={getattr(memory, 'shape', None)}, "
        f"expected=({expected_samples}, 65)"
    )
if not torch.isfinite(memory).all():
    raise SystemExit(f"{task}: DCR memory contains non-finite values")
print(f"==> [{task}] Verified DCM: samples={memory.shape[0]}; classes={memory.shape[1]}")
PY

mkdir -p "$handoff_source_dir"
cp -f "$dcm_weight_f" "${handoff_source_dir}/source_F.pt"
cp -f "$dcm_weight_b" "${handoff_source_dir}/source_B.pt"
cp -f "$dcm_weight_c" "${handoff_source_dir}/source_C.pt"

echo "==> [${task}] Profile=${profile}; DCM credit=${dcm_credit_mode}"
echo "==> [${task}] Rank-adaptive DCM ready"
echo "==> [${task}] CLM locks conflict memory; ARG corrects only Task-supported conflicts"
echo "==> [${task}] Conflict hard admission: 0%; total target passes: 31"

stage2_started="$(date +%s)"
stage2_started_iso="$(date '+%Y-%m-%dT%H:%M:%S%z')"
python image_target_of_oh_vs.py \
  --cfg cfgs/office-home/dcr.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC "$handoff_source" \
  MODEL.METHOD "$method_name" \
  SETTING.S "$source_index" SETTING.T "$target_index" \
  SETTING.SEED "$experiment_seed" \
  TEST.MAX_EPOCH 4 TEST.INTERVAL 4 \
  ACTIVE.CYCLE 4 ACTIVE.ADAPTATION_LIST "" \
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
  echo "Expected one ${task} residual log in ${run_dir}, found ${#logs[@]}" >&2
  exit 1
fi
latest_log="${logs[0]}"
if [ "$(grep -c "Task: ${task}" "$latest_log")" -ne 16 ]; then
  echo "${task} did not complete 4 cycles x 4 logged epochs" >&2
  exit 1
fi
if ! grep -q "DCR asymmetric residual guidance: cycle=4" "$latest_log"; then
  echo "DCR residual guidance was not active through cycle 4" >&2
  exit 1
fi
if ! grep -q "DCR refinement: enabled=True;.*soft_replacement_mode=task_supported;.*memory_write_mode=locked; credit_mode=${dcm_credit_mode};" "$latest_log"; then
  echo "${task} Stage-2 log does not match profile=${profile}" >&2
  exit 1
fi
for checkpoint in target_F.pt target_B.pt target_C.pt refined_credit_state.pt; do
  if [ ! -f "${run_dir}/${checkpoint}" ]; then
    echo "Missing final ${task} checkpoint: ${run_dir}/${checkpoint}" >&2
    exit 1
  fi
done

echo "==> [${task}] Final fixed checkpoint"
grep "Cycle: 4/4" "$latest_log" | tail -n 1
echo "==> [${task}] Full log: ${latest_log}"
dcr_timing_record_total "$timing_file"
echo "==> [${task}] Stage timing: ${timing_file}"
column -s, -t "$timing_file" 2>/dev/null || cat "$timing_file"
