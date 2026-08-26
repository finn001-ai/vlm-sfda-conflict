#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Stable DCR-SFDA protocol for one Office-Home task.
# Stage 1 builds delayed credit memory (DCM). Stage 2 protects conflicts with
# CLM and applies task-supported asymmetric residual guidance (ARG).
experiment_seed="${1:-2020}"
task="${2:-AC}"
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

dcm_method="dcr_memory_office_home_full_seed${experiment_seed}"
dcm_run_dir="output/uda/office-home/${task}/${dcm_method}"
dcm_state="${dcm_run_dir}/dcr_memory_state.pt"
legacy_dcm_run_dir="output/uda/office-home/${task}/duet_delayed_agreement_credit_office_home_full_seed${experiment_seed}"
legacy_dcm_state="${legacy_dcm_run_dir}/delayed_credit_state.pt"
legacy_handoff_dir="output/dac_duet_handoff_uniform5_office_home_seed${experiment_seed}_${task}/uda/office-home/${domain_keys[$source_index]}"
handoff_source="output/dcr_office_home_seed${experiment_seed}_${task}"
handoff_source_dir="${handoff_source}/uda/office-home/${domain_keys[$source_index]}"
method_name="dcr_office_home_full_seed${experiment_seed}"
run_dir="output/uda/office-home/${task}/${method_name}"
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

if [ -f "$dcm_state" ]; then
  echo "==> [${task}] Reusing DCR memory: ${dcm_state}"
elif [ -f "$legacy_dcm_state" ]; then
  dcm_run_dir="$legacy_dcm_run_dir"
  dcm_state="$legacy_dcm_state"
  echo "==> [${task}] Reusing legacy memory artifact: ${dcm_state}"
else
  if [ -d "$dcm_run_dir" ] && compgen -G "$dcm_run_dir/*.txt" > /dev/null; then
    echo "Partial DCM run exists but dcr_memory_state.pt is missing: $dcm_run_dir" >&2
    echo "Move that partial directory before rebuilding ${task} DCM" >&2
    exit 1
  fi
  echo "==> [${task}] Building DCM for 15 epochs"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/dcr.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$dcm_method" \
    SETTING.S "$source_index" SETTING.T "$target_index" \
    SETTING.SEED "$experiment_seed" \
    ACTIVE.ADAPTATION_LIST ""
fi

if [ ! -f "$dcm_state" ]; then
  echo "${task} DCM did not produce: $dcm_state" >&2
  exit 1
fi

# The earlier uniform handoff copied the fixed DAC F/B/C checkpoint into a
# source-shaped directory.  Some cloud cleanups retained that copy while
# removing target_F/B/C from the original DAC directory.  Both locations are
# byte-equivalent DAC-15 weights; never fall back to a post-DUET target model.
if [ -f "${dcm_run_dir}/target_F.pt" ] \
  && [ -f "${dcm_run_dir}/target_B.pt" ] \
  && [ -f "${dcm_run_dir}/target_C.pt" ]; then
  dcm_weight_f="${dcm_run_dir}/target_F.pt"
  dcm_weight_b="${dcm_run_dir}/target_B.pt"
  dcm_weight_c="${dcm_run_dir}/target_C.pt"
  dcm_weight_origin="dcm_run"
elif [ -f "${legacy_handoff_dir}/source_F.pt" ] \
  && [ -f "${legacy_handoff_dir}/source_B.pt" ] \
  && [ -f "${legacy_handoff_dir}/source_C.pt" ]; then
  dcm_weight_f="${legacy_handoff_dir}/source_F.pt"
  dcm_weight_b="${legacy_handoff_dir}/source_B.pt"
  dcm_weight_c="${legacy_handoff_dir}/source_C.pt"
  dcm_weight_origin="preserved_legacy_copy"
else
  echo "Missing ${task} DCM F/B/C weights in both supported locations:" >&2
  echo "  ${dcm_run_dir}/target_{F,B,C}.pt" >&2
  echo "  ${legacy_handoff_dir}/source_{F,B,C}.pt" >&2
  exit 1
fi

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

if [ -d "$run_dir" ] && compgen -G "$run_dir/*.txt" > /dev/null; then
  echo "Refusing to mix logs with an existing run: $run_dir" >&2
  echo "Move the existing directory aside before rerunning" >&2
  exit 1
fi

mkdir -p "$handoff_source_dir"
cp -f "$dcm_weight_f" "${handoff_source_dir}/source_F.pt"
cp -f "$dcm_weight_b" "${handoff_source_dir}/source_B.pt"
cp -f "$dcm_weight_c" "${handoff_source_dir}/source_C.pt"

echo "==> [${task}] DCM ready; weight_origin=${dcm_weight_origin}"
echo "==> [${task}] CLM locks conflict memory; ARG corrects only Task-supported conflicts"
echo "==> [${task}] Conflict hard admission: 0%; total target passes: 31"

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
  DCR.CREDIT_MODE delayed \
  DCR.FEEDBACK_MODE agreement_temporal

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
for checkpoint in target_F.pt target_B.pt target_C.pt refined_credit_state.pt; do
  if [ ! -f "${run_dir}/${checkpoint}" ]; then
    echo "Missing final ${task} checkpoint: ${run_dir}/${checkpoint}" >&2
    exit 1
  fi
done

echo "==> [${task}] Final fixed checkpoint"
grep "Cycle: 4/4" "$latest_log" | tail -n 1
echo "==> [${task}] Full log: ${latest_log}"
