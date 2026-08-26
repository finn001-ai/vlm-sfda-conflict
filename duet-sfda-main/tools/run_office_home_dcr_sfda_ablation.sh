#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Office-Home component ablations for DCR-SFDA.
# Usage: bash tools/run_office_home_dcr_sfda_ablation.sh SEED TASK VARIANT
# Core variants: full, dcm_uniform, clm_writable, arg_none.
# Focused variants: dcm_no_history, dcm_no_temporal, clm_frozen,
# arg_all_conflicts, no_cumulative, freeze_vlm.

experiment_seed="${1:-2020}"
task="${2:-AC}"
variant="${3:-}"

if [ -z "$variant" ]; then
  echo "Usage: $0 SEED TASK VARIANT" >&2
  echo "Variants: full dcm_uniform dcm_no_history dcm_no_temporal clm_writable clm_frozen arg_none arg_all_conflicts no_cumulative freeze_vlm" >&2
  exit 1
fi

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

dcm_tag="full"
dcm_credit_mode="delayed"
dcm_feedback_mode="agreement_temporal"
dcm_decay="0.9"
memory_write_mode="locked"
soft_replacement_mode="task_supported"
cumulative_agreement="True"
freeze_vlm="False"

case "$variant" in
  full) ;;
  dcm_uniform)
    dcm_tag="uniform"
    dcm_credit_mode="uniform"
    ;;
  dcm_no_history)
    dcm_tag="no_history"
    dcm_decay="0.0"
    ;;
  dcm_no_temporal)
    dcm_tag="no_temporal"
    dcm_feedback_mode="agreement_only"
    ;;
  clm_writable)
    memory_write_mode="writable"
    ;;
  clm_frozen)
    memory_write_mode="frozen"
    ;;
  arg_none)
    soft_replacement_mode="none"
    ;;
  arg_all_conflicts)
    soft_replacement_mode="all_conflicts"
    ;;
  no_cumulative)
    cumulative_agreement="False"
    ;;
  freeze_vlm)
    freeze_vlm="True"
    ;;
  *)
    echo "Unknown variant: $variant" >&2
    exit 1
    ;;
esac

if [ "$dcm_tag" = "full" ]; then
  dcm_method="duet_delayed_agreement_credit_office_home_full_seed${experiment_seed}"
else
  dcm_method="duet_delayed_agreement_credit_dcr_ablation_${dcm_tag}_office_home_seed${experiment_seed}"
fi
dcm_run_dir="output/uda/office-home/${task}/${dcm_method}"
dcm_state="${dcm_run_dir}/delayed_credit_state.pt"
legacy_handoff_dir="output/dac_duet_handoff_uniform5_office_home_seed${experiment_seed}_${task}/uda/office-home/${domain_keys[$source_index]}"
handoff_source="output/dcr_sfda_ablation_${variant}_office_home_seed${experiment_seed}_${task}"
handoff_source_dir="${handoff_source}/uda/office-home/${domain_keys[$source_index]}"
method_name="plmatch_dac_handoff_dcr_sfda_ablation_${variant}_office_home_seed${experiment_seed}"
run_dir="output/uda/office-home/${task}/${method_name}"
target_list="data/office-home/${domain_names[$target_index]}_list.txt"

for required_path in \
  "$target_list" \
  data/office-home/classname.txt \
  "source/uda/office-home/${domain_keys[$source_index]}/source_F.pt" \
  "source/uda/office-home/${domain_keys[$source_index]}/source_B.pt" \
  "source/uda/office-home/${domain_keys[$source_index]}/source_C.pt"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing ${task} input: $required_path" >&2
    exit 1
  fi
done

if [ ! -f "$dcm_state" ]; then
  if [ -d "$dcm_run_dir" ] && compgen -G "$dcm_run_dir/*.txt" > /dev/null; then
    echo "Partial DCM run exists but state is missing: $dcm_run_dir" >&2
    echo "Move that partial directory aside before rebuilding" >&2
    exit 1
  fi
  echo "==> [${task}/${variant}] Building 15-epoch DCM stage: ${dcm_tag}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/duet_delayed_agreement_credit.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$dcm_method" \
    SETTING.S "$source_index" SETTING.T "$target_index" \
    SETTING.SEED "$experiment_seed" \
    ACTIVE.ADAPTATION_LIST "" \
    DUET_CONSENSUS.CREDIT_MODE "$dcm_credit_mode" \
    DUET_CONSENSUS.FEEDBACK_MODE "$dcm_feedback_mode" \
    DUET_CONSENSUS.CREDIT_DECAY "$dcm_decay"
fi

if [ ! -f "$dcm_state" ]; then
  echo "DCM stage did not produce: $dcm_state" >&2
  exit 1
fi

if [ -f "${dcm_run_dir}/target_F.pt" ] \
  && [ -f "${dcm_run_dir}/target_B.pt" ] \
  && [ -f "${dcm_run_dir}/target_C.pt" ]; then
  dcm_weight_f="${dcm_run_dir}/target_F.pt"
  dcm_weight_b="${dcm_run_dir}/target_B.pt"
  dcm_weight_c="${dcm_run_dir}/target_C.pt"
  dcm_weight_origin="dcm_run"
elif [ "$dcm_tag" = "full" ] \
  && [ -f "${legacy_handoff_dir}/source_F.pt" ] \
  && [ -f "${legacy_handoff_dir}/source_B.pt" ] \
  && [ -f "${legacy_handoff_dir}/source_C.pt" ]; then
  dcm_weight_f="${legacy_handoff_dir}/source_F.pt"
  dcm_weight_b="${legacy_handoff_dir}/source_B.pt"
  dcm_weight_c="${legacy_handoff_dir}/source_C.pt"
  dcm_weight_origin="preserved_full_dcm_copy"
else
  echo "Missing ${task}/${variant} DCM F/B/C weights: ${dcm_run_dir}/target_{F,B,C}.pt" >&2
  exit 1
fi

target_samples=$(wc -l < "$target_list" | tr -d ' ')
python - "$dcm_state" "$target_samples" "$task" "$variant" <<'PY'
import sys

import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
expected_samples = int(sys.argv[2])
task, variant = sys.argv[3:5]
memory = state.get("memory")
if not isinstance(memory, torch.Tensor) or tuple(memory.shape) != (expected_samples, 65):
    raise SystemExit(
        f"{task}/{variant}: invalid memory shape={getattr(memory, 'shape', None)}, "
        f"expected=({expected_samples}, 65)"
    )
if not torch.isfinite(memory).all():
    raise SystemExit(f"{task}/{variant}: DCM memory contains non-finite values")
print(f"==> [{task}/{variant}] Verified state: samples={memory.shape[0]}; classes={memory.shape[1]}")
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

echo "==> DCR-SFDA ablation contract: task=${task}; variant=${variant}; seed=${experiment_seed}"
echo "==> DCM: tag=${dcm_tag}; credit=${dcm_credit_mode}; feedback=${dcm_feedback_mode}; decay=${dcm_decay}; weight_origin=${dcm_weight_origin}"
echo "==> CLM: memory_write_mode=${memory_write_mode}"
echo "==> ARG: soft_replacement_mode=${soft_replacement_mode}"
echo "==> Controls: conflict_hard=0; cumulative_agreement=${cumulative_agreement}; freeze_vlm=${freeze_vlm}; passes=31; target_gt_affects_training=False"

python image_target_of_oh_vs.py \
  --cfg cfgs/office-home/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC "$handoff_source" \
  MODEL.METHOD "$method_name" \
  SETTING.S "$source_index" SETTING.T "$target_index" \
  SETTING.SEED "$experiment_seed" \
  TEST.MAX_EPOCH 4 TEST.INTERVAL 4 \
  ACTIVE.CYCLE 4 ACTIVE.ADAPTATION_LIST "" \
  DUET_HANDOFF.FINAL_EXTRA_EPOCHS 0 \
  DUET_HANDOFF.CREDIT_PRESERVING True \
  DUET_HANDOFF.STATE_PATH "$dcm_state" \
  DUET_HANDOFF.CONFLICT_HARD_FRACTION 0.0 \
  DUET_HANDOFF.FREEZE_CLIP "$freeze_vlm" \
  DUET_HANDOFF.SOFT_REPLACEMENT_MODE "$soft_replacement_mode" \
  DUET_HANDOFF.MEMORY_WRITE_MODE "$memory_write_mode" \
  DUET_HANDOFF.CUMULATIVE_AGREEMENT_MASK "$cumulative_agreement" \
  DUET_HANDOFF.CREDIT_MODE "$dcm_credit_mode" \
  DUET_HANDOFF.FEEDBACK_MODE "$dcm_feedback_mode" \
  DUET_HANDOFF.CREDIT_DECAY "$dcm_decay"

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected one ${task}/${variant} log in ${run_dir}, found ${#logs[@]}" >&2
  exit 1
fi
latest_log="${logs[0]}"
if [ "$(grep -c "Task: ${task}" "$latest_log")" -ne 16 ]; then
  echo "${task}/${variant} did not complete 4 cycles x 4 epochs" >&2
  exit 1
fi
if ! grep -q "DAC credit residual KL: cycle=4" "$latest_log"; then
  echo "${task}/${variant} did not execute the ablation path through cycle 4" >&2
  exit 1
fi
for checkpoint in target_F.pt target_B.pt target_C.pt refined_credit_state.pt; do
  if [ ! -f "${run_dir}/${checkpoint}" ]; then
    echo "Missing ${task}/${variant} checkpoint: ${run_dir}/${checkpoint}" >&2
    exit 1
  fi
done

echo "==> [${task}/${variant}] Final fixed checkpoint"
grep "Cycle: 4/4" "$latest_log" | tail -n 1
echo "==> [${task}/${variant}] Full log: ${latest_log}"
