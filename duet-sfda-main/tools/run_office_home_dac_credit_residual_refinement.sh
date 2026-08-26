#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Surgical DAC anti-erasure experiment.
# Keeps released DUET's CLIP update and cumulative agreement curriculum.
# Adds no conflict hard labels.  Only unresolved conflicts for which DAC
# history prefers Task over CLIP receive a replacement KL target.
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

dac_method="duet_delayed_agreement_credit_office_home_full_seed${experiment_seed}"
dac_run_dir="output/uda/office-home/${task}/${dac_method}"
dac_state="${dac_run_dir}/delayed_credit_state.pt"
legacy_handoff_dir="output/dac_duet_handoff_uniform5_office_home_seed${experiment_seed}_${task}/uda/office-home/${domain_keys[$source_index]}"
handoff_source="output/dac_credit_residual_office_home_seed${experiment_seed}_${task}"
handoff_source_dir="${handoff_source}/uda/office-home/${domain_keys[$source_index]}"
method_name="plmatch_dac_handoff_credit_residual_office_home_full_seed${experiment_seed}"
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

if [ ! -f "$dac_state" ]; then
  if [ -d "$dac_run_dir" ] && compgen -G "$dac_run_dir/*.txt" > /dev/null; then
    echo "Partial DAC run exists but delayed_credit_state.pt is missing: $dac_run_dir" >&2
    echo "Move or delete that partial directory before rebuilding ${task} DAC" >&2
    exit 1
  fi
  echo "==> [${task}] DAC state missing; rebuilding full-data DAC for 15 epochs"
  python image_target_of_oh_vs.py \
    --cfg cfgs/office-home/duet_delayed_agreement_credit.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$dac_method" \
    SETTING.S "$source_index" SETTING.T "$target_index" \
    SETTING.SEED "$experiment_seed" \
    ACTIVE.ADAPTATION_LIST ""
fi

if [ ! -f "$dac_state" ]; then
  echo "${task} DAC rebuild did not produce: $dac_state" >&2
  exit 1
fi

# The earlier uniform handoff copied the fixed DAC F/B/C checkpoint into a
# source-shaped directory.  Some cloud cleanups retained that copy while
# removing target_F/B/C from the original DAC directory.  Both locations are
# byte-equivalent DAC-15 weights; never fall back to a post-DUET target model.
if [ -f "${dac_run_dir}/target_F.pt" ] \
  && [ -f "${dac_run_dir}/target_B.pt" ] \
  && [ -f "${dac_run_dir}/target_C.pt" ]; then
  dac_weight_f="${dac_run_dir}/target_F.pt"
  dac_weight_b="${dac_run_dir}/target_B.pt"
  dac_weight_c="${dac_run_dir}/target_C.pt"
  dac_weight_origin="original_dac_run"
elif [ -f "${legacy_handoff_dir}/source_F.pt" ] \
  && [ -f "${legacy_handoff_dir}/source_B.pt" ] \
  && [ -f "${legacy_handoff_dir}/source_C.pt" ]; then
  dac_weight_f="${legacy_handoff_dir}/source_F.pt"
  dac_weight_b="${legacy_handoff_dir}/source_B.pt"
  dac_weight_c="${legacy_handoff_dir}/source_C.pt"
  dac_weight_origin="preserved_uniform_handoff_copy"
else
  echo "Missing ${task} DAC F/B/C weights in both supported locations:" >&2
  echo "  ${dac_run_dir}/target_{F,B,C}.pt" >&2
  echo "  ${legacy_handoff_dir}/source_{F,B,C}.pt" >&2
  exit 1
fi

target_samples=$(wc -l < "$target_list" | tr -d ' ')
python - "$dac_state" "$target_samples" "$task" <<'PY'
import sys

import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
expected_samples = int(sys.argv[2])
task = sys.argv[3]
memory = state.get("memory")
if not isinstance(memory, torch.Tensor) or tuple(memory.shape) != (expected_samples, 65):
    raise SystemExit(
        f"{task}: invalid DAC memory: shape={getattr(memory, 'shape', None)}, "
        f"expected=({expected_samples}, 65)"
    )
if not torch.isfinite(memory).all():
    raise SystemExit(f"{task}: DAC memory contains non-finite values")
print(f"==> [{task}] Verified DAC state: samples={memory.shape[0]}; classes={memory.shape[1]}")
PY

if [ -d "$run_dir" ] && compgen -G "$run_dir/*.txt" > /dev/null; then
  echo "Refusing to mix logs with an existing run: $run_dir" >&2
  echo "Move the existing directory aside before rerunning" >&2
  exit 1
fi

mkdir -p "$handoff_source_dir"
cp -f "$dac_weight_f" "${handoff_source_dir}/source_F.pt"
cp -f "$dac_weight_b" "${handoff_source_dir}/source_B.pt"
cp -f "$dac_weight_c" "${handoff_source_dir}/source_C.pt"

echo "==> [${task}] DAC-15 ready; weight_origin=${dac_weight_origin}"
echo "==> [${task}] Released DUET retained: adaptive CLIP + cumulative agreements"
echo "==> [${task}] New residual: DAC soft correction only when history prefers Task"
echo "==> [${task}] Conflict hard admission: 0%; total target passes: 31"

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
  DUET_HANDOFF.STATE_PATH "$dac_state" \
  DUET_HANDOFF.CONFLICT_HARD_FRACTION 0.0 \
  DUET_HANDOFF.FREEZE_CLIP False \
  DUET_HANDOFF.SOFT_REPLACEMENT_MODE task_supported \
  DUET_HANDOFF.MEMORY_WRITE_MODE locked \
  DUET_HANDOFF.CUMULATIVE_AGREEMENT_MASK True \
  DUET_HANDOFF.CREDIT_DECAY 0.9 \
  DUET_HANDOFF.CREDIT_ETA 4.0 \
  DUET_HANDOFF.MEMORY_UPDATE_RATE 0.5 \
  DUET_HANDOFF.CREDIT_MODE delayed \
  DUET_HANDOFF.FEEDBACK_MODE agreement_temporal

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
if ! grep -q "DAC credit residual KL: cycle=4" "$latest_log"; then
  echo "DAC residual was not active through cycle 4" >&2
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
