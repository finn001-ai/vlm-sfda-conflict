#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Usage:
#   bash tools/run_office_home_dac_credit_preserving_refinement.sh 2020 AC
#
# Reuse one completed Office-Home DAC-15 checkpoint and run the same
# credit-preserving 4-cycle refinement used by the VisDA entry.  No setting is
# selected with target ground truth.
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
handoff_source="output/dac_credit_preserving_office_home_seed${experiment_seed}_${task}"
handoff_source_dir="${handoff_source}/uda/office-home/${domain_keys[$source_index]}"
method_name="plmatch_dac_handoff_credit_preserving_office_home_full_seed${experiment_seed}"
run_dir="output/uda/office-home/${task}/${method_name}"
target_list="data/office-home/${domain_names[$target_index]}_list.txt"

for required_path in \
  "$target_list" \
  data/office-home/classname.txt \
  "${dac_run_dir}/target_F.pt" \
  "${dac_run_dir}/target_B.pt" \
  "${dac_run_dir}/target_C.pt" \
  "${dac_state}"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing ${task} DAC input: $required_path" >&2
    echo "This experiment reuses the completed Office-Home DAC-15 checkpoint" >&2
    exit 1
  fi
done

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
        f"{task}: invalid full-data Office-Home DAC memory: "
        f"shape={getattr(memory, 'shape', None)}, expected=({expected_samples}, 65)"
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
cp -f "${dac_run_dir}/target_F.pt" "${handoff_source_dir}/source_F.pt"
cp -f "${dac_run_dir}/target_B.pt" "${handoff_source_dir}/source_B.pt"
cp -f "${dac_run_dir}/target_C.pt" "${handoff_source_dir}/source_C.pt"

echo "==> [${task}] Stage 1 reused: full-data DAC, 15 epochs"
echo "==> [${task}] Stage 2: credit-preserving refinement, 4 cycles x 4 epochs"
echo "==> [${task}] Conflict soft coverage: 100%; hard rank coverage: 80%"
echo "==> [${task}] Total target passes: 31; target GT affects training: False"

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
  DUET_HANDOFF.CONFLICT_HARD_FRACTION 0.8 \
  DUET_HANDOFF.FREEZE_CLIP True

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected one ${task} refinement log in ${run_dir}, found ${#logs[@]}" >&2
  exit 1
fi
latest_log="${logs[0]}"
if [ "$(grep -c "Task: ${task}" "$latest_log")" -ne 16 ]; then
  echo "${task} did not complete 4 cycles x 4 logged epochs" >&2
  exit 1
fi
if ! grep -q "DAC credit-preserving teacher: cycle=4" "$latest_log"; then
  echo "Credit-preserving teacher was not active through cycle 4" >&2
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
