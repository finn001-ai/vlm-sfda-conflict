#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Full-data VisDA-C evaluation of the Office-Home-locked residual method.
# Reuses DAC-15, keeps released DUET's adaptive CLIP and cumulative agreement
# curriculum, and applies no conflict hard labels.
experiment_seed="${1:-2020}"
dac_method="duet_delayed_agreement_credit_visda_full_seed${experiment_seed}"
dac_run_dir="output/uda/VISDA-C/TV/${dac_method}"
dac_state="${dac_run_dir}/delayed_credit_state.pt"
handoff_source="output/dac_credit_residual_visda_source_seed${experiment_seed}"
handoff_source_dir="${handoff_source}/uda/VISDA-C/T"
method_name="plmatch_dac_handoff_credit_residual_visda_full_seed${experiment_seed}"
run_dir="output/uda/VISDA-C/TV/${method_name}"

for required_path in \
  data/VISDA-C/train_list.txt \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  "$dac_state"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing full-data VisDA-C DAC input: $required_path" >&2
    echo "This entry reuses the completed VisDA DAC-15 run" >&2
    exit 1
  fi
done

full_samples=$(wc -l < data/VISDA-C/validation_list.txt | tr -d ' ')
if [ "$full_samples" -ne 55388 ]; then
  echo "Unexpected VisDA-C full target size: ${full_samples}; expected 55388" >&2
  exit 1
fi

# Prefer the original DAC target weights.  If cloud cleanup removed them,
# recover only from a source-shaped copy made before any DUET optimization.
dac_weight_f=""
dac_weight_b=""
dac_weight_c=""
dac_weight_origin=""
if [ -f "${dac_run_dir}/target_F.pt" ] \
  && [ -f "${dac_run_dir}/target_B.pt" ] \
  && [ -f "${dac_run_dir}/target_C.pt" ]; then
  dac_weight_f="${dac_run_dir}/target_F.pt"
  dac_weight_b="${dac_run_dir}/target_B.pt"
  dac_weight_c="${dac_run_dir}/target_C.pt"
  dac_weight_origin="original_dac_run"
else
  for preserved_dir in \
    "output/dac_duet_handoff_source_seed${experiment_seed}/uda/VISDA-C/T" \
    "output/dac_duet_handoff_uniform5_source_seed${experiment_seed}/uda/VISDA-C/T"; do
    if [ -f "${preserved_dir}/source_F.pt" ] \
      && [ -f "${preserved_dir}/source_B.pt" ] \
      && [ -f "${preserved_dir}/source_C.pt" ]; then
      dac_weight_f="${preserved_dir}/source_F.pt"
      dac_weight_b="${preserved_dir}/source_B.pt"
      dac_weight_c="${preserved_dir}/source_C.pt"
      dac_weight_origin="preserved_pre_duet_handoff_copy"
      break
    fi
  done
fi
if [ -z "$dac_weight_f" ]; then
  echo "Missing VisDA DAC F/B/C weights in the original run and preserved handoff copies" >&2
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
        "Invalid VisDA DAC memory: "
        f"shape={getattr(memory, 'shape', None)}, expected=({expected_samples}, 12)"
    )
if not torch.isfinite(memory).all():
    raise SystemExit("VisDA DAC memory contains non-finite values")
print(f"==> Verified VisDA DAC state: samples={memory.shape[0]}; classes={memory.shape[1]}")
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

echo "==> Reusing VisDA DAC-15; weight_origin=${dac_weight_origin}"
echo "==> Locked Office-Home residual: adaptive CLIP + cumulative agreements"
echo "==> DAC soft correction only when history prefers Task"
echo "==> Conflict hard admission: 0%; refinement: 4 cycles x 4 epochs"
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
  DUET_HANDOFF.CONFLICT_HARD_FRACTION 0.0 \
  DUET_HANDOFF.FREEZE_CLIP False \
  DUET_HANDOFF.SOFT_REPLACEMENT_MODE task_supported \
  DUET_HANDOFF.CUMULATIVE_AGREEMENT_MASK True

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected one VisDA residual log in ${run_dir}, found ${#logs[@]}" >&2
  exit 1
fi
latest_log="${logs[0]}"
if [ "$(grep -c "Task: TV" "$latest_log")" -ne 16 ]; then
  echo "VisDA residual did not complete 4 cycles x 4 logged epochs" >&2
  exit 1
fi
if ! grep -q "DAC credit residual KL: cycle=4" "$latest_log"; then
  echo "DAC residual was not active through cycle 4" >&2
  exit 1
fi
for checkpoint in target_F.pt target_B.pt target_C.pt refined_credit_state.pt; do
  if [ ! -f "${run_dir}/${checkpoint}" ]; then
    echo "Missing final VisDA checkpoint: ${run_dir}/${checkpoint}" >&2
    exit 1
  fi
done

echo "==> VisDA fixed final checkpoint"
grep "Cycle: 4/4" "$latest_log" | tail -n 1
echo "==> Same-budget DAC15+DUET16 reference: 91.54%"
echo "==> Full log: ${latest_log}"
