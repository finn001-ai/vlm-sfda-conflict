#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# DCR-SFDA on one DomainNet-126 transfer task.
# Usage: DATA_DIR=/path/to/data bash tools/run_domainnet126_dcr.sh 2020 CP
experiment_seed="${1:-2020}"
task="${2:-CP}"
domain_keys=(C P R S)
domain_names=(clipart painting real sketch)

case "$task" in
  CP) source_index=0; target_index=1 ;;
  CR) source_index=0; target_index=2 ;;
  CS) source_index=0; target_index=3 ;;
  PC) source_index=1; target_index=0 ;;
  PR) source_index=1; target_index=2 ;;
  PS) source_index=1; target_index=3 ;;
  RC) source_index=2; target_index=0 ;;
  RP) source_index=2; target_index=1 ;;
  RS) source_index=2; target_index=3 ;;
  SC) source_index=3; target_index=0 ;;
  SP) source_index=3; target_index=1 ;;
  SR) source_index=3; target_index=2 ;;
  *)
    echo "Task must be one of: CP CR CS PC PR PS RC RP RS SC SP SR" >&2
    exit 1
    ;;
esac

data_root="${DATA_DIR:-/home/sfda/data}"
target_list="data/domainnet126/${domain_names[$target_index]}_list.txt"
source_checkpoint="source/uda/domainnet126/${domain_keys[$source_index]}/best_${domain_names[$source_index]}_2020.pth"
dcm_method="dcr_memory_domainnet126_seed${experiment_seed}"
dcm_run_dir="output/uda/domainnet126/${task}/${dcm_method}"
dcm_state="${dcm_run_dir}/dcr_memory_state.pt"
legacy_dcm_run_dir="output/uda/domainnet126/${task}/duet_delayed_agreement_credit_dcr_sfda_domainnet126_seed${experiment_seed}"
legacy_dcm_state="${legacy_dcm_run_dir}/delayed_credit_state.pt"
handoff_source="output/dcr_domainnet126_seed${experiment_seed}_${task}"
handoff_source_dir="${handoff_source}/uda/domainnet126/${domain_keys[$source_index]}"
method_name="dcr_domainnet126_seed${experiment_seed}"
run_dir="output/uda/domainnet126/${task}/${method_name}"

for required_path in \
  "$target_list" \
  data/domainnet126/classname.txt \
  "$source_checkpoint" \
  cfgs/domainnet126/dcr.yaml; do
  if [ ! -f "$required_path" ]; then
    echo "Missing ${task} input: $required_path" >&2
    exit 1
  fi
done

python - "$target_list" "$data_root" "$task" <<'PY'
import sys
from pathlib import Path

list_path, data_root, task = sys.argv[1:4]
rows = Path(list_path).read_text().splitlines()
if not rows:
    raise SystemExit(f"{task}: empty target list: {list_path}")
relative = rows[0].rsplit(maxsplit=1)[0]
image = Path(relative)
if not image.is_absolute():
    image = Path(data_root) / "domainnet126" / image
if not image.is_file():
    raise SystemExit(
        f"{task}: first target image is missing: {image}\n"
        "Set DATA_DIR to the directory that contains domainnet126/."
    )
print(f"==> [{task}] Data root verified: {Path(data_root) / 'domainnet126'}")
PY

data_override=()
if [ -n "${DATA_DIR:-}" ]; then
  data_override=(DATA_DIR "$DATA_DIR")
fi

if [ -f "$dcm_state" ]; then
  echo "==> [${task}] Reusing DCR memory: ${dcm_state}"
elif [ -f "$legacy_dcm_state" ]; then
  dcm_run_dir="$legacy_dcm_run_dir"
  dcm_state="$legacy_dcm_state"
  echo "==> [${task}] Reusing legacy memory artifact: ${dcm_state}"
else
  if [ -d "$dcm_run_dir" ] && compgen -G "$dcm_run_dir/*.txt" > /dev/null; then
    echo "Partial DomainNet DCM run exists but state is missing: $dcm_run_dir" >&2
    echo "Move that directory aside before rebuilding" >&2
    exit 1
  fi
  echo "==> [${task}] Stage 1/2: DCM, 15 epochs"
  python image_target_of_oh_vs.py \
    --cfg cfgs/domainnet126/dcr.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$dcm_method" \
    SETTING.S "$source_index" SETTING.T "$target_index" \
    SETTING.SEED "$experiment_seed" \
    ACTIVE.ADAPTATION_LIST "" \
    "${data_override[@]}"
fi

for artifact in "$dcm_state" \
  "${dcm_run_dir}/target_F.pt" \
  "${dcm_run_dir}/target_B.pt" \
  "${dcm_run_dir}/target_C.pt"; do
  if [ ! -f "$artifact" ]; then
    echo "Missing completed ${task} DCM artifact: $artifact" >&2
    exit 1
  fi
done

# `wc -l` counts newline characters and undercounts DomainNet list files whose
# final record has no trailing newline.  Count records instead so this matches
# Python's splitlines() and the number of rows used by the dataset loader.
target_samples=$(awk 'END {print NR}' "$target_list")
python - "$dcm_state" "$target_samples" "$task" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
expected = (int(sys.argv[2]), 126)
memory = state.get("memory")
if not isinstance(memory, torch.Tensor) or tuple(memory.shape) != expected:
    raise SystemExit(
        f"{sys.argv[3]}: invalid DCM memory shape={getattr(memory, 'shape', None)}, expected={expected}"
    )
if not torch.isfinite(memory).all():
    raise SystemExit(f"{sys.argv[3]}: DCM memory contains non-finite values")
print(f"==> [{sys.argv[3]}] DCM state verified: samples={expected[0]}; classes=126")
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

echo "==> [${task}] Stage 2/2: DCR residual refinement, 4 cycles x 4 epochs"
echo "==> DCM=delayed; CLM=locked; ARG=task_supported; passes=31"
echo "==> Conflict hard admission=0; cumulative agreement=True; VLM adaptive=True"
echo "==> target_gt_affects_training=False"

python image_target_of_oh_vs.py \
  --cfg cfgs/domainnet126/dcr.yaml \
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
  DCR.CREDIT_MODE delayed \
  DCR.FEEDBACK_MODE agreement_temporal \
  "${data_override[@]}"

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected one ${task} DCR log in ${run_dir}, found ${#logs[@]}" >&2
  exit 1
fi
latest_log="${logs[0]}"
if [ "$(grep -c "Task: ${task}" "$latest_log")" -ne 16 ]; then
  echo "${task} did not complete 4 cycles x 4 epochs" >&2
  exit 1
fi
if ! grep -q "DCR asymmetric residual guidance: cycle=4" "$latest_log"; then
  echo "${task} did not execute DCR through cycle 4" >&2
  exit 1
fi
for checkpoint in target_F.pt target_B.pt target_C.pt refined_credit_state.pt; do
  if [ ! -f "${run_dir}/${checkpoint}" ]; then
    echo "Missing ${task} final artifact: ${run_dir}/${checkpoint}" >&2
    exit 1
  fi
done

echo "==> [${task}] Final fixed checkpoint"
grep "Cycle: 4/4" "$latest_log" | tail -n 1
echo "==> [${task}] Full log: ${latest_log}"
