#!/usr/bin/env bash
set -euo pipefail

# VisDA-C matched test：
#   control   = Stage14 temporal-precision target head
#   candidate = 同一宿主 + Boundary-Flip
#
# 默认直接删除并覆盖本脚本对应的输出，不再要求用户手动移动残留目录。

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

seed="${SEED:-2020}"
run_control="${RUN_CONTROL:-1}"
adaptation_list="${ADAPTATION_LIST:-}"
control_method="temporal_precision_head_control_visda_seed${seed}"
candidate_method="boundary_flip_duet_visda_seed${seed}"
result_dir="output/uda/VISDA-C/boundary_flip_duet_seed${seed}"
control_dir="output/uda/VISDA-C/TV/${control_method}"
candidate_dir="output/uda/VISDA-C/TV/${candidate_method}"

if [ "$run_control" != "0" ] && [ "$run_control" != "1" ]; then
  echo "RUN_CONTROL must be 0 or 1" >&2
  exit 1
fi
case "$seed" in
  ""|*[!0-9]*)
    echo "SEED must be a non-negative integer" >&2
    exit 1
    ;;
esac

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$path" ]; then
    echo "Missing VisDA-C input: $path" >&2
    exit 1
  fi
done

adaptation_opts=()
evaluation_list="data/VISDA-C/validation_list.txt"
if [ -n "$adaptation_list" ]; then
  if [ ! -f "$adaptation_list" ]; then
    echo "ADAPTATION_LIST does not exist: $adaptation_list" >&2
    exit 1
  fi
  adaptation_opts+=(DCCL.ADAPTATION_LIST "$adaptation_list")
  evaluation_list="$adaptation_list"
fi

mkdir -p "$result_dir"
sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt \
  > "${result_dir}/source_sha256.txt"
sha256sum "$evaluation_list" \
  > "${result_dir}/adaptation_list_sha256.txt"

if [ "$run_control" = "1" ]; then
  echo "==> Overwriting VisDA-C Stage14 control: ${control_dir}"
  rm -rf -- "$control_dir"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/temporal_precision_head.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source SETTING.SEED "$seed" \
    SETTING.S 0 SETTING.T 1 \
    MODEL.METHOD "$control_method" \
    "${adaptation_opts[@]}"
  python tools/extract_final_accuracy.py \
    --glob "${control_dir}/*.txt" \
    > "${result_dir}/control_accuracy.csv"
fi

echo "==> Overwriting VisDA-C Boundary-Flip candidate: ${candidate_dir}"
rm -rf -- "$candidate_dir"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/boundary_flip_duet.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source SETTING.SEED "$seed" \
  SETTING.S 0 SETTING.T 1 \
  MODEL.METHOD "$candidate_method" \
  "${adaptation_opts[@]}"

python tools/extract_final_accuracy.py \
  --glob "${candidate_dir}/*.txt" \
  > "${result_dir}/candidate_accuracy.csv"

python tools/analyze_visda_boundary_flip_duet.py \
  --candidate-csv "${result_dir}/candidate_accuracy.csv" \
  --control-csv "${result_dir}/control_accuracy.csv" \
  --diagnostics-glob \
    "${candidate_dir}/temporal_diagnostics/*_cycle*.npz" \
  --log-glob "${candidate_dir}/*.txt" \
  --output "${result_dir}/gate.json"

echo "Candidate accuracy: ${result_dir}/candidate_accuracy.csv"
echo "Gate: ${result_dir}/gate.json"
