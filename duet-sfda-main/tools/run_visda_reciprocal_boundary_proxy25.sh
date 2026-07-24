#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
result_dir="output/uda/VISDA-C"
control_method="plmatch_reciprocal_boundary_proxy25_control_seed2020"
host_method="reciprocal_boundary_proxy25_host_control_seed2020"
margin_method="reciprocal_boundary_proxy25_margin_seed2020"
consistency_method="reciprocal_boundary_proxy25_margin_consistency_seed2020"
full_method="reciprocal_boundary_proxy25_full_seed2020"

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

expected_proxy=$(mktemp)
trap 'rm -f "$expected_proxy"' EXIT
python tools/prepare_visda_proxy_subset.py \
  --input data/VISDA-C/validation_list.txt \
  --output "$expected_proxy" \
  --ratio 0.25 \
  --seed 2020 \
  --force > /dev/null
if [ ! -f "$proxy_list" ]; then
  python tools/prepare_visda_proxy_subset.py \
    --input data/VISDA-C/validation_list.txt \
    --output "$proxy_list" \
    --ratio 0.25 \
    --seed 2020
fi
if ! cmp -s "$expected_proxy" "$proxy_list"; then
  echo "VisDA proxy list is not the deterministic ratio=0.25 seed=2020 split" >&2
  exit 1
fi

validate_run() {
  method=$1
  pattern="output/uda/VISDA-C/TV/${method}/*.txt"
  logs=()
  while IFS= read -r path; do
    logs+=("$path")
  done < <(compgen -G "$pattern" || true)
  if [ "${#logs[@]}" -eq 0 ]; then
    return 1
  fi
  if [ "${#logs[@]}" -ne 1 ]; then
    echo "${method}: expected exactly one log, found ${#logs[@]}" >&2
    exit 1
  fi
  checkpoints=$(grep -c "Task: TV" "${logs[0]}" || true)
  if [ "$checkpoints" -ne 16 ]; then
    echo "${method}: incomplete existing run (${checkpoints}/16); refusing to mix runs" >&2
    exit 1
  fi
  return 0
}

if validate_run "$control_method"; then
  echo "==> Reusing matched official-DUET VisDA proxy control"
else
  echo "==> Running matched official-DUET VisDA proxy control"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/plmatch.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$control_method" \
    SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 \
    ACTIVE.CYCLE 4 \
    ACTIVE.ADAPTATION_LIST "$proxy_list"
  validate_run "$control_method"
fi

run_candidate() {
  method=$1
  enabled=$2
  consistency_weight=$3
  keep_weight=$4
  description=$5
  if validate_run "$method"; then
    echo "==> Reusing ${description}"
    return
  fi
  echo "==> Running ${description}"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/reciprocal_boundary.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$method" \
    SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 \
    ACTIVE.CYCLE 4 \
    DCCL.ADAPTATION_LIST "$proxy_list" \
    DCCL.RECIPROCAL_BOUNDARY "$enabled" \
    DCCL.BOUNDARY_CONSISTENCY_PAR "$consistency_weight" \
    DCCL.BOUNDARY_KEEP_PAR "$keep_weight"
  validate_run "$method"
}

run_candidate "$host_method" False 0.0 0.0 \
  "boundary-disabled DCCL host parity control"
run_candidate "$margin_method" True 0.0 0.0 \
  "reciprocal-boundary margin-only ablation"
run_candidate "$consistency_method" True 0.05 0.0 \
  "reciprocal-boundary margin + pair-consistency ablation"
run_candidate "$full_method" True 0.05 0.05 \
  "full reciprocal-boundary method"

mkdir -p "$result_dir"
sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt \
  > "$result_dir/reciprocal_boundary_proxy25_source_sha256.txt"
sha256sum "$proxy_list" \
  > "$result_dir/reciprocal_boundary_proxy25_list_sha256.txt"

python tools/summarize_reciprocal_boundary_preflight.py visda \
  --control-glob "output/uda/VISDA-C/TV/${control_method}/*.txt" \
  --host-glob "output/uda/VISDA-C/TV/${host_method}/*.txt" \
  --margin-glob "output/uda/VISDA-C/TV/${margin_method}/*.txt" \
  --consistency-glob "output/uda/VISDA-C/TV/${consistency_method}/*.txt" \
  --full-glob "output/uda/VISDA-C/TV/${full_method}/*.txt" \
  --out "$result_dir/reciprocal_boundary_proxy25_gate.json" \
  --csv-out "$result_dir/reciprocal_boundary_proxy25_ablation.csv"

echo "==> VisDA reciprocal-boundary proxy ablation complete"
echo "Gate: $result_dir/reciprocal_boundary_proxy25_gate.json"
echo "Ablation: $result_dir/reciprocal_boundary_proxy25_ablation.csv"
