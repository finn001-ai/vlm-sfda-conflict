#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
result_dir="output/uda/VISDA-C/stage14_visda_gap_audit"
duet_method="plmatch_stage14_gap_audit_proxy25_seed2020"
stage14_method="temporal_precision_head_stage14_gap_audit_proxy25_seed2020"

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
  temporal_dir=$2
  log_pattern="output/uda/VISDA-C/TV/${method}/*.txt"
  logs=()
  while IFS= read -r path; do
    logs+=("$path")
  done < <(compgen -G "$log_pattern" || true)
  if [ "${#logs[@]}" -eq 0 ]; then
    return 1
  fi
  if [ "${#logs[@]}" -ne 1 ]; then
    echo "${method}: expected exactly one log, found ${#logs[@]}" >&2
    exit 1
  fi
  checkpoints=$(grep -c "Task: TV" "${logs[0]}" || true)
  if [ "$checkpoints" -ne 16 ]; then
    echo "${method}: incomplete run (${checkpoints}/16); refusing to mix runs" >&2
    exit 1
  fi
  final_snapshot="output/uda/VISDA-C/TV/${method}/failure_audit/final_full.npz"
  if [ ! -f "$final_snapshot" ]; then
    echo "${method}: completed log exists but final failure-audit snapshot is missing" >&2
    exit 1
  fi
  temporal_pattern="output/uda/VISDA-C/TV/${method}/${temporal_dir}/*cycle*.npz"
  temporal_paths=()
  while IFS= read -r path; do
    temporal_paths+=("$path")
  done < <(compgen -G "$temporal_pattern" || true)
  temporal_count=${#temporal_paths[@]}
  if [ "$temporal_count" -ne 4 ]; then
    echo "${method}: expected 4 temporal snapshots, found ${temporal_count}" >&2
    exit 1
  fi
  return 0
}

if validate_run "$duet_method" "failure_audit"; then
  echo "==> Reusing matched official DUET audit run"
else
  echo "==> Running matched official DUET audit control"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/plmatch.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$duet_method" \
    SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 \
    ACTIVE.CYCLE 4 \
    ACTIVE.ADAPTATION_LIST "$proxy_list" \
    FAILURE_AUDIT.ENABLED True
  validate_run "$duet_method" "failure_audit"
fi

if validate_run "$stage14_method" "temporal_diagnostics"; then
  echo "==> Reusing matched Stage14 audit run"
else
  echo "==> Running matched Stage14 audit candidate"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/temporal_precision_head.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$stage14_method" \
    SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 \
    ACTIVE.CYCLE 4 \
    DCCL.ADAPTATION_LIST "$proxy_list" \
    DCCL.TEMPORAL_DIAG True \
    FAILURE_AUDIT.ENABLED True
  validate_run "$stage14_method" "temporal_diagnostics"
fi

mkdir -p "$result_dir"
sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt \
  > "$result_dir/source_sha256.txt"
sha256sum "$proxy_list" > "$result_dir/proxy_list_sha256.txt"

python tools/analyze_visda_stage14_gap.py \
  --duet-final \
    "output/uda/VISDA-C/TV/${duet_method}/failure_audit/final_full.npz" \
  --stage14-final \
    "output/uda/VISDA-C/TV/${stage14_method}/failure_audit/final_full.npz" \
  --duet-temporal-glob \
    "output/uda/VISDA-C/TV/${duet_method}/failure_audit/pre_cycle*.npz" \
  --stage14-temporal-glob \
    "output/uda/VISDA-C/TV/${stage14_method}/temporal_diagnostics/*cycle*.npz" \
  --class-names data/VISDA-C/classname.txt \
  --out-dir "$result_dir" \
  --seed 2020

echo "==> Stage14 VisDA gap audit complete"
echo "Summary: $result_dir/stage14_visda_gap_summary.json"
echo "Per class: $result_dir/per_class.csv"
echo "Confusion pairs: $result_dir/pair_confusion_geometry.csv"
echo "t-SNE: $result_dir/stage14_visda_feature_tsne.png"
echo "This audit does not launch or approve a new method."
