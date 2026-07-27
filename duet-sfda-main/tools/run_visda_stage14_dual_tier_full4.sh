#!/usr/bin/env bash
set -euo pipefail

# 本脚本是“Stage14 + Pending 弱监督”的固定复现实验入口。
# Stage14 基线 YAML 保持 PL_MEMORY=stable；以下命令行参数只在本实验中覆盖为
# dual_tier，避免把 Pending 消融混入原始 Stage14 存档。
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

result_dir="output/uda/VISDA-C/stage14_dual_tier_full4_seed2020"
control_method="plmatch_stage14_prior_memory_full4_control_seed2020"
both_stable_method="temporal_precision_head_stage14_bothprior_stable_full4_seed2020"
none_stable_method="temporal_precision_head_stage14_none_stable_full4_seed2020"
both_monotonic_method="temporal_precision_head_stage14_bothprior_monotonic_full4_seed2020"
none_monotonic_method="temporal_precision_head_stage14_none_monotonic_full4_seed2020"
both_dual_method="temporal_precision_head_stage14_bothprior_dualtier_full4_seed2020"
none_dual_method="temporal_precision_head_stage14_none_dualtier_full4_seed2020"

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

sample_count=$(wc -l < data/VISDA-C/validation_list.txt)
if [ "$sample_count" -ne 55388 ]; then
  echo "Expected 55388 full VisDA validation samples, found ${sample_count}" >&2
  exit 1
fi

mkdir -p "$result_dir"
sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt \
  > "$result_dir/source_sha256.txt"
sha256sum data/VISDA-C/validation_list.txt \
  > "$result_dir/validation_list_sha256.txt"

find_one_log() {
  method=$1
  pattern="output/uda/VISDA-C/TV/${method}/*.txt"
  logs=()
  while IFS= read -r path; do
    logs+=("$path")
  done < <(compgen -G "$pattern" || true)
  if [ "${#logs[@]}" -ne 1 ]; then
    echo "${method}: expected exactly one log, found ${#logs[@]}" >&2
    return 1
  fi
  printf '%s\n' "${logs[0]}"
}

validate_run() {
  method=$1
  log=$(find_one_log "$method") || return 1
  checkpoints=$(grep -c "Task: TV" "$log" || true)
  refreshes=$(
    grep -c "Number of valid pseudo-labeled samples" "$log" || true
  )
  if [ "$checkpoints" -ne 16 ] || [ "$refreshes" -ne 4 ]; then
    echo "${method}: incomplete run (${checkpoints}/16 checkpoints; ${refreshes}/4 refreshes)" >&2
    return 1
  fi
  if ! grep -q "Cycle: 4/4" "$log"; then
    echo "${method}: log is not a four-cycle run" >&2
    return 1
  fi
  if ! grep -q \
    "Number of valid pseudo-labeled samples: [0-9]*/55388" \
    "$log"; then
    echo "${method}: log does not prove full-data VisDA adaptation" >&2
    return 1
  fi
}

validate_reference() {
  method=$1
  description=$2
  if ! validate_run "$method"; then
    echo "Missing completed ${description}." >&2
    echo "First run: bash tools/run_visda_stage14_prior_memory_full4.sh" >&2
    exit 1
  fi
}

validate_dual_tier() {
  method=$1
  validate_run "$method" || return 1
  log=$(find_one_log "$method")
  memory_records=$(
    grep -c "DCCL pseudo-label memory: mode=dual_tier" "$log" || true
  )
  if [ "$memory_records" -ne 4 ]; then
    echo "${method}: expected four dual-tier memory records, found ${memory_records}" >&2
    return 1
  fi
  if ! grep -q "PL_PENDING_WEIGHT: 0.5" "$log"; then
    echo "${method}: log does not contain the fixed pending weight 0.5" >&2
    return 1
  fi
}

run_dual_tier() {
  method=$1
  calib_mode=$2
  description=$3
  if compgen -G "output/uda/VISDA-C/TV/${method}/*" >/dev/null; then
    if validate_dual_tier "$method"; then
      echo "==> Reusing ${description}"
      return
    fi
    echo "${method}: an incomplete or invalid output already exists." >&2
    echo "Move that method output directory aside, then rerun." >&2
    exit 1
  fi

  echo "==> Running ${description}"
  # Pending 的实际逐样本 CE 权重为 0.5 * mix_conf；
  # Conflict 的 hard CE 权重保持为 0。
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/temporal_precision_head.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$method" \
    SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 \
    ACTIVE.CYCLE 4 \
    ACTIVE.CLS_PAR 0.4 \
    ACTIVE.CON_PAR 0.2 \
    ACTIVE.KL_PAR 0.4 \
    DCCL.ADAPTATION_LIST "" \
    DCCL.CALIB_MODE "$calib_mode" \
    DCCL.CALIB_POWER 0.5 \
    DCCL.PL_MEMORY dual_tier \
    DCCL.PL_STABLE_CYCLES 2 \
    DCCL.PL_STABLE_MEMORY reversible \
    DCCL.PL_MEMORY_WARMUP_CYCLES 1 \
    DCCL.PL_MEMORY_MIN_CONF 0.0 \
    DCCL.PL_PENDING_WEIGHT 0.5 \
    DCCL.PL_EXPAND none \
    DCCL.PL_TOPK_PER_CLASS 0 \
    DCCL.PL_CLASS_BALANCE False \
    DCCL.TARGET_HEAD_ADAPT True \
    DCCL.GTR_PAR 0.05 \
    DCCL.TEMPORAL_DIAG True
  validate_dual_tier "$method"
}

validate_reference "$control_method" "DUET control"
validate_reference "$both_stable_method" "both_prior + stable reference"
validate_reference "$none_stable_method" "none + stable reference"
validate_reference "$both_monotonic_method" "both_prior + monotonic reference"
validate_reference "$none_monotonic_method" "none + monotonic reference"

run_dual_tier \
  "$both_dual_method" both_prior \
  "Stage14 dual-tier memory (both_prior, full data, seed 2020)"
run_dual_tier \
  "$none_dual_method" none \
  "Stage14 dual-tier memory (no prior, full data, seed 2020)"

python tools/summarize_visda_stage14_dual_tier.py \
  --duet-glob "output/uda/VISDA-C/TV/${control_method}/*.txt" \
  --both-prior-stable-glob \
    "output/uda/VISDA-C/TV/${both_stable_method}/*.txt" \
  --none-stable-glob \
    "output/uda/VISDA-C/TV/${none_stable_method}/*.txt" \
  --both-prior-monotonic-glob \
    "output/uda/VISDA-C/TV/${both_monotonic_method}/*.txt" \
  --none-monotonic-glob \
    "output/uda/VISDA-C/TV/${none_monotonic_method}/*.txt" \
  --both-prior-dual-tier-glob \
    "output/uda/VISDA-C/TV/${both_dual_method}/*.txt" \
  --none-dual-tier-glob \
    "output/uda/VISDA-C/TV/${none_dual_method}/*.txt" \
  --class-names data/VISDA-C/classname.txt \
  --out "$result_dir/dual_tier_summary.json" \
  --csv-out "$result_dir/per_class_dual_tier.csv" \
  --min-memory-gain 0.10 \
  --min-duet-gain 0.10

echo "==> Stage14 dual-tier full-data experiment complete"
echo "Summary: $result_dir/dual_tier_summary.json"
echo "Per-class table: $result_dir/per_class_dual_tier.csv"
echo "No 8-cycle run, seed sweep, or third-party visual module was started."
