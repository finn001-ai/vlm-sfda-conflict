#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

result_dir="output/uda/VISDA-C/stage14_prior_memory_full4_seed2020"
control_method="plmatch_stage14_prior_memory_full4_control_seed2020"
current_method="temporal_precision_head_stage14_bothprior_stable_full4_seed2020"
none_stable_method="temporal_precision_head_stage14_none_stable_full4_seed2020"
both_monotonic_method="temporal_precision_head_stage14_bothprior_monotonic_full4_seed2020"
none_monotonic_method="temporal_precision_head_stage14_none_monotonic_full4_seed2020"

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
  refreshes=$(
    grep -c "Number of valid pseudo-labeled samples" "${logs[0]}" || true
  )
  if [ "$checkpoints" -ne 16 ]; then
    echo "${method}: incomplete run (${checkpoints}/16 checkpoints)" >&2
    echo "Move its output directory aside, then rerun this script." >&2
    exit 1
  fi
  if [ "$refreshes" -ne 4 ]; then
    echo "${method}: incomplete run (${refreshes}/4 pseudo-label refreshes)" >&2
    exit 1
  fi
  if ! grep -q "Cycle: 4/4" "${logs[0]}"; then
    echo "${method}: log is not a four-cycle run" >&2
    exit 1
  fi
  if ! grep -q \
    "Number of valid pseudo-labeled samples: [0-9]*/55388" \
    "${logs[0]}"; then
    echo "${method}: log does not prove full-data VisDA adaptation" >&2
    exit 1
  fi
  return 0
}

run_control() {
  if validate_run "$control_method"; then
    echo "==> Reusing matched full-data DUET control"
    return
  fi
  echo "==> Phase 1/2: running full-data DUET control (seed 2020, 4 cycles)"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/plmatch.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$control_method" \
    SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 \
    ACTIVE.CYCLE 4 \
    ACTIVE.ADAPTATION_LIST ""
  validate_run "$control_method"
}

run_candidate() {
  method=$1
  calib_mode=$2
  pl_memory=$3
  description=$4
  if validate_run "$method"; then
    echo "==> Reusing ${description}"
    return
  fi
  echo "==> Running ${description}"
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
    DCCL.PL_MEMORY "$pl_memory" \
    DCCL.TARGET_HEAD_ADAPT True \
    DCCL.GTR_PAR 0.05 \
    DCCL.TEMPORAL_DIAG True
  validate_run "$method"
}

run_control
run_candidate \
  "$current_method" both_prior stable \
  "Phase 1/2 current Stage14 (both_prior + stable)"

control_glob="output/uda/VISDA-C/TV/${control_method}/*.txt"
current_glob="output/uda/VISDA-C/TV/${current_method}/*.txt"
reproduction_json="$result_dir/reproduction_gate.json"

set +e
python tools/summarize_visda_stage14_prior_memory.py reproduction \
  --control-glob "$control_glob" \
  --current-glob "$current_glob" \
  --class-names data/VISDA-C/classname.txt \
  --out "$reproduction_json" \
  --max-reproduction-delta -0.15
reproduction_status=$?
set -e

if [ "$reproduction_status" -eq 2 ]; then
  echo "==> STOP: the matched full-data Stage14 gap was not reproduced." >&2
  echo "See: $reproduction_json" >&2
  exit 2
fi
if [ "$reproduction_status" -ne 0 ]; then
  echo "Reproduction analysis failed; refusing to launch the factorial." >&2
  exit "$reproduction_status"
fi

echo "==> Gap reproduced. Starting the remaining fixed 2x2 factorial arms."
run_candidate \
  "$none_stable_method" none stable \
  "Phase 2/2 calibration ablation (none + stable)"
run_candidate \
  "$both_monotonic_method" both_prior monotonic \
  "Phase 2/2 memory ablation (both_prior + monotonic)"
run_candidate \
  "$none_monotonic_method" none monotonic \
  "Phase 2/2 interaction ablation (none + monotonic)"

factorial_json="$result_dir/factorial_summary.json"
factorial_csv="$result_dir/per_class_factorial.csv"
python tools/summarize_visda_stage14_prior_memory.py factorial \
  --control-glob "$control_glob" \
  --current-glob "$current_glob" \
  --none-stable-glob \
    "output/uda/VISDA-C/TV/${none_stable_method}/*.txt" \
  --both-prior-monotonic-glob \
    "output/uda/VISDA-C/TV/${both_monotonic_method}/*.txt" \
  --none-monotonic-glob \
    "output/uda/VISDA-C/TV/${none_monotonic_method}/*.txt" \
  --class-names data/VISDA-C/classname.txt \
  --out "$factorial_json" \
  --csv-out "$factorial_csv" \
  --max-reproduction-delta -0.15 \
  --min-material-effect 0.10 \
  --min-duet-gain 0.10

echo "==> Full-data Stage14 prior/memory factorial complete"
echo "Reproduction gate: $reproduction_json"
echo "Factorial conclusion: $factorial_json"
echo "Per-class table: $factorial_csv"
echo "No eight-cycle run or seed sweep was started automatically."
