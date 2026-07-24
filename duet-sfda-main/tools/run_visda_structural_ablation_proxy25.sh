#!/usr/bin/env bash
set -euo pipefail

proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
result_dir="output/uda/VISDA-C"
control_method="plmatch_visda_proxy25_seed2020"
v1_method="temporal_precision_head_visda_proxy25_v1_monotonic_head"
v2_method="temporal_precision_head_visda_proxy25_v2_stable_nohead"
v3_method="temporal_precision_head_visda_proxy25_v3_monotonic_nohead"

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
  echo "==> Creating deterministic VisDA-C 25% proxy list"
  python tools/prepare_visda_proxy_subset.py \
    --input data/VISDA-C/validation_list.txt \
    --output "$proxy_list" \
    --ratio 0.25 \
    --seed 2020
fi
if ! cmp -s "$expected_proxy" "$proxy_list"; then
  echo "Proxy list does not match deterministic ratio=0.25 seed=2020 selection" >&2
  exit 1
fi

control_glob="output/uda/VISDA-C/TV/${control_method}/*.txt"
if ! compgen -G "$control_glob" > /dev/null; then
  echo "Missing matched official-DUET proxy control: $control_glob" >&2
  echo "Run first: bash tools/run_visda_plmatch_proxy25_control.sh" >&2
  exit 1
fi
control_logs=(output/uda/VISDA-C/TV/${control_method}/*.txt)
if [ "${#control_logs[@]}" -ne 1 ]; then
  echo "Expected exactly one official-DUET proxy control log" >&2
  exit 1
fi
control_checkpoints=$(grep -c "Task: TV" "${control_logs[0]}")
if [ "$control_checkpoints" -ne 16 ]; then
  echo "Official-DUET proxy control is incomplete: ${control_checkpoints}/16 checkpoints" >&2
  exit 1
fi

mkdir -p "$result_dir"
sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt \
  > "$result_dir/visda_structural_ablation_proxy25_source_sha256.txt"
sha256sum "$proxy_list" \
  > "$result_dir/visda_structural_ablation_proxy25_list_sha256.txt"

run_variant() {
  method=$1
  pl_memory=$2
  target_head=$3
  artifact_prefix=$4
  description=$5
  log_glob="output/uda/VISDA-C/TV/${method}/*.txt"

  if compgen -G "$log_glob" > /dev/null; then
    candidate_logs=(output/uda/VISDA-C/TV/${method}/*.txt)
    if [ "${#candidate_logs[@]}" -ne 1 ]; then
      echo "${description}: expected exactly one existing log" >&2
      exit 1
    fi
    checkpoint_count=$(grep -c "Task: TV" "${candidate_logs[0]}")
    if [ "$checkpoint_count" -ne 16 ]; then
      echo "${description}: existing log is incomplete (${checkpoint_count}/16); refusing to mix runs" >&2
      exit 1
    fi
    echo "==> Reusing completed ${description}"
  else
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
      DCCL.ADAPTATION_LIST "$proxy_list" \
      DCCL.CALIB_MODE both_prior \
      DCCL.CALIB_POWER 0.5 \
      DCCL.PL_MEMORY "$pl_memory" \
      DCCL.TARGET_HEAD_ADAPT "$target_head" \
      DCCL.GTR_PAR 0.0 \
      DCCL.CONSISTENCY_STOP_GRAD False \
      DCCL.LOSS_DIAG True \
      DCCL.TEMPORAL_DIAG True
  fi

  candidate_logs=(output/uda/VISDA-C/TV/${method}/*.txt)
  if [ "${#candidate_logs[@]}" -ne 1 ] || ! grep -q "Task: TV" "${candidate_logs[0]}"; then
    echo "${description}: training produced no VisDA-C accuracy records" >&2
    exit 1
  fi
  checkpoint_count=$(grep -c "Task: TV" "${candidate_logs[0]}")
  if [ "$checkpoint_count" -ne 16 ]; then
    echo "${description}: expected 16 checkpoints, found ${checkpoint_count}" >&2
    exit 1
  fi
  if ! grep -q \
    "DCCL adaptation proxy list: ${proxy_list}; adaptation_samples=13847; full_evaluation_samples=55388" \
    "${candidate_logs[0]}"; then
    echo "${description}: proxy/full loader counts were not verified" >&2
    exit 1
  fi
  if ! grep -q "PL_MEMORY: ${pl_memory}" "${candidate_logs[0]}"; then
    echo "${description}: PL_MEMORY=${pl_memory} was not verified" >&2
    exit 1
  fi
  if ! grep -q "TARGET_HEAD_ADAPT: ${target_head}" "${candidate_logs[0]}"; then
    echo "${description}: TARGET_HEAD_ADAPT=${target_head} was not verified" >&2
    exit 1
  fi
  if ! grep -q "GTR_PAR: 0.0" "${candidate_logs[0]}"; then
    echo "${description}: GTR_PAR=0 was not verified" >&2
    exit 1
  fi

  python tools/extract_final_accuracy.py \
    --glob "$log_glob" \
    --selection peak \
    > "$result_dir/${artifact_prefix}_accuracy.csv"

  python tools/summarize_visda_temporal_precision_head.py \
    --glob "$log_glob" \
    --out "$result_dir/${artifact_prefix}_summary.json" \
    --csv-out "$result_dir/${artifact_prefix}_per_class.csv" \
    --class-names data/VISDA-C/classname.txt

  python tools/analyze_temporal_conflict_dynamics.py \
    --glob "output/uda/VISDA-C/TV/${method}/temporal_diagnostics/*_cycle*.npz" \
    --out "$result_dir/${artifact_prefix}_dynamics.json" \
    --min-pass-tasks 1
}

run_variant \
  "$v1_method" monotonic True \
  "visda_structural_v1_monotonic_head" \
  "V1 monotonic memory + target head"

run_variant \
  "$v2_method" stable False \
  "visda_structural_v2_stable_nohead" \
  "V2 stable/reversible memory + no target head"

run_variant \
  "$v3_method" monotonic False \
  "visda_structural_v3_monotonic_nohead" \
  "V3 monotonic memory + no target head"

python tools/summarize_visda_structural_ablation.py \
  --control-glob "$control_glob" \
  --v1-glob "output/uda/VISDA-C/TV/${v1_method}/*.txt" \
  --v2-glob "output/uda/VISDA-C/TV/${v2_method}/*.txt" \
  --v3-glob "output/uda/VISDA-C/TV/${v3_method}/*.txt" \
  --out "$result_dir/visda_structural_ablation_proxy25_gate.json" \
  --csv-out "$result_dir/visda_structural_ablation_proxy25_results.csv" \
  --class-names data/VISDA-C/classname.txt \
  --min-final-improvement 0.15 \
  --min-hard-mean-improvement 0.0 \
  --max-other9-regression 0.10 \
  --max-hard-class-regression 0.50

echo "==> Structural ablation complete"
echo "Gate: $result_dir/visda_structural_ablation_proxy25_gate.json"
echo "Table: $result_dir/visda_structural_ablation_proxy25_results.csv"
echo "Do not run a full-data variant unless the gate reports pass_proxy_gate."
