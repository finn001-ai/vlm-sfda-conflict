#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

seed=2020
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
cycle1_checkpoint="output/checkpoints/duet_fcp_context_visda_proxy25_seed2020_cycle1.pt"
method="duet_first_cycle_prior_context_transformer_reliability_gate80_visda_proxy25_seed${seed}"
run_dir="output/uda/VISDA-C/TV/${method}"

for path in "$proxy_list" "$cycle1_checkpoint" \
  data/VISDA-C/validation_list.txt data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -s "$path" ]; then
    echo "Missing required input/cache: $path" >&2
    exit 1
  fi
done

case "$run_dir" in
  output/uda/VISDA-C/TV/duet_first_cycle_prior_context_transformer_reliability_gate80_visda_proxy25_seed2020) ;;
  *) echo "Refusing unexpected output path: $run_dir" >&2; exit 1 ;;
esac
rm -rf -- "$run_dir"

echo "==> Resume exact Cycle 1; run reliability-gated full-posterior fusion in Cycle 2"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior_context_transformer.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 2 \
  ACTIVE.ADAPTATION_LIST "$proxy_list" \
  DUET_CONTEXT.CYCLE_CHECKPOINT_RESUME_PATH "$cycle1_checkpoint" \
  DUET_CONTEXT.TRAIN_STEPS_PER_CYCLE 0 \
  DUET_CONTEXT.COMPARATOR_COVERAGE_FRACTION 0.80 \
  DUET_CONTEXT.CONFLICT_MEMORY_ENABLED False \
  DUET_CONTEXT.RELIABILITY_GATE_ENABLED True \
  DUET_CONTEXT.RELIABILITY_GATE_COVERAGE_FRACTION 0.80 \
  DUET_CONTEXT.RELIABILITY_GATE_TEMPERATURE 0.25 \
  DUET_CONTEXT.RELIABILITY_GATE_NEIGHBORS 5 \
  DUET_CONTEXT.REAL_MULTIVIEW_ENABLED False \
  DUET_CONTEXT.REAL_MULTIVIEW_RESIDUAL_FALLBACK False \
  DUET_CONTEXT.SOFT_ONLY_ADMISSION True \
  DUET_CONTEXT.EARLY_STOP_ENABLED False \
  DUET_CONTEXT.REAL_CONFLICT_GT_PROBE_ENABLED False \
  DUET_CONTEXT.REAL_CONFLICT_GT_PROBE_EXTENDED_20D_ENABLED False \
  DUET_CONTEXT.AGREEMENT_AMBIGUITY_EVAL_ENABLED False \
  DUET_CONTEXT.AGREEMENT_COMPARATOR_PROBE_ENABLED False \
  DUET_CONTEXT.AGREEMENT_SYNTHETIC_FEASIBILITY_ENABLED False \
  DUET_CONTEXT.EVAL_TRAJECTORY_ENABLED False

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected one log, found ${#logs[@]}" >&2
  exit 1
fi
log_file="${logs[0]}"
grep -q "DUET cycle checkpoint resumed:.*completed_cycles=1; next_cycle=2" "$log_file"
grep -q "DUET reliability-gated comparator: cycle=2;.*coverage=80" "$log_file"
grep -q "DUET reliability-gated soft target: cycle=2;.*hard_admission=0" "$log_file"
grep -q "DUET reliability-gate isolation: cycle=2; label_mask=original_duet" "$log_file"
if grep -q "DUET context transformer admitted: cycle=2" "$log_file"; then
  echo "Reliability gate unexpectedly changed hard admission" >&2
  exit 1
fi

echo "==> Reliability-gate result"
grep "DUET reliability-gated comparator:" "$log_file"
grep "DUET reliability-gated comparator eval-only:" "$log_file"
grep "Task: TV" "$log_file" | tail -4
echo "==> Full log: $log_file"
