#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Resume the locked proxy25 Cycle-1 state and run only Cycle 2. The formal
# intervention is an isolated soft A/B loss on 80% of real conflicts.
seed=2020
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
cycle1_checkpoint="output/checkpoints/duet_fcp_context_visda_proxy25_seed2020_cycle1.pt"
method="duet_first_cycle_prior_context_transformer_conflict_memory80_visda_proxy25_seed${seed}"
run_dir="output/uda/VISDA-C/TV/${method}"

case "$method" in
  duet_first_cycle_prior_context_transformer_*) ;;
  *)
    echo "Method name would bypass the checkpoint-aware implementation: $method" >&2
    exit 1
    ;;
esac

for path in \
  "$proxy_list" \
  "$cycle1_checkpoint" \
  data/VISDA-C/train_list.txt \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -s "$path" ]; then
    echo "Missing required input/cache: $path" >&2
    exit 1
  fi
done

case "$run_dir" in
  output/uda/VISDA-C/TV/duet_first_cycle_prior_context_transformer_conflict_memory80_visda_proxy25_seed2020) ;;
  *)
    echo "Refusing to clear unexpected path: $run_dir" >&2
    exit 1
    ;;
esac
rm -rf -- "$run_dir"

echo "==> Resume exact Cycle 1; run only Cycle 2 on VisDA proxy25"
echo "==> Persistent conflict belief: 80% pool; auxiliary loss weight=0.10"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior_context_transformer.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 2 \
  ACTIVE.ADAPTATION_LIST "$proxy_list" \
  DUET_CONTEXT.CYCLE_CHECKPOINT_RESUME_PATH "$cycle1_checkpoint" \
  DUET_CONTEXT.TRAIN_STEPS_PER_CYCLE 400 \
  DUET_CONTEXT.COMPARATOR_COVERAGE_FRACTION 0.80 \
  DUET_CONTEXT.CONFLICT_MEMORY_ENABLED True \
  DUET_CONTEXT.CONFLICT_MEMORY_COVERAGE_FRACTION 0.80 \
  DUET_CONTEXT.CONFLICT_MEMORY_LOSS_WEIGHT 0.10 \
  DUET_CONTEXT.CONFLICT_MEMORY_TEMPERATURE 0.50 \
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
  echo "Expected exactly one log, found ${#logs[@]}" >&2
  exit 1
fi
log_file="${logs[0]}"
grep -q "DUET cycle checkpoint resumed:.*completed_cycles=1; next_cycle=2" "$log_file"
grep -q "DUET persistent conflict memory: cycle=2;.*raw_coverage=80" "$log_file"
grep -q "DUET conflict-memory isolation: cycle=2; label_mask=original_duet" "$log_file"
grep -q "conflict_pairwise_loss =" "$log_file"
grep -q "DUET conflict-memory training reach: cycle=2;" "$log_file"
if grep -q "DUET context transformer admitted: cycle=2" "$log_file"; then
  echo "Conflict memory unexpectedly changed hard admission" >&2
  exit 1
fi

echo "==> Cycle-2 conflict-memory summary"
grep "DUET persistent conflict memory:" "$log_file"
grep "DUET persistent conflict memory eval-only:" "$log_file"
grep "DUET conflict-memory training reach:" "$log_file"
grep "Task: TV" "$log_file" | tail -4
echo "==> Full log: $log_file"
