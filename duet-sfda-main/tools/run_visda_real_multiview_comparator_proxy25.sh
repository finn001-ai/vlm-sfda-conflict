#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Formal Cycle-2 experiment on the locked VisDA-C proxy25 adaptation subset.
# It must resume the existing exact Cycle-1 cache: this runner never reruns C1.

seed=2020
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
cycle1_checkpoint="output/checkpoints/duet_fcp_context_visda_proxy25_seed2020_cycle1.pt"
method="duet_first_cycle_prior_context_transformer_real_multiview_visda_proxy25_seed${seed}"
run_dir="output/uda/VISDA-C/TV/${method}"

# image_target_of_oh_vs.py dispatches the checkpoint-aware implementation only
# for this exact prefix. Refuse to run if a future rename would fall through to
# the generic duet_first_cycle_prior/plmatch implementation.
case "$method" in
  duet_first_cycle_prior_context_transformer_*) ;;
  *)
    echo "Method name would bypass the checkpoint-aware context implementation: $method" >&2
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

expected_proxy=$(mktemp)
trap 'rm -f "$expected_proxy"' EXIT
python tools/prepare_visda_proxy_subset.py \
  --input data/VISDA-C/validation_list.txt \
  --output "$expected_proxy" \
  --ratio 0.25 \
  --seed "$seed" \
  --force > /dev/null
if ! cmp -s "$expected_proxy" "$proxy_list"; then
  echo "Proxy list is not the locked ratio=0.25 seed=2020 subset" >&2
  exit 1
fi

proxy_samples=$(wc -l < "$proxy_list" | tr -d ' ')
full_samples=$(wc -l < data/VISDA-C/validation_list.txt | tr -d ' ')
if [ "$proxy_samples" -ne 13847 ] || [ "$full_samples" -ne 55388 ]; then
  echo "Unexpected VisDA-C size: proxy=${proxy_samples}, full=${full_samples}" >&2
  exit 1
fi

case "$run_dir" in
  output/uda/VISDA-C/TV/duet_first_cycle_prior_context_transformer_real_multiview_visda_proxy25_seed2020) ;;
  *)
    echo "Refusing to clear unexpected VisDA-C path: $run_dir" >&2
    exit 1
    ;;
esac
rm -rf -- "$run_dir"

echo "==> Resume exact Cycle 1; run only Cycle 2 on proxy25"
echo "==> Formal method: synthetic pretrain + GT-free real weak/strong soft fine-tune"
echo "==> Admission coverage=50%; real supervision coverage=60%"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior_context_transformer.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 2 \
  ACTIVE.ADAPTATION_LIST "$proxy_list" \
  DUET_CONTEXT.CYCLE_CHECKPOINT_RESUME_PATH "$cycle1_checkpoint" \
  DUET_CONTEXT.COMPARATOR_COVERAGE_FRACTION 0.50 \
  DUET_CONTEXT.REAL_MULTIVIEW_ENABLED True \
  DUET_CONTEXT.REAL_MULTIVIEW_TRAIN_FRACTION 0.60 \
  DUET_CONTEXT.REAL_MULTIVIEW_FINETUNE_STEPS 100 \
  DUET_CONTEXT.REAL_MULTIVIEW_TEMPERATURE 0.50 \
  DUET_CONTEXT.REAL_MULTIVIEW_SYNTHETIC_MIX_FRACTION 0.25 \
  DUET_CONTEXT.EARLY_STOP_ENABLED False \
  DUET_CONTEXT.REAL_CONFLICT_GT_PROBE_ENABLED False \
  DUET_CONTEXT.REAL_CONFLICT_GT_PROBE_EXTENDED_20D_ENABLED False \
  DUET_CONTEXT.AGREEMENT_AMBIGUITY_EVAL_ENABLED False \
  DUET_CONTEXT.AGREEMENT_COMPARATOR_PROBE_ENABLED False \
  DUET_CONTEXT.AGREEMENT_SYNTHETIC_FEASIBILITY_ENABLED False \
  DUET_CONTEXT.EVAL_TRAJECTORY_ENABLED False

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected exactly one formal-method log, found ${#logs[@]}" >&2
  exit 1
fi
log_file="${logs[0]}"
if ! grep -q "DUET cycle checkpoint resumed:.*completed_cycles=1; next_cycle=2" "$log_file"; then
  echo "Run did not resume the exact Cycle-1 checkpoint" >&2
  exit 1
fi
if [ "$(grep -c "DUET comparator real-multiview training: cycle=2" "$log_file")" -ne 1 ]; then
  echo "Cycle 2 did not execute real-multiview comparator fine-tuning exactly once" >&2
  exit 1
fi
if ! grep -q "DUET comparator selection: cycle=2; mode=rank_coverage;.*requested_coverage=50.00%" "$log_file"; then
  echo "Formal admission was not fixed at 50%" >&2
  exit 1
fi
if ! grep -q "DUET context eval-only: cycle=2;.*resolved_subset_duet_fallback_acc=.*resolved_gain_over_duet_fallback=.*coverage_weighted_gain_over_duet_fallback=" "$log_file"; then
  echo "Same-subset actual DUET fallback comparison is missing" >&2
  exit 1
fi
if grep -q "GT feature probe.*cycle=2" "$log_file"; then
  echo "GT feature probes must be disabled in the formal-method run" >&2
  exit 1
fi
if [ "$(grep -c "Task: TV" "$log_file")" -ne 4 ]; then
  echo "Expected four Cycle-2 task checkpoints after resume" >&2
  exit 1
fi

echo "==> Formal-method summary"
grep "DUET comparator real-multiview training: cycle=2" "$log_file"
grep "DUET comparator selection: cycle=2" "$log_file"
grep "DUET context eval-only: cycle=2" "$log_file"
grep "all_mix_output Accuracy" "$log_file" | tail -1
echo "==> Full log: $log_file"
