#!/usr/bin/env bash
set -euo pipefail

# DUET-FCP + swap-conflict hard-label selection, VisDA-C TV, seed 2020.
# Only bidirectional_cross_support (pure-swap) conflicts receive hard labels:
# cycle 0 -> CLIP top1 (no gate); cycle >= 1 -> |log(eA)-log(eB)| >= D with
# D defaulting to 4.0 (override with SWAP_GATE_D).
seed="${SEED:-2020}"
gate_d="${SWAP_GATE_D:-4.0}"
method="duet_first_cycle_prior_swap_selection_visda_seed${seed}"
run_dir="output/uda/VISDA-C/TV/${method}"

for path in \
  data/VISDA-C/train_list.txt \
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

if [ -e "$run_dir" ]; then
  echo "Refusing to overwrite existing run: $run_dir" >&2
  exit 1
fi

echo "==> DUET-FCP with swap-conflict selection, VisDA-C TV, seed=${seed}, gate_D=${gate_d}"
echo "==> Scope: bidirectional_cross_support only; abstain samples stay out of the loss"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior_swap_selection.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 8 \
  DUET_SWAP.GATE_D "$gate_d"

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected exactly one VisDA-C log, found ${#logs[@]}" >&2
  exit 1
fi
log="${logs[0]}"
if [ "$(grep -c "DUET swap-conflict selection: enabled=True" "$log")" -ne 1 ]; then
  echo "Swap-conflict selection activation contract failed" >&2
  exit 1
fi
if [ "$(grep -c "DUET swap-conflict selection: cycle=" "$log")" -ne 8 ]; then
  echo "Expected per-cycle swap logs for all 8 cycles" >&2
  exit 1
fi
if [ "$(grep -c "DUET first-cycle prior schedule: cycle=1; active=True" "$log")" -ne 1 ]; then
  echo "First-cycle prior activation contract failed" >&2
  exit 1
fi
if [ "$(grep -c "DUET first-cycle prior schedule: cycle=.*; active=False" "$log")" -ne 7 ]; then
  echo "Cycles 2-8 did not preserve raw DUET probabilities" >&2
  exit 1
fi
if [ "$(grep -c "Task: TV" "$log")" -ne 32 ]; then
  echo "VisDA-C did not finish 8 cycles / 32 checkpoints" >&2
  exit 1
fi

grep "DUET swap-conflict selection: cycle=" "$log"
echo "==> Run complete: ${run_dir}"
