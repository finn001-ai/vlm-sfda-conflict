#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Ablation #2 / baseline: pure duet_first_cycle_prior with POWER=0.8 on
# VISDA-C.  Runs through the candidate entry with DUET_CONTEXT.ENABLED=False,
# which must fully degenerate to the original DUET-FCP training path.

seed=2020
method="duet_first_cycle_prior_context_transformer_visda_full_seed${seed}"
run_dir="output/uda/VISDA-C/TV/${method}"
result_dir="output/uda/VISDA-C"

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

case "$run_dir" in
  output/uda/VISDA-C/TV/duet_first_cycle_prior_context_transformer_visda_full_seed2020) ;;
  *)
    echo "Refusing to clear unexpected VisDA-C path: $run_dir" >&2
    exit 1
    ;;
esac
rm -rf -- "$run_dir"

echo "==> DUET-FCP POWER=0.8 baseline (context disabled) VisDA-C, 8 cycles, seed=${seed}"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior_context_transformer.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 8 \
  DUET_FCP.POWER 0.8 \
  DUET_CONTEXT.ENABLED False

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected exactly one VisDA-C log, found ${#logs[@]}" >&2
  exit 1
fi
if ! grep -q "DUET context transformer: requested=True; enabled=False" "${logs[0]}"; then
  echo "VisDA-C did not degenerate to pure DUET-FCP (context disabled)" >&2
  exit 1
fi
if [ "$(grep -c "DUET first-cycle prior schedule: cycle=1; active=True" "${logs[0]}")" -ne 1 ]; then
  echo "First-cycle prior activation contract failed" >&2
  exit 1
fi
if [ "$(grep -c "Task: TV" "${logs[0]}")" -ne 32 ]; then
  echo "VisDA-C did not finish 8 cycles / 32 checkpoints" >&2
  exit 1
fi

python tools/summarize_visda_run.py \
  --glob "$run_dir/*.txt" \
  --out "$result_dir/duet_fcp_context_visda_baseline_power08_seed2020_summary.json" \
  --csv-out "$result_dir/duet_fcp_context_visda_baseline_power08_seed2020_per_class.csv" \
  --class-names data/VISDA-C/classname.txt

echo "==> VisDA summary: $result_dir/duet_fcp_context_visda_baseline_power08_seed2020_summary.json"
