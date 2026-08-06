#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Control / ablation runner for the Context Transformer candidate on VISDA-C.
# Usage:
#
#   bash tools/run_visda_duet_context_controls.sh [refiner] [ablation]
#
#   refiner : transformer (default) | cosine_knn | prototype
#   ablation: both (default) | strict_only | weak_only | no_third | no_abstain

seed=2020
refiner="${1:-transformer}"
ablation="${2:-both}"
method="duet_first_cycle_prior_context_transformer_visda_${refiner}_${ablation}_seed${seed}"
run_dir="output/uda/VISDA-C/TV/${method}"
result_dir="output/uda/VISDA-C"

case "$refiner" in
  transformer|cosine_knn|prototype) ;;
  *) echo "Unknown refiner: $refiner" >&2; exit 1 ;;
esac
case "$ablation" in
  both|strict_only|weak_only|no_third|no_abstain) ;;
  *) echo "Unknown ablation: $ablation" >&2; exit 1 ;;
esac

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
  output/uda/VISDA-C/TV/duet_first_cycle_prior_context_transformer_*_seed2020) ;;
  *)
    echo "Refusing to clear unexpected VisDA-C path: $run_dir" >&2
    exit 1
    ;;
esac
rm -rf -- "$run_dir"

ablation_opts=()
case "$ablation" in
  strict_only) ablation_opts+=(DUET_CONTEXT.USE_WEAK_AGREEMENT False) ;;
  weak_only)   ablation_opts+=(DUET_CONTEXT.USE_STRICT_CONFLICT False) ;;
  no_third)    ablation_opts+=(DUET_CONTEXT.ALLOW_THIRD_CLASS False) ;;
  no_abstain)  ablation_opts+=(DUET_CONTEXT.ABSTAIN_WHEN_UNCERTAIN False) ;;
esac

echo "==> DUET-FCP + Context (refiner=${refiner}, ablation=${ablation}) VisDA-C, 8 cycles"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior_context_transformer.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 8 \
  DUET_CONTEXT.REFINER_TYPE "$refiner" \
  "${ablation_opts[@]}"

logs=("$run_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected exactly one VisDA-C log, found ${#logs[@]}" >&2
  exit 1
fi
if ! grep -q "DUET context refinement: cycle=1; active=True; refiner=${refiner}" "${logs[0]}"; then
  echo "VisDA-C did not run refiner=${refiner}" >&2
  exit 1
fi
if [ "$(grep -c "Task: TV" "${logs[0]}")" -ne 32 ]; then
  echo "VisDA-C did not finish 8 cycles / 32 checkpoints" >&2
  exit 1
fi

python tools/summarize_visda_run.py \
  --glob "$run_dir/*.txt" \
  --out "$result_dir/duet_fcp_context_visda_${refiner}_${ablation}_seed${seed}_summary.json" \
  --csv-out "$result_dir/duet_fcp_context_visda_${refiner}_${ablation}_seed${seed}_per_class.csv" \
  --class-names data/VISDA-C/classname.txt

echo "==> VisDA summary: $result_dir/duet_fcp_context_visda_${refiner}_${ablation}_seed${seed}_summary.json"
