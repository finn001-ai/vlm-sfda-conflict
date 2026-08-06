#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Control / ablation runner for the Context Transformer candidate on
# Office-Home.  Usage:
#
#   bash tools/run_office_home_duet_context_controls.sh [refiner] [ablation]
#
#   refiner : transformer (default) | cosine_knn | prototype
#   ablation: both (default) | strict_only | weak_only | no_third | no_abstain
#
# Ablation matrix (section 17 of the request):
#   3. strict only  -> ablation=strict_only
#   4. weak only    -> ablation=weak_only
#   6. no third     -> ablation=no_third
#   7. allow third  -> ablation=both
#   8. no abstain   -> ablation=no_abstain
#   9. allow abstain-> ablation=both
#  10. cosine kNN   -> refiner=cosine_knn
#  11. prototype    -> refiner=prototype
#  12. Transformer  -> refiner=transformer

seed=2020
refiner="${1:-transformer}"
ablation="${2:-both}"
method="duet_first_cycle_prior_context_transformer_office_home_${refiner}_${ablation}_seed${seed}"
domain_keys=(A C P R)

case "$refiner" in
  transformer|cosine_knn|prototype) ;;
  *) echo "Unknown refiner: $refiner" >&2; exit 1 ;;
esac
case "$ablation" in
  both|strict_only|weak_only|no_third|no_abstain) ;;
  *) echo "Unknown ablation: $ablation" >&2; exit 1 ;;
esac

for path in data/office-home/classname.txt; do
  if [ ! -f "$path" ]; then
    echo "Missing Office-Home input: $path" >&2
    exit 1
  fi
done
for key in "${domain_keys[@]}"; do
  for part in F B C; do
    path="source/uda/office-home/${key}/source_${part}.pt"
    if [ ! -f "$path" ]; then
      echo "Missing Office-Home source weight: $path" >&2
      exit 1
    fi
  done
done

ablation_opts=()
case "$ablation" in
  strict_only) ablation_opts+=(DUET_CONTEXT.USE_WEAK_AGREEMENT False) ;;
  weak_only)   ablation_opts+=(DUET_CONTEXT.USE_STRICT_CONFLICT False) ;;
  no_third)    ablation_opts+=(DUET_CONTEXT.ALLOW_THIRD_CLASS False) ;;
  no_abstain)  ablation_opts+=(DUET_CONTEXT.ABSTAIN_WHEN_UNCERTAIN False) ;;
esac

for s in 0 1 2 3; do
  for t in 0 1 2 3; do
    if [ "$s" -eq "$t" ]; then
      continue
    fi
    task="${domain_keys[$s]}${domain_keys[$t]}"
    task_dir="output/uda/office-home/${task}/${method}"
    case "$task_dir" in
      output/uda/office-home/??/duet_first_cycle_prior_context_transformer_*_seed2020) ;;
      *)
        echo "Refusing to clear unexpected Office-Home path: $task_dir" >&2
        exit 1
        ;;
    esac
    rm -rf -- "$task_dir"

    echo "==> DUET-FCP + Context (refiner=${refiner}, ablation=${ablation}) Office-Home: ${task}"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/duet_first_cycle_prior_context_transformer.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$method" \
      SETTING.SEED "$seed" SETTING.S "$s" SETTING.T "$t" \
      ACTIVE.CYCLE 4 \
      DUET_CONTEXT.REFINER_TYPE "$refiner" \
      "${ablation_opts[@]}"

    logs=("$task_dir"/*.txt)
    if [ "${#logs[@]}" -ne 1 ]; then
      echo "Expected one ${task} log, found ${#logs[@]}" >&2
      exit 1
    fi
    if ! grep -q "DUET context refinement: cycle=1; active=True; refiner=${refiner}" "${logs[0]}"; then
      echo "${task} did not run refiner=${refiner}" >&2
      exit 1
    fi
  done
done

echo "==> Office-Home controls finished: refiner=${refiner}, ablation=${ablation}"
