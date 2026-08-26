#!/usr/bin/env bash
set -euo pipefail

seed=2020
method="duet_first_cycle_prior_visda_proxy25_seed${seed}"
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
candidate_dir="output/uda/VISDA-C/TV/${method}"
result_dir="output/uda/VISDA-C"
control_summary="${CONTROL_SUMMARY:-${result_dir}/plmatch_visda_proxy25_seed2020_summary.json}"
candidate_summary="${result_dir}/duet_fcp_visda_proxy25_seed2020_summary.json"
gate="${result_dir}/duet_fcp_visda_proxy25_seed2020_gate.json"

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

if [ ! -f "$control_summary" ]; then
  echo "Missing matched DUET proxy summary: $control_summary" >&2
  echo "Run once: bash tools/run_visda_plmatch_proxy25_control.sh" >&2
  echo "Or set CONTROL_SUMMARY to an existing matched summary." >&2
  exit 1
fi

expected_proxy=$(mktemp)
trap 'rm -f "$expected_proxy"' EXIT
python tools/prepare_visda_proxy_subset.py \
  --input data/VISDA-C/validation_list.txt \
  --output "$expected_proxy" \
  --ratio 0.25 \
  --seed "$seed" \
  --force > /dev/null

if [ ! -f "$proxy_list" ]; then
  python tools/prepare_visda_proxy_subset.py \
    --input data/VISDA-C/validation_list.txt \
    --output "$proxy_list" \
    --ratio 0.25 \
    --seed "$seed"
fi
if ! cmp -s "$expected_proxy" "$proxy_list"; then
  echo "Proxy list is not the deterministic ratio=0.25 seed=2020 subset" >&2
  exit 1
fi

case "$candidate_dir" in
  output/uda/VISDA-C/TV/duet_first_cycle_prior_*) ;;
  *)
    echo "Refusing to clear unexpected candidate path: $candidate_dir" >&2
    exit 1
    ;;
esac
rm -rf -- "$candidate_dir"

echo "==> DUET-FCP proxy: original DUET plus first-cycle prior only"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 4 \
  ACTIVE.ADAPTATION_LIST "$proxy_list"

logs=("$candidate_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected exactly one candidate log, found ${#logs[@]}" >&2
  exit 1
fi
if ! grep -q "DUET first-cycle prior: enabled=True; power=0.500" "${logs[0]}"; then
  echo "Candidate did not enable the first-cycle prior" >&2
  exit 1
fi
if [ "$(grep -c "DUET first-cycle prior schedule: cycle=1; active=True" "${logs[0]}")" -ne 1 ]; then
  echo "First-cycle prior activation contract failed" >&2
  exit 1
fi
if [ "$(grep -c "DUET first-cycle prior schedule: cycle=.*; active=False" "${logs[0]}")" -ne 3 ]; then
  echo "Later-cycle prior identity contract failed" >&2
  exit 1
fi
if [ "$(grep -c "Task: TV" "${logs[0]}")" -ne 16 ]; then
  echo "Candidate did not finish the 4-cycle proxy contract" >&2
  exit 1
fi

python tools/summarize_visda_run.py \
  --glob "$candidate_dir/*.txt" \
  --out "$candidate_summary" \
  --csv-out "$result_dir/duet_fcp_visda_proxy25_seed2020_per_class.csv" \
  --class-names data/VISDA-C/classname.txt

python tools/analyze_duet_fcp_visda_proxy.py \
  --control-summary "$control_summary" \
  --candidate-summary "$candidate_summary" \
  --out "$gate"

echo "==> Gate: $gate"
