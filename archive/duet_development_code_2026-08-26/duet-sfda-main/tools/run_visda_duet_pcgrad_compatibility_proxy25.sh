#!/usr/bin/env bash
set -euo pipefail

seed=2020
method="duet_pcgrad_compatibility_visda_proxy25_seed${seed}"
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
result_dir="output/uda/VISDA-C"
candidate_dir="${result_dir}/TV/${method}"
parameter_dir="${result_dir}/TV/plmatch_pcgrad_parameter_audit_seed2020/conflict_pcgrad_parameter_audit"
audit_dir="${parameter_dir}/pcgrad_compatibility_audit"
audit_summary="${audit_dir}/visda_conflict_pcgrad_compatibility_summary.json"
audit_lock="${audit_dir}/visda_conflict_pcgrad_compatibility_signal_lock.json"
audit_signal="${audit_dir}/visda_conflict_pcgrad_compatibility_label_free.npz"
control_summary="${result_dir}/plmatch_visda_proxy25_seed2020_summary.json"
control_source_hash="${result_dir}/plmatch_visda_proxy25_seed2020_source_sha256.txt"
control_proxy_hash="${result_dir}/plmatch_visda_proxy25_seed2020_proxy_sha256.txt"
candidate_summary="${result_dir}/${method}_summary.json"
candidate_per_class="${result_dir}/${method}_per_class.csv"
gate="${result_dir}/${method}_gate.json"
candidate_source_hash="${result_dir}/${method}_source_sha256.txt"
candidate_proxy_hash="${result_dir}/${method}_proxy_sha256.txt"
candidate_contract_hash="${result_dir}/${method}_contract_sha256.txt"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt \
  "$audit_summary" \
  "$audit_lock" \
  "$audit_signal" \
  "$control_summary" \
  "$control_source_hash" \
  "$control_proxy_hash"; do
  if [ ! -f "$path" ]; then
    echo "Missing compatibility-proxy input: $path" >&2
    exit 1
  fi
done

python - "$audit_summary" "$audit_lock" "$audit_signal" <<'PY'
import hashlib
import json
import sys

summary_path, lock_path, signal_path = sys.argv[1:]
summary = json.load(open(summary_path))
lock = json.load(open(lock_path))
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
checks = {
    "exploratory_preflight_passed": (
        summary.get("decision") == "PASS_EXPLORATORY_COMPATIBILITY_PREFLIGHT"
    ),
    "one_proxy_authorized": summary.get("matched_proxy_authorized") is True,
    "full_training_forbidden": summary.get("full_training_authorized") is False,
    "seed_sweep_forbidden": summary.get("seed_sweep_authorized") is False,
    "raw_reject_preserved": summary.get("raw_pcgrad_decision_unchanged") == "REJECT",
    "labels_locked": summary.get(
        "labels_used_only_after_compatibility_signal_lock"
    ) is True,
    "lock_hash": sha(lock_path)
    == summary.get("artifacts", {}).get("signal_lock_sha256"),
    "signal_hash": sha(signal_path) == lock.get("signal_sha256"),
    "candidate_identity": (
        summary.get("candidate")
        == "cycle2_full_gradient_compatibility_fraction_for_conflict_pcgrad"
    ),
    "label_free_rule": (
        lock.get("contains_target_labels") is False
        and lock.get("contains_target_paths") is False
        and lock.get("candidate_contract", {}).get("target_labels_used_by_rule")
        is False
        and lock.get("candidate_contract", {}).get("thresholds_fitted") is False
        and lock.get("candidate_contract", {}).get("hyperparameters_added") == 0
    ),
    "cycle_scope": lock.get("candidate_contract", {}).get("active_cycle") == 2,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Compatibility proxy evidence contract failed: {failed}")
print("==> Exploratory compatibility evidence authorizes one matched proxy25")
print("    Raw PCGrad remains REJECT; this is a distinct label-free gradient rule.")
print("    Target labels do not enter the fraction, scope, or optimizer update.")
PY

expected_proxy=$(mktemp)
observed_source_hash=$(mktemp)
observed_proxy_hash=$(mktemp)
historical_plmatch=$(mktemp)
trap 'rm -f "$expected_proxy" "$observed_source_hash" "$observed_proxy_hash" "$historical_plmatch"' EXIT

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

sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt > "$observed_source_hash"
sha256sum "$proxy_list" > "$observed_proxy_hash"
if ! cmp -s "$observed_source_hash" "$control_source_hash"; then
  echo "Source checkpoint hashes differ from the matched arithmetic-DUET control" >&2
  exit 1
fi
if ! cmp -s "$observed_proxy_hash" "$control_proxy_hash"; then
  echo "Proxy-list hash differs from the matched arithmetic-DUET control" >&2
  exit 1
fi
python - "$control_summary" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
final = summary.get("final", {})
if not (
    summary.get("num_checkpoints") == 16
    and final.get("cycle") == 4
    and final.get("accuracy") == 87.93
):
    raise SystemExit("Arithmetic-DUET control is not the locked proxy25 run")
PY
if ! git show "91ef7df:duet-sfda-main/src/methods/oh/plmatch.py" > "$historical_plmatch"; then
  echo "Cannot audit the historical arithmetic-DUET control commit" >&2
  exit 1
fi
if ! grep -Fq "all_mix_output = (all_output + clip_all_output) / 2" "$historical_plmatch"; then
  echo "Historical matched control was not arithmetic probability fusion" >&2
  exit 1
fi

if [ -d "$candidate_dir" ]; then
  echo "Existing candidate directory found; refusing to overwrite: $candidate_dir" >&2
  exit 1
fi
mkdir -p "$result_dir"
cp "$observed_source_hash" "$candidate_source_hash"
cp "$observed_proxy_hash" "$candidate_proxy_hash"
sha256sum \
  conf.py \
  cfgs/visda/duet_pcgrad_compatibility.yaml \
  src/methods/oh/plmatch.py \
  src/methods/oh/duet_pcgrad_compatibility.py \
  src/utils/pcgrad_parameter_audit.py \
  src/utils/pcgrad_compatibility.py \
  "$audit_summary" \
  "$audit_lock" \
  > "$candidate_contract_hash"

echo "==> One matched arithmetic-DUET + cycle-2 compatibility-PCGrad proxy25"
echo "==> Cycles 1, 3, and 4 are unchanged; only cycle 2 gets the extra VJP"
echo "==> No control rerun; locked control final is 87.93"
echo "==> Expected GPU time: about 40-50 minutes"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_pcgrad_compatibility.yaml \
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
if ! grep -Fq \
  "DUET PCGrad compatibility: enabled=True; active_cycle=2;" \
  "${logs[0]}"; then
  echo "Candidate did not enable compatibility-controlled PCGrad" >&2
  exit 1
fi
if [ "$(grep -Ec 'DUET PCGrad compatibility cycle summary: cycle=[134]; active=False; audited_batches=0;' "${logs[0]}")" -ne 3 ]; then
  echo "Non-audited cycles changed or did not report the locked contract" >&2
  exit 1
fi
if [ "$(grep -Ec 'DUET PCGrad compatibility cycle summary: cycle=2; active=True; audited_batches=[1-9][0-9]*; applied_batches=[1-9][0-9]*; unresolved_rows=[1-9][0-9]*; output_pcgrad_active_rows=[1-9][0-9]*;' "${logs[0]}")" -ne 1 ]; then
  echo "Cycle-2 compatibility intervention was inactive or incomplete" >&2
  exit 1
fi
if [ "$(grep -c 'Task: TV' "${logs[0]}")" -ne 16 ]; then
  echo "Candidate did not finish the four-cycle proxy contract" >&2
  exit 1
fi

python tools/summarize_visda_run.py \
  --glob "$candidate_dir/*.txt" \
  --out "$candidate_summary" \
  --csv-out "$candidate_per_class" \
  --class-names data/VISDA-C/classname.txt

python tools/analyze_duet_pcgrad_compatibility_proxy.py \
  --control-summary "$control_summary" \
  --candidate-summary "$candidate_summary" \
  --control-provenance matched_current_source_and_proxy_hashes \
  --out "$gate"

echo "==> Gate: $gate"
echo "==> Even PASS does not authorize or start a full VisDA run"
