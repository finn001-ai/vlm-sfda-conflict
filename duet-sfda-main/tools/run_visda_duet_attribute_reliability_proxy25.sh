#!/usr/bin/env bash
set -euo pipefail

seed=2020
method="duet_attribute_reliability_kl_visda_proxy25_seed${seed}"
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
candidate_dir="output/uda/VISDA-C/TV/${method}"
result_dir="output/uda/VISDA-C"
offline_dir="${result_dir}/TV/plmatch_visda_attribute_reliability_audit_seed${seed}/attribute_reliability_audit"
offline_summary="${offline_dir}/visda_conflict_attribute_reliability_summary.json"
offline_lock="${offline_dir}/visda_conflict_attribute_reliability_target_lock.json"
current_control_summary="${result_dir}/plmatch_visda_proxy25_seed2020_summary.json"
current_source_hash="${result_dir}/plmatch_visda_proxy25_seed2020_source_sha256.txt"
current_proxy_hash="${result_dir}/plmatch_visda_proxy25_seed2020_proxy_sha256.txt"
archive_control_dir="${ARCHIVED_CONTROL_DIR:-../archive/sfda_conflict_visda_proxy_loss_audit_2026-07-23}"
archive_control_summary="${archive_control_dir}/plmatch_visda_proxy25_seed2020_summary.json"
archive_control_log="${archive_control_dir}/plmatch_proxy25_control_terminal_record.txt"
archive_checksums="${archive_control_dir}/SHA256SUMS"
historical_control_commit="91ef7df"
candidate_summary="${result_dir}/duet_attribute_reliability_kl_visda_proxy25_seed2020_summary.json"
candidate_per_class="${result_dir}/duet_attribute_reliability_kl_visda_proxy25_seed2020_per_class.csv"
gate="${result_dir}/duet_attribute_reliability_kl_visda_proxy25_seed2020_gate.json"
candidate_source_hash="${result_dir}/duet_attribute_reliability_kl_visda_proxy25_seed2020_source_sha256.txt"
candidate_proxy_hash="${result_dir}/duet_attribute_reliability_kl_visda_proxy25_seed2020_proxy_sha256.txt"
candidate_contract_hash="${result_dir}/duet_attribute_reliability_kl_visda_proxy25_seed2020_contract_sha256.txt"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt \
  "$offline_summary" \
  "$offline_lock"; do
  if [ ! -f "$path" ]; then
    echo "Missing attribute-reliability proxy input: $path" >&2
    exit 1
  fi
done

python - "$offline_summary" "$offline_lock" <<'PY'
import hashlib
import json
import sys

summary_path, lock_path = sys.argv[1:]
summary = json.load(open(summary_path))
lock = json.load(open(lock_path))
actual_lock_hash = hashlib.sha256(open(lock_path, "rb").read()).hexdigest()
fixed = summary.get("oracle_metrics", {}).get("comparisons", {}).get("fixed_clip", {})
accuracy = fixed.get("accuracy", {})
checks = {
    "input_contract": summary.get("input_contract", {}).get("passed") is True,
    "lock_hash": actual_lock_hash == summary.get("target_lock_sha256"),
    "labels_locked": summary.get("labels_used_only_after_target_lock") is True,
    "accuracy_gain": accuracy.get("gain_pp", 0.0) >= 1.0,
    "accuracy_ci": accuracy.get("paired_bootstrap_95_ci_pp", [0.0])[0] > 0.0,
    "soft_metrics": all(fixed.get("checks", {}).values()),
    "mass_shift": summary.get("label_free_diagnostic", {}).get(
        "max_abs_class_mass_shift_pp", 999.0
    ) <= 1.0,
    "label_free_rule": (
        lock.get("contains_target_labels") is False
        and lock.get("formula", {}).get("target_label_thresholds") is False
        and lock.get("formula", {}).get("tunable_coefficients") is False
        and lock.get("formula", {}).get("class_specific_rules") is False
    ),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Offline evidence contract failed: {failed}")
print("==> Offline evidence admitted for one proxy25 run")
print("    The earlier REJECT reflects the truck/non-car safety gate; it is not hidden.")
print(f"    conflict_gain_pp={accuracy['gain_pp']:.6f}")
print(f"    truck_gain_pp={summary['class_safety']['truck']['accuracy_gain_pp']:.6f}")
print(f"    noncar_net={summary['class_safety']['noncar_net_corrections']}")
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

control_summary="$current_control_summary"
control_provenance="matched_current_source_and_proxy_hashes"
if [ -f "$current_control_summary" ] && \
   [ -f "$current_source_hash" ] && \
   [ -f "$current_proxy_hash" ]; then
  if ! cmp -s "$observed_source_hash" "$current_source_hash"; then
    echo "Source checkpoint hashes differ from the matched control" >&2
    exit 1
  fi
  if ! cmp -s "$observed_proxy_hash" "$current_proxy_hash"; then
    echo "Proxy-list hash differs from the matched control" >&2
    exit 1
  fi
  echo "==> Reusing matched current-output arithmetic-DUET proxy control"
else
  for artifact in "$archive_control_summary" "$archive_control_log" "$archive_checksums"; do
    if [ ! -f "$artifact" ]; then
      echo "Missing archived DUET control evidence: $artifact" >&2
      exit 1
    fi
  done
  verify_archive_file() {
    local filename="$1"
    local expected_sha
    local actual_sha
    expected_sha=$(awk -v filename="$filename" '$2 == filename {print $1}' "$archive_checksums")
    actual_sha=$(sha256sum "${archive_control_dir}/${filename}" | awk '{print $1}')
    if [ -z "$expected_sha" ] || [ "$actual_sha" != "$expected_sha" ]; then
      echo "Archived control checksum failed: $filename" >&2
      exit 1
    fi
  }
  verify_archive_file "plmatch_visda_proxy25_seed2020_summary.json"
  verify_archive_file "plmatch_proxy25_control_terminal_record.txt"
  control_summary="$archive_control_summary"
  control_provenance="archived_control_without_source_or_proxy_hashes"
  echo "==> Reusing archived arithmetic-DUET proxy control: final=87.93"
  echo "    Provenance limitation: historical source/list hashes were not archived"
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
if ! git show "${historical_control_commit}:duet-sfda-main/src/methods/oh/plmatch.py" \
  > "$historical_plmatch"; then
  echo "Cannot audit historical DUET control commit: $historical_control_commit" >&2
  exit 1
fi
if ! grep -Fq "all_mix_output = (all_output + clip_all_output) / 2" "$historical_plmatch"; then
  echo "Historical DUET control was not arithmetic probability fusion" >&2
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
  cfgs/visda/duet_attribute_reliability_kl.yaml \
  src/methods/oh/plmatch.py \
  src/methods/oh/duet_attribute_reliability_kl.py \
  src/utils/attribute_reliability.py \
  src/utils/pairwise_attribute_audit.py \
  > "$candidate_contract_hash"

echo "==> One matched DUET attribute-reliability proxy25 candidate"
echo "==> Only cycle-1 unresolved-conflict KL targets change"
echo "==> Archived control runtime was 2186.87 seconds; no control rerun"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_attribute_reliability_kl.yaml \
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
if ! grep -q \
  "DUET attribute reliability KL: enabled=True; first_cycle_only=True; target=unresolved_conflicts" \
  "${logs[0]}"; then
  echo "Candidate did not enable the locked KL intervention" >&2
  exit 1
fi
if [ "$(grep -Ec 'DUET attribute reliability KL applied: cycle=1; active_conflicts=[1-9][0-9]*; changed_top1=[1-9][0-9]*;' "${logs[0]}")" -ne 1 ]; then
  echo "Cycle-1 attribute-reliability target contract failed" >&2
  exit 1
fi
if [ "$(grep -Ec 'DUET attribute reliability KL applied: cycle=[234]; active_conflicts=0;' "${logs[0]}")" -ne 3 ]; then
  echo "Attribute-reliability KL was not restricted to cycle 1" >&2
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

python tools/analyze_duet_attribute_reliability_proxy.py \
  --control-summary "$control_summary" \
  --candidate-summary "$candidate_summary" \
  --control-provenance "$control_provenance" \
  --out "$gate"

echo "==> Gate: $gate"
echo "==> Even PASS does not authorize or start a full VisDA run"
