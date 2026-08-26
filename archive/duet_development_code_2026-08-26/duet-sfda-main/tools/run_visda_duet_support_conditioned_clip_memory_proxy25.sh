#!/usr/bin/env bash
set -euo pipefail

seed=2020
method="duet_support_conditioned_clip_memory_visda_proxy25_seed${seed}"
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
candidate_dir="output/uda/VISDA-C/TV/${method}"
result_dir="output/uda/VISDA-C"
audit_dir="${result_dir}/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed${seed}/cycle2_conflict_memory_audit"
audit_summary="${audit_dir}/visda_cycle2_conflict_memory_summary.json"
audit_lock="${audit_dir}/visda_cycle2_conflict_memory_signal_lock.json"
current_control_summary="${result_dir}/plmatch_visda_proxy25_seed2020_summary.json"
current_source_hash="${result_dir}/plmatch_visda_proxy25_seed2020_source_sha256.txt"
current_proxy_hash="${result_dir}/plmatch_visda_proxy25_seed2020_proxy_sha256.txt"
archive_control_dir="${ARCHIVED_CONTROL_DIR:-../archive/sfda_conflict_visda_proxy_loss_audit_2026-07-23}"
archive_control_summary="${archive_control_dir}/plmatch_visda_proxy25_seed2020_summary.json"
archive_control_log="${archive_control_dir}/plmatch_proxy25_control_terminal_record.txt"
archive_checksums="${archive_control_dir}/SHA256SUMS"
historical_control_commit="91ef7df"
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
  "$audit_lock"; do
  if [ ! -f "$path" ]; then
    echo "Missing unresolved-memory proxy input: $path" >&2
    exit 1
  fi
done

python - "$audit_summary" "$audit_lock" <<'PY'
import hashlib
import json
import sys

summary_path, lock_path = sys.argv[1:]
summary = json.load(open(summary_path))
lock = json.load(open(lock_path))
actual_lock_hash = hashlib.sha256(open(lock_path, "rb").read()).hexdigest()
gate_checks = summary.get("gate", {}).get("checks", {})
failed_gate_checks = sorted(
    name for name, passed in gate_checks.items() if not passed
)
still = summary.get("oracle_metrics", {}).get(
    "still_conflict_first_order_comparison_vs_clip", {}
)
checks = {
    "predeclared_gate_failure_is_not_hidden": summary.get("decision") == "REJECT",
    "only_negative_burden_check_failed": failed_gate_checks
    == ["candidate_negative_burden_not_worse"],
    "input_contract": summary.get("input_contract", {}).get("passed") is True,
    "lock_hash": actual_lock_hash == summary.get("signal_lock_sha256"),
    "labels_locked": summary.get("labels_used_only_after_signal_lock") is True,
    "candidate_identity": (
        summary.get("candidate")
        == "cycle2_clip_conditioned_on_current_top2_union_for_cycle1_conflicts"
        == lock.get("candidate")
    ),
    "label_free_training_rule": (
        lock.get("contains_target_labels") is False
        and lock.get("contains_target_paths") is False
        and lock.get("candidate_contract", {}).get("target_labels") is False
        and lock.get("candidate_contract", {}).get("fitted_thresholds") is False
        and lock.get("candidate_contract", {}).get("loss_term_added") is False
        and lock.get("candidate_contract", {}).get("hard_pseudo_label_changed") is False
    ),
    "still_conflict_signal_positive": (
        still.get("samples", 0) > 0
        and still.get("mean_difference", 0.0) > 0.0
        and still.get("paired_bootstrap_95_ci", [0.0])[0] > 0.0
    ),
    "all_classes_nonnegative": summary.get("oracle_metrics", {}).get(
        "minimum_class_first_order_delta_vs_clip", -1.0
    )
    >= 0.0,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Unresolved-memory evidence contract failed: {failed}")
print("==> Cycle-2 evidence locked for one matched proxy25 experiment")
print("    The predeclared audit remains REJECT; it is not relabeled as PASS.")
print("    Sole failed check: candidate_negative_burden_not_worse")
print(
    "    Still-conflict first-order delta={:.6f}; 95% CI=[{:.6f}, {:.6f}]".format(
        still["mean_difference"], *still["paired_bootstrap_95_ci"]
    )
)
print("    Target labels are oracle diagnostic only and do not enter the rule.")
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
  cfgs/visda/duet_support_conditioned_clip_memory.yaml \
  src/methods/oh/plmatch.py \
  src/methods/oh/duet_support_conditioned_clip_memory.py \
  src/utils/support_conditioned_clip.py \
  > "$candidate_contract_hash"

echo "==> One matched unresolved-memory support-conditioned CLIP proxy25"
echo "==> Cycle 1 is unchanged from the prior candidate"
echo "==> Cycles 2-4 condition only cycle-1 conflicts still unresolved"
echo "==> No control rerun; archived control runtime was 2186.87 seconds"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_support_conditioned_clip_memory.yaml \
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
  "DUET support-conditioned CLIP memory: enabled=True; memory=cycle1_task_clip_conflicts; target=currently_unresolved; support=task_clip_top2_union" \
  "${logs[0]}"; then
  echo "Candidate did not enable the locked unresolved-memory intervention" >&2
  exit 1
fi
if [ "$(grep -Ec 'DUET support-conditioned CLIP applied: cycle=[1234]; active_conflicts=[1-9][0-9]*; changed_top1=0; mean_support_size=' "${logs[0]}")" -ne 4 ]; then
  echo "Support-conditioned target was not active on unresolved conflicts in all cycles" >&2
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

python tools/analyze_duet_support_conditioned_clip_proxy.py \
  --control-summary "$control_summary" \
  --candidate-summary "$candidate_summary" \
  --control-provenance "$control_provenance" \
  --out "$gate"

echo "==> Gate: $gate"
echo "==> Even PASS does not authorize or start a full VisDA run"
