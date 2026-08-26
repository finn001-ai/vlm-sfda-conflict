#!/usr/bin/env bash
set -euo pipefail

seed=2020
method="plmatch_pcgrad_parameter_audit_seed${seed}"
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
result_dir="output/uda/VISDA-C"
output_dir="${result_dir}/TV/${method}"
audit_dir="${output_dir}/conflict_pcgrad_parameter_audit"
summary="${audit_dir}/visda_conflict_pcgrad_parameter_summary.json"
control_method="plmatch_visda_proxy25_seed2020"
control_dir="${result_dir}/TV/${control_method}"
control_summary="${result_dir}/plmatch_visda_proxy25_seed2020_summary.json"
control_source_hash="${result_dir}/plmatch_visda_proxy25_seed2020_source_sha256.txt"
control_proxy_hash="${result_dir}/plmatch_visda_proxy25_seed2020_proxy_sha256.txt"
feature_summary="${result_dir}/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020/conflict_pcgrad_feature_jacobian_audit/visda_conflict_pcgrad_feature_jacobian_summary.json"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt \
  "$control_summary" \
  "$control_source_hash" \
  "$control_proxy_hash" \
  "$feature_summary"; do
  if [ ! -f "$path" ]; then
    echo "Missing exact parameter-audit input: $path" >&2
    exit 1
  fi
done

control_logs=("$control_dir"/*.txt)
if [ "${#control_logs[@]}" -ne 1 ]; then
  echo "Expected exactly one matched arithmetic-DUET control log" >&2
  exit 1
fi
control_log="${control_logs[0]}"

python - "$control_summary" "$feature_summary" <<'PY'
import json
import sys

control = json.load(open(sys.argv[1]))
feature = json.load(open(sys.argv[2]))
if control.get("final", {}).get("accuracy") != 87.93:
    raise SystemExit("Matched arithmetic-DUET proxy control is not locked at 87.93")
if feature.get("decision") != "NEEDS_EXACT_CONTROL_PARAMETER_AUDIT":
    raise SystemExit("Frozen-head feature Jacobian did not authorize this audit")
if not feature.get("labels_used_only_after_signal_lock"):
    raise SystemExit("Feature-Jacobian oracle-label contract is invalid")
print("==> Evidence lock passed: output -> frozen head -> exact parameter audit")
PY

expected_proxy=$(mktemp)
observed_source_hash=$(mktemp)
observed_proxy_hash=$(mktemp)
trap 'rm -f "$expected_proxy" "$observed_source_hash" "$observed_proxy_hash"' EXIT
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
  echo "Source checkpoints differ from the matched arithmetic-DUET control" >&2
  exit 1
fi
if ! cmp -s "$observed_proxy_hash" "$control_proxy_hash"; then
  echo "Proxy list differs from the matched arithmetic-DUET control" >&2
  exit 1
fi

if [ -d "$output_dir" ]; then
  audit_logs=("$output_dir"/*.txt)
  if [ "${#audit_logs[@]}" -ne 1 ] || \
     [ ! -f "${audit_dir}/visda_conflict_pcgrad_parameter_runtime_raw.json" ] || \
     ! grep -q \
       'PCGrad exact parameter audit stop: after_pre_cycle=2; cycle2_optimizer_steps=0; parameters_updated_by_audit=False' \
       "${audit_logs[0]}"; then
    echo "Existing output is not a completed exact parameter audit" >&2
    exit 1
  fi
  echo "==> Reusing completed exact parameter evidence; GPU will not be started"
else
  echo "==> One pure arithmetic-DUET cycle, then exact no-update parameter audit"
  echo "==> Ten fixed batches: 100 unresolved conflicts + 540 admitted context rows"
  echo "==> Target labels enter only after selection and per-batch gradient locks"
  echo "==> Expected GPU time: about 12-16 minutes; no cycle-2 optimizer step"
  python image_target_of_oh_vs.py \
    --cfg cfgs/visda/plmatch.yaml \
    CKPT_DIR . SETTING.OUTPUT_SRC source \
    MODEL.METHOD "$method" \
    SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
    ACTIVE.CYCLE 2 \
    ACTIVE.ADAPTATION_LIST "$proxy_list" \
    FAILURE_AUDIT.ENABLED False \
    PCGRAD_PARAMETER_AUDIT.ENABLED True \
    PCGRAD_PARAMETER_AUDIT.DIR conflict_pcgrad_parameter_audit
fi

audit_logs=("$output_dir"/*.txt)
if [ "${#audit_logs[@]}" -ne 1 ]; then
  echo "Expected exactly one exact parameter-audit log" >&2
  exit 1
fi
audit_log="${audit_logs[0]}"
if [ "$(grep -c 'Task: TV' "$audit_log")" -ne 4 ]; then
  echo "Audit must run exactly one DUET cycle / four evaluation checkpoints" >&2
  exit 1
fi
if ! grep -q \
  'PCGrad exact parameter audit stop: after_pre_cycle=2; cycle2_optimizer_steps=0; parameters_updated_by_audit=False' \
  "$audit_log"; then
  echo "Audit did not stop before cycle-2 optimization" >&2
  exit 1
fi

python tools/finalize_visda_pcgrad_parameter_audit.py \
  --audit-dir "$audit_dir" \
  --audit-log "$audit_log" \
  --control-log "$control_log" \
  --control-summary "$control_summary" \
  --feature-summary "$feature_summary" \
  --output "$summary"

echo "==> Exact parameter audit complete: $summary"
echo "==> PASS authorizes exactly one matched proxy25 design, never a full run"
