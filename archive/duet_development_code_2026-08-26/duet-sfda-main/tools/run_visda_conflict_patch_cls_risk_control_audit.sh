#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
source_dir="${base_dir}/patch_cls_contribution_audit"
audit_dir="${source_dir}/risk_control_audit"
summary="${audit_dir}/visda_conflict_patch_cls_risk_control_summary.json"

source_signal="${source_dir}/visda_conflict_patch_cls_contribution_label_free.npz"
source_lock="${source_dir}/visda_conflict_patch_cls_contribution_signal_lock.json"
source_oracle="${source_dir}/visda_conflict_patch_cls_contribution_oracle_diagnostic.csv"
source_classwise="${source_dir}/visda_conflict_patch_cls_contribution_classwise_oracle_diagnostic.csv"
source_summary="${source_dir}/visda_conflict_patch_cls_contribution_summary.json"

for path in \
  "$source_signal" \
  "$source_lock" \
  "$source_oracle" \
  "$source_classwise" \
  "$source_summary"; do
  if [ ! -f "$path" ]; then
    echo "Missing patch risk-control input: $path" >&2
    echo "Run first: bash tools/run_visda_conflict_patch_cls_contribution_audit.sh" >&2
    exit 1
  fi
done

if [ -e "$audit_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed risk-control audit found; refusing to overwrite: $audit_dir" >&2
    exit 1
  fi
  incomplete_dir="${audit_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$audit_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> CPU-only exploratory patch-to-CLS risk-control audit"
echo "==> Stable rescues: upper full-head-margin median, no searched fraction"
echo "==> Pseudo-class mass shift capped at the prior predeclared 1% limit"
echo "==> Reads only locked arrays before writing the new signal lock"
echo "==> No image/model/checkpoint, forward, backward, optimizer, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_patch_cls_risk_control.py \
  --source-signal "$source_signal" \
  --source-lock "$source_lock" \
  --source-oracle "$source_oracle" \
  --source-classwise "$source_classwise" \
  --source-summary "$source_summary" \
  --output-dir "$audit_dir"

for artifact in \
  "${audit_dir}/visda_conflict_patch_cls_risk_control_label_free.npz" \
  "${audit_dir}/visda_conflict_patch_cls_risk_control_signal_lock.json" \
  "${audit_dir}/visda_conflict_patch_cls_risk_control_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_patch_cls_risk_control_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_patch_cls_risk_control_summary.md" \
  "$summary"; do
  if [ ! -s "$artifact" ]; then
    echo "Missing or empty audit artifact: $artifact" >&2
    exit 1
  fi
done

python - "$summary" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    summary = json.load(handle)
safety = summary["safety"]
required = {
    "target_images_loaded": False,
    "model_or_checkpoint_loaded": False,
    "forward_calls": 0,
    "backward_calls": 0,
    "optimizer_constructed": False,
    "parameter_updates": 0,
    "parameter_audit_authorized": False,
    "proxy_authorized": False,
    "training_authorized": False,
}
for key, expected in required.items():
    if safety.get(key) != expected:
        raise SystemExit(f"Audit safety contract failed: {key}={safety.get(key)!r}")
if summary.get("decision") not in {
    "PASS_EXPLORATORY_PATCH_CLS_RISK_CONTROL",
    "REJECT",
}:
    raise SystemExit(f"Unexpected audit decision: {summary.get('decision')!r}")
print(json.dumps({
    "decision": summary["decision"],
    "checks": summary["gate"]["checks"],
    "runtime_seconds": summary["runtime_seconds"],
}, indent=2))
PY

echo "==> Audit complete: $summary"
echo "==> Exploratory PASS authorizes one held-out full-target audit only"
