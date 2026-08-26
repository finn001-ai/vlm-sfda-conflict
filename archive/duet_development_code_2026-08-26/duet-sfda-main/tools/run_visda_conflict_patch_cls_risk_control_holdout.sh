#!/usr/bin/env bash
set -euo pipefail

feature_dir="output/uda/VISDA-C/TV/plmatch_visda_feature_gravity_audit_seed2020/feature_gravity_audit"
exploratory_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020/patch_cls_contribution_audit/risk_control_audit"
audit_dir="${feature_dir}/patch_cls_holdout_audit"
summary="${audit_dir}/visda_conflict_patch_cls_risk_control_holdout_summary.json"

source_signal="${feature_dir}/visda_conflict_feature_gravity_signals.npz"
source_lock="${feature_dir}/visda_conflict_feature_gravity_signal_lock.json"
exploratory_summary="${exploratory_dir}/visda_conflict_patch_cls_risk_control_summary.json"

if [ ! -f "$exploratory_summary" ]; then
  echo "==> Exploratory CPU lock is missing; generating it from the completed patch audit"
  bash tools/run_visda_conflict_patch_cls_risk_control_audit.sh
fi

for path in \
  "$source_signal" \
  "$source_lock" \
  "$exploratory_summary" \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/validation_proxy25_seed2020_list.txt \
  data/VISDA-C/classname.txt; do
  if [ ! -f "$path" ]; then
    echo "Missing held-out audit input: $path" >&2
    if [ "$path" = "$source_signal" ] || [ "$path" = "$source_lock" ]; then
      echo "The previously completed full-target feature-gravity artifacts are required." >&2
      echo "Do not rerun them if they exist elsewhere; copy the locked NPZ and JSON into the path above." >&2
    fi
    exit 1
  fi
done

if [ -e "$audit_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed held-out audit found; refusing to overwrite: $audit_dir" >&2
    exit 1
  fi
  incomplete_dir="${audit_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$audit_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> Frozen CLIP patch-to-CLS risk-control held-out confirmation"
echo "==> Excludes all 13,847 proxy25 design paths before the image forward"
echo "==> Reuses locked full-target task/CLIP candidates; no ResNet replay"
echo "==> Exact frozen rule: stable upper median plus 1% pseudo-class mass cap"
echo "==> One deterministic CLIP center-crop forward on held-out conflicts"
echo "==> No backward, optimizer, parameter update, proxy training, or full training"

python tools/audit_visda_conflict_patch_cls_risk_control_holdout.py \
  --source-signal "$source_signal" \
  --source-lock "$source_lock" \
  --exploratory-summary "$exploratory_summary" \
  --output-dir "$audit_dir"

for artifact in \
  "${audit_dir}/visda_conflict_patch_cls_risk_control_holdout_label_free.npz" \
  "${audit_dir}/visda_conflict_patch_cls_risk_control_holdout_signal_lock.json" \
  "${audit_dir}/visda_conflict_patch_cls_risk_control_holdout_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_patch_cls_risk_control_holdout_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_patch_cls_risk_control_holdout_summary.md" \
  "$summary"; do
  if [ ! -s "$artifact" ]; then
    echo "Missing or empty held-out artifact: $artifact" >&2
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
    "source_or_task_model_loaded": False,
    "clip_model_frozen": True,
    "clip_image_forward_scope": "heldout_conflicts_only",
    "backward_calls": 0,
    "optimizer_constructed": False,
    "parameter_updates": 0,
    "proxy_training_started": False,
    "full_training_started": False,
    "full_training_authorized": False,
}
for key, expected in required.items():
    if safety.get(key) != expected:
        raise SystemExit(f"Held-out safety contract failed: {key}={safety.get(key)!r}")
if summary.get("decision") not in {
    "PASS_HELDOUT_PATCH_CLS_RISK_CONTROL",
    "REJECT",
}:
    raise SystemExit(f"Unexpected held-out decision: {summary.get('decision')!r}")
print(json.dumps({
    "decision": summary["decision"],
    "checks": summary["gate"]["checks"],
    "runtime_seconds": summary["runtime_seconds"],
}, indent=2))
PY

echo "==> Held-out audit complete: $summary"
echo "==> Even PASS authorizes one parameter-impact audit only, never training"
