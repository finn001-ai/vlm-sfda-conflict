#!/usr/bin/env bash
set -euo pipefail

feature_dir="output/uda/VISDA-C/TV/plmatch_visda_feature_gravity_audit_seed2020/feature_gravity_audit"
holdout_dir="${feature_dir}/patch_cls_holdout_audit"
audit_dir="${holdout_dir}/kl_suppression_impact_audit"
summary="${audit_dir}/visda_patch_cls_kl_suppression_impact_summary.json"

for path in \
  "${feature_dir}/visda_conflict_feature_gravity_signals.npz" \
  "${feature_dir}/visda_conflict_feature_gravity_signal_lock.json" \
  "${holdout_dir}/visda_conflict_patch_cls_risk_control_holdout_label_free.npz" \
  "${holdout_dir}/visda_conflict_patch_cls_risk_control_holdout_signal_lock.json" \
  "${holdout_dir}/visda_conflict_patch_cls_risk_control_holdout_oracle_diagnostic.csv" \
  "${holdout_dir}/visda_conflict_patch_cls_risk_control_holdout_summary.json" \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$path" ]; then
    echo "Missing patch KL-suppression audit input: $path" >&2
    exit 1
  fi
done

if [ -e "$audit_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed KL-suppression audit found; refusing to overwrite: $audit_dir" >&2
    exit 1
  fi
  incomplete_dir="${audit_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$audit_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> CPU-only patch-selected CLIP-KL suppression impact audit"
echo "==> Candidate removes CLIP-KL only on the 613 locked held-out rescues"
echo "==> No task hard pseudo-label; consistency and all other rows stay fixed"
echo "==> Replays locked logits and exact frozen source_C.pt feature Jacobian"
echo "==> No image/model forward, backward, optimizer, proxy run, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_patch_cls_kl_suppression_impact.py \
  --output-dir "$audit_dir"

for artifact in \
  "${audit_dir}/visda_patch_cls_kl_suppression_impact_label_free.npz" \
  "${audit_dir}/visda_patch_cls_kl_suppression_impact_signal_lock.json" \
  "${audit_dir}/visda_patch_cls_kl_suppression_impact_oracle_diagnostic.csv" \
  "${audit_dir}/visda_patch_cls_kl_suppression_impact_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_patch_cls_kl_suppression_impact_summary.md" \
  "$summary"; do
  if [ ! -s "$artifact" ]; then
    echo "Missing or empty KL-suppression artifact: $artifact" >&2
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
    "resnet_bottleneck_or_clip_loaded": False,
    "classifier_checkpoint_loaded_on_cpu": True,
    "model_forward_calls": 0,
    "backward_calls": 0,
    "optimizer_constructed": False,
    "parameter_updates": 0,
    "proxy_authorized": False,
    "training_authorized": False,
}
for key, expected in required.items():
    if safety.get(key) != expected:
        raise SystemExit(f"KL-suppression safety contract failed: {key}={safety.get(key)!r}")
if summary.get("decision") not in {"NEEDS_EXACT_PARAMETER_AUDIT", "REJECT"}:
    raise SystemExit(f"Unexpected decision: {summary.get('decision')!r}")
print(json.dumps({
    "decision": summary["decision"],
    "checks": summary["gate"]["checks"],
    "runtime_seconds": summary["runtime_seconds"],
}, indent=2))
PY

echo "==> Impact audit complete: $summary"
echo "==> Even PASS authorizes one no-update parameter audit only, never training"
