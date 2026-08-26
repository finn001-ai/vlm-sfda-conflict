#!/usr/bin/env bash
set -euo pipefail

suppression_dir="output/uda/VISDA-C/TV/plmatch_visda_feature_gravity_audit_seed2020/feature_gravity_audit/patch_cls_holdout_audit/kl_suppression_impact_audit"
audit_dir="${suppression_dir}/pair_neutralization_audit"
summary="${audit_dir}/visda_patch_cls_pair_neutralization_summary.json"

for path in \
  "${suppression_dir}/visda_patch_cls_kl_suppression_impact_label_free.npz" \
  "${suppression_dir}/visda_patch_cls_kl_suppression_impact_signal_lock.json" \
  "${suppression_dir}/visda_patch_cls_kl_suppression_impact_oracle_diagnostic.csv" \
  "${suppression_dir}/visda_patch_cls_kl_suppression_impact_summary.json" \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$path" ]; then
    echo "Missing patch pair-neutralization audit input: $path" >&2
    echo "Run first: bash tools/run_visda_patch_cls_kl_suppression_impact_audit.sh" >&2
    exit 1
  fi
done

if [ -e "$audit_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed pair-neutralization audit found; refusing to overwrite: $audit_dir" >&2
    exit 1
  fi
  incomplete_dir="${audit_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$audit_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> CPU-only patch-selected task/CLIP pair-neutralization audit"
echo "==> Equalizes only the two candidate probabilities in the CLIP target"
echo "==> Preserves candidate-pair mass and every other class probability"
echo "==> No hard task label, mask/loss-weight change, image/model forward, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_patch_cls_pair_neutralization.py \
  --output-dir "$audit_dir"

for artifact in \
  "${audit_dir}/visda_patch_cls_pair_neutralization_label_free.npz" \
  "${audit_dir}/visda_patch_cls_pair_neutralization_signal_lock.json" \
  "${audit_dir}/visda_patch_cls_pair_neutralization_oracle_diagnostic.csv" \
  "${audit_dir}/visda_patch_cls_pair_neutralization_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_patch_cls_pair_neutralization_summary.md" \
  "$summary"; do
  if [ ! -s "$artifact" ]; then
    echo "Missing or empty pair-neutralization artifact: $artifact" >&2
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
        raise SystemExit(
            f"Pair-neutralization safety contract failed: {key}={safety.get(key)!r}"
        )
if summary.get("decision") not in {"NEEDS_EXACT_PARAMETER_AUDIT", "REJECT"}:
    raise SystemExit(f"Unexpected decision: {summary.get('decision')!r}")
print(json.dumps({
    "decision": summary["decision"],
    "checks": summary["gate"]["checks"],
    "runtime_seconds": summary["runtime_seconds"],
}, indent=2))
PY

echo "==> Audit complete: $summary"
echo "==> Even PASS authorizes one no-update parameter audit only, never training"
