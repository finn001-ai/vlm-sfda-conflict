#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot_dir="${base_dir}/cycle2_conflict_memory_snapshots"
patch_dir="${base_dir}/patch_cls_contribution_audit/risk_control_audit"
holdout_dir="output/uda/VISDA-C/TV/plmatch_visda_feature_gravity_audit_seed2020/feature_gravity_audit/patch_cls_holdout_audit"
audit_dir="${patch_dir}/temporal_persistence_audit"
summary="${audit_dir}/visda_patch_cls_temporal_persistence_summary.json"

for path in \
  "${snapshot_dir}/pre_cycle01.npz" \
  "${snapshot_dir}/pre_cycle02.npz" \
  "${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json" \
  "${patch_dir}/visda_conflict_patch_cls_risk_control_label_free.npz" \
  "${patch_dir}/visda_conflict_patch_cls_risk_control_signal_lock.json" \
  "${patch_dir}/visda_conflict_patch_cls_risk_control_summary.json" \
  "${holdout_dir}/visda_conflict_patch_cls_risk_control_holdout_summary.json" \
  data/VISDA-C/validation_proxy25_seed2020_list.txt \
  data/VISDA-C/classname.txt; do
  if [ ! -f "$path" ]; then
    echo "Missing patch temporal-persistence input: $path" >&2
    exit 1
  fi
done

if [ -e "$audit_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed temporal-persistence audit found; refusing to overwrite: $audit_dir" >&2
    exit 1
  fi
  incomplete_dir="${audit_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$audit_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> CPU-only frozen patch-to-CLS temporal-persistence audit"
echo "==> Reuses the 208 locked proxy patch rescues and pre-cycle-2 snapshot"
echo "==> Candidate changes inference prediction only; no loss is changed"
echo "==> Controls: cycle-2 task, CLIP, confidence, arithmetic, RMS, and mixed"
echo "==> Snapshot provenance is exploratory and not pure-DUET confirmation"
echo "==> No image/model/checkpoint, forward, backward, optimizer, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_patch_cls_temporal_persistence.py \
  --output-dir "$audit_dir"

for artifact in \
  "${audit_dir}/visda_patch_cls_temporal_persistence_label_free.npz" \
  "${audit_dir}/visda_patch_cls_temporal_persistence_signal_lock.json" \
  "${audit_dir}/visda_patch_cls_temporal_persistence_oracle_diagnostic.csv" \
  "${audit_dir}/visda_patch_cls_temporal_persistence_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_patch_cls_temporal_persistence_summary.md" \
  "$summary"; do
  if [ ! -s "$artifact" ]; then
    echo "Missing or empty temporal-persistence artifact: $artifact" >&2
    exit 1
  fi
done

python - "$summary" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    summary = json.load(handle)
safety = summary["safety"]
for key in (
    "target_images_loaded", "model_or_checkpoint_loaded", "forward_calls",
    "backward_calls", "optimizer_constructed", "parameter_updates",
    "proxy_training_authorized", "full_training_authorized",
):
    if safety.get(key) not in (False, 0):
        raise SystemExit(f"Temporal-persistence safety contract failed: {key}")
if summary.get("decision") not in {
    "PASS_EXPLORATORY_PATCH_TEMPORAL_PERSISTENCE", "REJECT"
}:
    raise SystemExit(f"Unexpected temporal-persistence decision: {summary.get('decision')!r}")
print(json.dumps({
    "decision": summary["decision"],
    "checks": summary["gate"]["checks"],
    "runtime_seconds": summary["runtime_seconds"],
}, indent=2))
PY

echo "==> Audit complete: $summary"
echo "==> Even PASS authorizes only a pure-DUET cycle-2 snapshot confirmation"
