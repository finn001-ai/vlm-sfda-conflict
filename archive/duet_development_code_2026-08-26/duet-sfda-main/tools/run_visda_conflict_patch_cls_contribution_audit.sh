#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${base_dir}/cycle2_conflict_memory_snapshots/pre_cycle01.npz"
source_lock="${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/patch_cls_contribution_audit"
summary="${audit_dir}/visda_conflict_patch_cls_contribution_summary.json"

for path in "$snapshot" "$source_lock" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing patch-to-CLS contribution audit input: $path" >&2
    echo "Run first: bash tools/run_visda_cycle2_conflict_memory_audit.sh" >&2
    exit 1
  fi
done

if [ -e "$audit_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed patch-to-CLS audit found; refusing to overwrite: $audit_dir" >&2
    exit 1
  fi
  incomplete_dir="${audit_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$audit_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> Frozen CLIP ViT-B/32 patch-to-CLS contribution preflight"
echo "==> Scope: 7,070 locked proxy25 task/CLIP conflicts"
echo "==> Default fixed CLIP; task rescue requires full/even/odd-head unanimity"
echo "==> One deterministic CLIP center-crop forward; no task/source model"
echo "==> No optimizer, backward, parameter update, proxy training, or full training"

python tools/audit_visda_conflict_patch_cls_contribution.py \
  --snapshot "$snapshot" \
  --source-lock "$source_lock" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --output-dir "$audit_dir"

for artifact in \
  "${audit_dir}/visda_conflict_patch_cls_contribution_label_free.npz" \
  "${audit_dir}/visda_conflict_patch_cls_contribution_signal_lock.json" \
  "${audit_dir}/visda_conflict_patch_cls_contribution_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_patch_cls_contribution_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_patch_cls_contribution_summary.md" \
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
    "task_model_loaded": False,
    "source_checkpoint_loaded": False,
    "clip_parameters_frozen": True,
    "optimizer_constructed": False,
    "backward_calls": 0,
    "parameter_updates": 0,
    "training_authorized": False,
}
for key, expected in required.items():
    if safety.get(key) != expected:
        raise SystemExit(f"Audit safety contract failed: {key}={safety.get(key)!r}")
if summary.get("decision") not in {
    "PASS_PATCH_CLS_CONTRIBUTION_PREFLIGHT",
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
echo "==> Even PASS authorizes one parameter-impact audit only; no training was started"
