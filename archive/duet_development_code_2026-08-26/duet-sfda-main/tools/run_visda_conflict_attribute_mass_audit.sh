#!/usr/bin/env bash
set -euo pipefail

seed=2020
input_dir="output/uda/VISDA-C/TV/plmatch_visda_pairwise_attribute_audit_seed${seed}/pairwise_attribute_audit"
run_dir="output/uda/VISDA-C/TV/plmatch_visda_attribute_mass_audit_seed${seed}"
audit_dir="${run_dir}/attribute_mass_audit"
summary="${audit_dir}/visda_conflict_attribute_mass_summary.json"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  "${input_dir}/visda_conflict_pairwise_attribute_signals.npz" \
  "${input_dir}/visda_conflict_pairwise_attribute_signal_lock.json" \
  "${input_dir}/visda_conflict_pairwise_attribute_summary.json"; do
  if [ ! -f "$path" ]; then
    echo "Missing attribute-mass audit input: $path" >&2
    exit 1
  fi
done

if [ -e "$run_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed attribute-mass audit found; refusing to overwrite: $run_dir" >&2
    exit 1
  fi
  incomplete_dir="${run_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$run_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> CPU-only VisDA pairwise attribute probability-mass audit"
echo "==> Reads the previously locked NPZ; does not read its path-bearing CSV"
echo "==> CLIP ViT-B/32 is loaded only to recover its frozen logit_scale scalar"
echo "==> No target image load, no model forward, no optimizer, no backward, no training"
CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_attribute_mass.py \
  --input-dir "$input_dir" \
  --output-dir "$audit_dir" \
  --target-list data/VISDA-C/validation_list.txt \
  --class-names data/VISDA-C/classname.txt \
  --clip-architecture ViT-B/32 \
  --seed "$seed"

for artifact in \
  "${audit_dir}/visda_conflict_attribute_mass_target.npz" \
  "${audit_dir}/visda_conflict_attribute_mass_class_mass.csv" \
  "${audit_dir}/visda_conflict_attribute_mass_target_lock.json" \
  "${audit_dir}/visda_conflict_attribute_mass_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_attribute_mass_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_attribute_mass_summary.md" \
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
required = {
    "oracle_diagnostic": True,
    "labels_used_only_after_target_lock": True,
}
for key, expected in required.items():
    if summary.get(key) != expected:
        raise SystemExit(f"Audit safety contract failed: {key}={summary.get(key)!r}")
safety = summary.get("safety_contract", {})
required_safety = {
    "target_images_loaded": False,
    "model_forward_calls": 0,
    "optimizer_constructed": False,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "model_parameters_updated": False,
    "training_code_modified": False,
    "training_authorized": False,
}
for key, expected in required_safety.items():
    if safety.get(key) != expected:
        raise SystemExit(f"Audit safety contract failed: {key}={safety.get(key)!r}")
if summary.get("decision") not in {"PASS_OFFLINE_GATE", "REJECT"}:
    raise SystemExit(f"Unexpected audit decision: {summary.get('decision')!r}")
print(json.dumps({
    "decision": summary["decision"],
    "checks": summary["gate"]["checks"],
    "runtime_seconds": summary["runtime_seconds"],
}, indent=2))
PY

echo "==> Audit complete: $summary"
echo "==> PASS_OFFLINE_GATE still does not authorize or start training"
