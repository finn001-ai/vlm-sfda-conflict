#!/usr/bin/env bash
set -euo pipefail

seed=2020
input_dir="output/uda/VISDA-C/TV/plmatch_visda_pairwise_attribute_audit_seed${seed}/pairwise_attribute_audit"
run_dir="output/uda/VISDA-C/TV/plmatch_visda_candidate_set_audit_seed${seed}"
audit_dir="${run_dir}/candidate_set_audit"
summary="${audit_dir}/visda_conflict_candidate_set_summary.json"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  "${input_dir}/visda_conflict_pairwise_attribute_signals.npz" \
  "${input_dir}/visda_conflict_pairwise_attribute_signal_lock.json" \
  "${input_dir}/visda_conflict_pairwise_attribute_summary.json"; do
  if [ ! -f "$path" ]; then
    echo "Missing candidate-set audit input: $path" >&2
    exit 1
  fi
done

if [ -e "$run_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed candidate-set audit found; refusing to overwrite: $run_dir" >&2
    exit 1
  fi
  incomplete_dir="${run_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$run_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> CPU-only VisDA task/CLIP top-1 versus top-2 candidate-set audit"
echo "==> Reads the previously locked probability NPZ; no source CSV is read"
echo "==> No target image, model/CLIP load, checkpoint, forward, backward, optimizer, or training"
CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_candidate_set.py \
  --input-dir "$input_dir" \
  --output-dir "$audit_dir" \
  --target-list data/VISDA-C/validation_list.txt \
  --class-names data/VISDA-C/classname.txt \
  --seed "$seed"

for artifact in \
  "${audit_dir}/visda_conflict_candidate_set_label_free.npz" \
  "${audit_dir}/visda_conflict_candidate_set_size_distribution.csv" \
  "${audit_dir}/visda_conflict_candidate_set_signal_lock.json" \
  "${audit_dir}/visda_conflict_candidate_set_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_candidate_set_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_candidate_set_summary.md" \
  "$summary"; do
  if [ ! -s "$artifact" ]; then
    echo "Missing or empty audit artifact: $artifact" >&2
    exit 1
  fi
done

python - "$summary" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
if summary.get("oracle_diagnostic") is not True:
    raise SystemExit("Oracle diagnostic marker is missing")
if summary.get("labels_used_only_after_signal_lock") is not True:
    raise SystemExit("Label-lock ordering contract failed")
safety = summary.get("safety_contract", {})
required = {
    "target_images_loaded": False,
    "model_checkpoint_loads": 0,
    "model_forward_calls": 0,
    "optimizer_constructed": False,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "model_parameters_updated": False,
    "training_code_modified": False,
    "training_authorized": False,
}
for key, expected in required.items():
    if safety.get(key) != expected:
        raise SystemExit(f"Safety contract failed: {key}={safety.get(key)!r}")
if summary.get("decision") not in {"PASS_CANDIDATE_SET_PREFLIGHT", "REJECT"}:
    raise SystemExit(f"Unexpected decision: {summary.get('decision')!r}")
metrics = summary["oracle_metrics"]
print(json.dumps({
    "decision": summary["decision"],
    "checks": summary["gate"]["checks"],
    "top1_union_coverage_pct": metrics["top1_union_coverage_pct"],
    "top2_union_coverage_pct": metrics["top2_union_coverage_pct"],
    "top2_minus_top1_coverage_pp": metrics["top2_minus_top1_coverage_pp"],
    "recovered_top1_misses_pct": metrics["recovered_top1_misses_pct"],
    "mean_top2_set_size": summary["label_free_metrics"]["top2_union_set_size_mean"],
    "minimum_class_top2_coverage_pct": metrics["minimum_class_top2_coverage_pct"],
    "car_top2_coverage_pct": metrics["car_top2_coverage_pct"],
    "truck_top2_coverage_pct": metrics["truck_top2_coverage_pct"],
    "runtime_seconds": summary["runtime_seconds"],
}, indent=2))
PY

echo "==> Audit complete: $summary"
echo "==> PASS_CANDIDATE_SET_PREFLIGHT authorizes method design only, never training"
