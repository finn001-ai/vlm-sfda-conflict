#!/usr/bin/env bash
set -euo pipefail

seed=2020
source_dir="output/uda/VISDA-C/TV/plmatch_visda_pairwise_attribute_audit_seed${seed}/pairwise_attribute_audit"
reliability_dir="output/uda/VISDA-C/TV/plmatch_visda_attribute_reliability_audit_seed${seed}/attribute_reliability_audit"
candidate_dir="output/uda/VISDA-C/TV/duet_attribute_reliability_kl_visda_proxy25_seed${seed}"
proxy_gate="output/uda/VISDA-C/duet_attribute_reliability_kl_visda_proxy25_seed2020_gate.json"
run_dir="output/uda/VISDA-C/TV/plmatch_visda_attribute_kl_influence_audit_seed${seed}"
audit_dir="${run_dir}/attribute_kl_influence_audit"
summary="${audit_dir}/visda_conflict_attribute_kl_influence_summary.json"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/validation_proxy25_seed2020_list.txt \
  data/VISDA-C/classname.txt \
  "${source_dir}/visda_conflict_pairwise_attribute_signals.npz" \
  "${source_dir}/visda_conflict_pairwise_attribute_signal_lock.json" \
  "${source_dir}/visda_conflict_pairwise_attribute_summary.json" \
  "${reliability_dir}/visda_conflict_attribute_reliability_target.npz" \
  "${reliability_dir}/visda_conflict_attribute_reliability_target_lock.json" \
  "${reliability_dir}/visda_conflict_attribute_reliability_summary.json" \
  "$proxy_gate"; do
  if [ ! -f "$path" ]; then
    echo "Missing attribute KL-influence audit input: $path" >&2
    exit 1
  fi
done

candidate_logs=("${candidate_dir}"/*.txt)
if [ "${#candidate_logs[@]}" -ne 1 ] || [ ! -f "${candidate_logs[0]}" ]; then
  echo "Expected exactly one completed attribute-reliability candidate log" >&2
  exit 1
fi

if [ -e "$run_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed KL-influence audit found; refusing to overwrite: $run_dir" >&2
    exit 1
  fi
  incomplete_dir="${run_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$run_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> CPU-only exact attribute-KL logit influence audit"
echo "==> Uses only locked NPZ probabilities and the completed matched-proxy artifacts"
echo "==> No target image, checkpoint, model/CLIP load, forward, backward, optimizer, or training"
CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_attribute_kl_influence.py \
  --source-dir "$source_dir" \
  --reliability-dir "$reliability_dir" \
  --output-dir "$audit_dir" \
  --target-list data/VISDA-C/validation_list.txt \
  --proxy-list data/VISDA-C/validation_proxy25_seed2020_list.txt \
  --class-names data/VISDA-C/classname.txt \
  --candidate-log "${candidate_logs[0]}" \
  --proxy-gate "$proxy_gate" \
  --seed "$seed" \
  --kl-weight 0.4

for artifact in \
  "${audit_dir}/visda_conflict_attribute_kl_influence_direction.npz" \
  "${audit_dir}/visda_conflict_attribute_kl_influence_class_mass.csv" \
  "${audit_dir}/visda_conflict_attribute_kl_influence_direction_lock.json" \
  "${audit_dir}/visda_conflict_attribute_kl_influence_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_attribute_kl_influence_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_attribute_kl_influence_categorywise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_attribute_kl_influence_summary.md" \
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
if summary.get("labels_used_only_after_direction_lock") is not True:
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
if summary.get("decision") not in {
    "REJECT_ATTRIBUTE_BRANCH",
    "PASS_DIAGNOSTIC_ONLY",
}:
    raise SystemExit(f"Unexpected decision: {summary.get('decision')!r}")
oracle = summary["oracle_metrics"]
print(json.dumps({
    "decision": summary["decision"],
    "diagnosis": summary["gate"]["diagnosis"],
    "active_conflicts": summary["label_free_diagnostic"]["active_conflicts"],
    "changed_top1": summary["label_free_diagnostic"]["changed_top1"],
    "increment_vs_control_norm_pct": summary["label_free_diagnostic"]["increment_vs_control_norm_pct"],
    "mean_incremental_projection": oracle["mean_incremental_projection"],
    "projection_ci": oracle["incremental_projection_bootstrap_95_ci"],
    "helpful_coverage_pct": oracle["helpful_coverage_pct"],
    "harmful_coverage_pct": oracle["harmful_coverage_pct"],
    "car_projection": next(row["mean_incremental_projection"] for row in oracle["classwise"] if row["class"] == "car"),
    "truck_projection": next(row["mean_incremental_projection"] for row in oracle["classwise"] if row["class"] == "truck"),
    "runtime_seconds": summary["runtime_seconds"],
}, indent=2))
PY

echo "==> Audit complete: $summary"
echo "==> This diagnostic never authorizes another proxy or full VisDA run"
