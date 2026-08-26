#!/usr/bin/env bash
set -euo pipefail

seed=2020
probability_dir="output/uda/VISDA-C/TV/plmatch_visda_pairwise_attribute_audit_seed${seed}/pairwise_attribute_audit"
candidate_dir="output/uda/VISDA-C/TV/plmatch_visda_candidate_set_audit_seed${seed}/candidate_set_audit"
run_dir="output/uda/VISDA-C/TV/plmatch_visda_candidate_set_gradient_audit_seed${seed}"
audit_dir="${run_dir}/candidate_set_gradient_audit"
summary="${audit_dir}/visda_conflict_candidate_set_gradient_summary.json"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  "${probability_dir}/visda_conflict_pairwise_attribute_signals.npz" \
  "${probability_dir}/visda_conflict_pairwise_attribute_signal_lock.json" \
  "${candidate_dir}/visda_conflict_candidate_set_label_free.npz" \
  "${candidate_dir}/visda_conflict_candidate_set_signal_lock.json"; do
  if [ ! -f "$path" ]; then
    echo "Missing candidate-set gradient input: $path" >&2
    exit 1
  fi
done

if [ -e "$run_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed candidate-set gradient audit found; refusing to overwrite: $run_dir" >&2
    exit 1
  fi
  incomplete_dir="${run_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$run_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> CPU-only exact candidate-set logit-gradient audit"
echo "==> Compares DUET CLIP KL, top-1 set loss, and top-2 set loss"
echo "==> No image/model/checkpoint load, forward, backward, optimizer, or training"
CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_candidate_set_gradient.py \
  --probability-dir "$probability_dir" \
  --candidate-dir "$candidate_dir" \
  --output-dir "$audit_dir" \
  --target-list data/VISDA-C/validation_list.txt \
  --class-names data/VISDA-C/classname.txt \
  --seed "$seed"

for artifact in \
  "${audit_dir}/visda_conflict_candidate_set_gradient_label_free.npz" \
  "${audit_dir}/visda_conflict_candidate_set_gradient_signal_lock.json" \
  "${audit_dir}/visda_conflict_candidate_set_gradient_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_candidate_set_gradient_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_candidate_set_gradient_summary.md" \
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
if summary.get("decision") not in {"PASS_SET_GRADIENT_PREFLIGHT", "REJECT"}:
    raise SystemExit(f"Unexpected decision: {summary.get('decision')!r}")
metrics = summary["oracle_metrics"]
print(json.dumps({
    "decision": summary["decision"],
    "checks": summary["gate"]["checks"],
    "method_metrics": metrics["methods"],
    "comparisons": metrics["comparisons"],
    "class_macro_first_order_delta_vs_clip": metrics[
        "class_macro_first_order_delta_vs_clip"
    ],
    "hard_class_first_order_delta_vs_clip": metrics[
        "hard_class_first_order_delta_vs_clip"
    ],
    "runtime_seconds": summary["runtime_seconds"],
}, indent=2))
PY

echo "==> Audit complete: $summary"
echo "==> PASS_SET_GRADIENT_PREFLIGHT authorizes one matched proxy design only, never training"
