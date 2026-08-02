#!/usr/bin/env bash
set -euo pipefail

seed=2020
probability_dir="output/uda/VISDA-C/TV/plmatch_visda_pairwise_attribute_audit_seed${seed}/pairwise_attribute_audit"
candidate_dir="output/uda/VISDA-C/TV/plmatch_visda_candidate_set_audit_seed${seed}/candidate_set_audit"
run_dir="output/uda/VISDA-C/TV/plmatch_visda_support_conditioned_clip_audit_seed${seed}"
audit_dir="${run_dir}/support_conditioned_clip_audit"
summary="${audit_dir}/visda_conflict_support_conditioned_clip_summary.json"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  "${probability_dir}/visda_conflict_pairwise_attribute_signals.npz" \
  "${probability_dir}/visda_conflict_pairwise_attribute_signal_lock.json" \
  "${candidate_dir}/visda_conflict_candidate_set_label_free.npz" \
  "${candidate_dir}/visda_conflict_candidate_set_signal_lock.json"; do
  if [ ! -f "$path" ]; then
    echo "Missing support-conditioned CLIP audit input: $path" >&2
    exit 1
  fi
done

if [ -e "$run_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed support-conditioned CLIP audit found; refusing to overwrite: $run_dir" >&2
    exit 1
  fi
  incomplete_dir="${run_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$run_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> CPU-only support-conditioned CLIP target audit"
echo "==> Candidate keeps CLIP relative mass inside task/CLIP top-2 union"
echo "==> No image/model/checkpoint load, forward, backward, optimizer, or training"
CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_support_conditioned_clip.py \
  --probability-dir "$probability_dir" \
  --candidate-dir "$candidate_dir" \
  --output-dir "$audit_dir" \
  --target-list data/VISDA-C/validation_list.txt \
  --class-names data/VISDA-C/classname.txt \
  --seed "$seed"

for artifact in \
  "${audit_dir}/visda_conflict_support_conditioned_clip_label_free.npz" \
  "${audit_dir}/visda_conflict_support_conditioned_clip_class_mass.csv" \
  "${audit_dir}/visda_conflict_support_conditioned_clip_signal_lock.json" \
  "${audit_dir}/visda_conflict_support_conditioned_clip_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_support_conditioned_clip_classwise_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_support_conditioned_clip_summary.md" \
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
if summary.get("decision") not in {
    "PASS_SUPPORT_CONDITIONED_CLIP_PREFLIGHT",
    "REJECT",
}:
    raise SystemExit(f"Unexpected decision: {summary.get('decision')!r}")
oracle = summary["oracle_metrics"]
label_free = summary["label_free_metrics"]
print(json.dumps({
    "decision": summary["decision"],
    "checks": summary["gate"]["checks"],
    "candidate_label_free": label_free["targets"]["top2_union"],
    "method_metrics": oracle["methods"],
    "comparisons": oracle["comparisons"],
    "minimum_class_first_order_delta_vs_clip": oracle[
        "minimum_class_first_order_delta_vs_clip"
    ],
    "class_macro_first_order_delta_vs_clip": oracle[
        "class_macro_first_order_delta_vs_clip"
    ],
    "runtime_seconds": summary["runtime_seconds"],
}, indent=2))
PY

echo "==> Audit complete: $summary"
echo "==> PASS_SUPPORT_CONDITIONED_CLIP_PREFLIGHT authorizes one matched proxy design only"
