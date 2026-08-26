#!/usr/bin/env bash
set -euo pipefail

seed=2020
method="plmatch_visda_spatial_causal_audit_seed${seed}"
run_dir="output/uda/VISDA-C/TV/${method}"
audit_dir="${run_dir}/spatial_causal_audit"
summary="${audit_dir}/visda_conflict_spatial_causal_summary.json"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$path" ]; then
    echo "Missing VisDA-C audit input: $path" >&2
    exit 1
  fi
done

if [ -e "$run_dir" ]; then
  if [ -s "$summary" ]; then
    echo "Completed spatial-causal audit found; refusing to overwrite: $run_dir" >&2
    exit 1
  fi
  incomplete_dir="${run_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
  mv -- "$run_dir" "$incomplete_dir"
  echo "==> Archived incomplete audit: $incomplete_dir"
fi

echo "==> Frozen VisDA spatial-causal conflict audit"
echo "==> Same task model and CLIP ViT-B/32; 1,024 rows x 64 masks"
echo "==> No optimizer, no backward, no adaptation, no checkpoints"
python tools/audit_visda_conflict_spatial_causal.py \
  --cfg cfgs/visda/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.ADAPTATION_LIST ""

for artifact in \
  "${audit_dir}/visda_conflict_spatial_causal_signals.csv" \
  "${audit_dir}/visda_conflict_spatial_causal_signals.npz" \
  "${audit_dir}/visda_conflict_spatial_causal_signal_lock.json" \
  "${audit_dir}/visda_conflict_spatial_causal_oracle_diagnostic.csv" \
  "${audit_dir}/visda_conflict_spatial_causal_classwise_oracle_diagnostic.csv" \
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
    "labels_used_only_after_signal_lock": True,
    "optimizer_constructed": False,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "model_parameters_updated": False,
    "training_authorized": False,
}
for key, expected in required.items():
    if summary.get(key) != expected:
        raise SystemExit(f"Audit safety contract failed: {key}={summary.get(key)!r}")
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
