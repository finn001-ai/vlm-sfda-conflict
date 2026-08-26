#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot_dir="${base_dir}/cycle2_conflict_memory_snapshots"
pre_cycle1="${snapshot_dir}/pre_cycle01.npz"
pre_cycle2="${snapshot_dir}/pre_cycle02.npz"
source_lock="${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/agreement_revocation_audit"

for path in "$pre_cycle1" "$pre_cycle2" "$source_lock" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing agreement revocation audit input: $path" >&2
    echo "Run first: bash tools/run_visda_cycle2_conflict_memory_audit.sh" >&2
    exit 1
  fi
done

echo "==> CPU-only cycle-2 DUET stale-agreement revocation audit"
echo "==> Compares reversible current agreement with the production monotonic mask"
echo "==> Matched comparators revoke identical per-class counts by four confidences"
echo "==> No image/model/checkpoint load, forward, backward, optimizer, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_agreement_revocation.py \
  --pre-cycle1 "$pre_cycle1" \
  --pre-cycle2 "$pre_cycle2" \
  --source-lock "$source_lock" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/visda_agreement_revocation_summary.json"
echo "==> PASS authorizes design review only; no proxy or full training was started"
