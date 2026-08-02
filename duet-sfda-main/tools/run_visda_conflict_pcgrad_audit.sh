#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${base_dir}/cycle2_conflict_memory_snapshots/pre_cycle02.npz"
source_lock="${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/conflict_pcgrad_audit"

for path in "$snapshot" "$source_lock" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing conflict-PCGrad audit input: $path" >&2
    echo "Run first: bash tools/run_visda_cycle2_conflict_memory_audit.sh" >&2
    exit 1
  fi
done

echo "==> CPU-only cycle-2 DUET KL/consistency output-gradient audit"
echo "==> Scope: 1,978 currently unresolved task/CLIP conflicts"
echo "==> Candidate: parameter-free symmetric two-objective PCGrad"
echo "==> Replays three probability floors to audit float32 underflow stability"
echo "==> No image/model/checkpoint load, forward, backward, optimizer, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_pcgrad.py \
  --snapshot "$snapshot" \
  --source-lock "$source_lock" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --expected-query-count 1978 \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/visda_conflict_pcgrad_summary.json"
echo "==> NEEDS_PARAMETER_AUDIT still authorizes no proxy/full training"
