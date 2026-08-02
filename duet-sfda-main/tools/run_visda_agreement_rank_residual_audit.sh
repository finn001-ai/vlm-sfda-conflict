#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${base_dir}/cycle2_conflict_memory_snapshots/pre_cycle01.npz"
source_lock="${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/agreement_rank_residual_audit"

for path in "$snapshot" "$source_lock" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing agreement rank-residual audit input: $path" >&2
    echo "Run first: bash tools/run_visda_cycle2_conflict_memory_audit.sh" >&2
    exit 1
  fi
done

echo "==> CPU-only cycle-1 DUET false-agreement rank-residual audit"
echo "==> Reuses the locked pre-cycle-1 snapshot; no GPU process is started"
echo "==> Signal: task/CLIP agree on top-1 but oppose each other's runner-up"
echo "==> Comparison: identical per-class coverage versus three confidence filters"
echo "==> No image, checkpoint, model, forward, backward, optimizer, or training"

python tools/audit_visda_agreement_rank_residual.py \
  --snapshot "$snapshot" \
  --source-lock "$source_lock" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/visda_agreement_rank_residual_summary.json"
echo "==> PASS authorizes method design only; no proxy or full training was started"
