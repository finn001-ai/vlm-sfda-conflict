#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${base_dir}/cycle2_conflict_memory_snapshots/pre_cycle02.npz"
source_lock="${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/agreement_neighbor_clip_cycle2_audit"
stem="visda_conflict_agreement_neighbor_clip_cycle2"

for path in "$snapshot" "$source_lock" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing cycle-2 agreement-neighbor audit input: $path" >&2
    echo "Run first: bash tools/run_visda_cycle2_conflict_memory_audit.sh" >&2
    exit 1
  fi
done

echo "==> CPU-only VisDA cycle-2 agreement-neighbor CLIP timing audit"
echo "==> Same fixed K=5 rule; no K search, class route, or fitted threshold"
echo "==> References: currently agreeing admitted samples in adapted task space"
echo "==> Queries: currently conflicting samples still unresolved by DUET"
echo "==> No image/model/checkpoint load, forward, backward, optimizer, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_agreement_neighbor_clip.py \
  --snapshot "$snapshot" \
  --source-lock "$source_lock" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --expected-cycle 2 \
  --query-mode unresolved_current_conflicts \
  --stem "$stem" \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/${stem}_summary.json"
echo "==> PASS authorizes one matched-proxy design review only; no training was started"
