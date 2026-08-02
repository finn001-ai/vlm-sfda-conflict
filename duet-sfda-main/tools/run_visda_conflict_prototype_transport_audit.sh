#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${base_dir}/cycle2_conflict_memory_snapshots/pre_cycle01.npz"
source_lock="${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
source_classifier="source/uda/VISDA-C/T/source_C.pt"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/conflict_prototype_transport_audit"

for path in \
  "$snapshot" "$source_lock" "$source_classifier" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing prototype-transport input: $path" >&2
    echo "This audit reuses the completed cycle-2 memory snapshot; do not rerun GPU work." >&2
    exit 1
  fi
done

echo "==> CPU-only VisDA source-prototype/CLIP transport audit"
echo "==> Scope: the 7,070 locked pre-cycle-1 task/CLIP conflicts"
echo "==> Preserves exact fixed-CLIP conflict class counts; no fitted temperature"
echo "==> Loads only task features and source_C.pt; no target image/model forward"
echo "==> No backward, optimizer, parameter update, proxy run, or full training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_prototype_transport.py \
  --snapshot "$snapshot" \
  --source-lock "$source_lock" \
  --source-classifier "$source_classifier" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/visda_conflict_prototype_transport_summary.json"
echo "==> PASS authorizes design review only; no GPU/proxy/full training was started"
