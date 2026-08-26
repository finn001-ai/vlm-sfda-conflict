#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${base_dir}/cycle2_conflict_memory_snapshots/pre_cycle01.npz"
source_lock="${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
source_classifier="source/uda/VISDA-C/T/source_C.pt"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/conflict_vsfot_alignment_audit"

for path in \
  "$snapshot" "$source_lock" "$source_classifier" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing VSFOT-alignment input: $path" >&2
    echo "This audit reuses completed locked artifacts; do not rerun GPU work." >&2
    exit 1
  fi
done

echo "==> CPU-only isolated public-VSFOT alignment audit"
echo "==> Eight fixed batch-order replays; Sinkhorn reg=0.2 and batch size=64"
echo "==> Controls: DUET CLIP KL and identical transport classification-only"
echo "==> Loads only locked probabilities/features and source_C.pt"
echo "==> No image/model forward, backward, optimizer, parameter update, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_vsfot_alignment.py \
  --snapshot "$snapshot" \
  --source-lock "$source_lock" \
  --source-classifier "$source_classifier" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/visda_conflict_vsfot_alignment_summary.json"
echo "==> Even PASS authorizes design review only; no GPU/proxy/full training was started"
