#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${base_dir}/cycle2_conflict_memory_snapshots/pre_cycle01.npz"
source_lock="${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
prior_lock="${base_dir}/conflict_vsfot_alignment_audit/visda_conflict_vsfot_alignment_signal_lock.json"
prior_summary="${base_dir}/conflict_vsfot_alignment_audit/visda_conflict_vsfot_alignment_summary.json"
source_classifier="source/uda/VISDA-C/T/source_C.pt"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/agreement_transport_full_gradient_audit"

for path in \
  "$snapshot" "$source_lock" "$prior_lock" "$prior_summary" \
  "$source_classifier" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing agreement-transport input: $path" >&2
    echo "This audit reuses completed locked artifacts; do not rerun GPU work." >&2
    exit 1
  fi
done

echo "==> CPU-only agreement-only transport full-DUET gradient audit"
echo "==> Candidate changes only the KL target on task/CLIP agreements"
echo "==> Controls: original full DUET and matched duplicate-hard-CE replacement"
echo "==> Eight fixed batch replays; conflicts and all loss weights stay fixed"
echo "==> No image/model forward, backward, optimizer, parameter update, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_agreement_transport_full_gradient.py \
  --snapshot "$snapshot" \
  --source-lock "$source_lock" \
  --prior-vsfot-lock "$prior_lock" \
  --prior-vsfot-summary "$prior_summary" \
  --source-classifier "$source_classifier" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/visda_agreement_transport_full_gradient_summary.json"
echo "==> Even PASS authorizes no GPU/proxy/full training"
