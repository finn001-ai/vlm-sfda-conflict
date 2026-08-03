#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${base_dir}/cycle2_conflict_memory_snapshots/pre_cycle01.npz"
source_lock="${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/agreement_label_impact_audit"

for path in "$snapshot" "$source_lock" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing agreement label-impact audit input: $path" >&2
    echo "Run first: bash tools/run_visda_cycle2_conflict_memory_audit.sh" >&2
    exit 1
  fi
done

echo "==> CPU-only VisDA agreement-anchored label-impact audit"
echo "==> References: 6,777 cycle-1 task/CLIP agreements"
echo "==> Queries: 7,070 conflicts; choices stay inside task-top2 union CLIP-top2"
echo "==> Signal: diagonal-Fisher classifier-head impact on agreement loss"
echo "==> No image/model/checkpoint load, forward, backward, optimizer, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_agreement_label_impact.py \
  --snapshot "$snapshot" \
  --source-lock "$source_lock" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/visda_conflict_agreement_label_impact_summary.json"
echo "==> PASS authorizes one exact parameter audit only; no training was started"
