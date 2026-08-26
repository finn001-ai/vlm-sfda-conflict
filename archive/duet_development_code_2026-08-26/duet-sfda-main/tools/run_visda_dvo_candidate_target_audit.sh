#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${base_dir}/cycle2_conflict_memory_snapshots/pre_cycle01.npz"
source_lock="${base_dir}/cycle2_conflict_memory_audit/visda_cycle2_conflict_memory_signal_lock.json"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/dvo_candidate_target_audit"

for path in "$snapshot" "$source_lock" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing DVO candidate-target audit input: $path" >&2
    echo "Run first: bash tools/run_visda_cycle2_conflict_memory_audit.sh" >&2
    exit 1
  fi
done

echo "==> CPU-only cycle-1 DUET DVO/TMI candidate-target audit"
echo "==> Changes only the CLIP-visual soft target on task/CLIP conflicts"
echo "==> Hard top-1, task CE, task CLIP KL, consistency, and masks stay fixed"
echo "==> Replays output-level TMI gradients over eight fixed batch orders"
echo "==> No image/model/checkpoint load, parameter update, optimizer, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_dvo_candidate_target.py \
  --snapshot "$snapshot" \
  --source-lock "$source_lock" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/visda_dvo_candidate_target_summary.json"
echo "==> PASS authorizes review of one matched proxy design only; no training was started"
