#!/usr/bin/env bash
set -euo pipefail

base_dir="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${base_dir}/cycle2_conflict_memory_snapshots/pre_cycle02.npz"
pcgrad_dir="${base_dir}/conflict_pcgrad_audit"
pcgrad_signal="${pcgrad_dir}/visda_conflict_pcgrad_label_free.npz"
pcgrad_lock="${pcgrad_dir}/visda_conflict_pcgrad_signal_lock.json"
pcgrad_summary="${pcgrad_dir}/visda_conflict_pcgrad_summary.json"
source_classifier="source/uda/VISDA-C/T/source_C.pt"
target_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
class_names="data/VISDA-C/classname.txt"
audit_dir="${base_dir}/conflict_pcgrad_feature_jacobian_audit"

for path in \
  "$snapshot" "$pcgrad_signal" "$pcgrad_lock" "$pcgrad_summary" \
  "$source_classifier" "$target_list" "$class_names"; do
  if [ ! -f "$path" ]; then
    echo "Missing PCGrad feature-Jacobian input: $path" >&2
    echo "Run first: bash tools/run_visda_conflict_pcgrad_audit.sh" >&2
    exit 1
  fi
done

echo "==> CPU-only frozen-classifier feature-Jacobian audit"
echo "==> Maps the locked PCGrad output directions through source_C.pt"
echo "==> Loads no target image, ResNet, bottleneck, or CLIP model"
echo "==> No model forward, backward, optimizer, parameter update, or training"

CUDA_VISIBLE_DEVICES="" python tools/audit_visda_conflict_pcgrad_feature_jacobian.py \
  --snapshot "$snapshot" \
  --pcgrad-signal "$pcgrad_signal" \
  --pcgrad-lock "$pcgrad_lock" \
  --pcgrad-summary "$pcgrad_summary" \
  --source-classifier "$source_classifier" \
  --target-list "$target_list" \
  --class-names "$class_names" \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/visda_conflict_pcgrad_feature_jacobian_summary.json"
echo "==> Even PASS authorizes no GPU, proxy, or full training"
