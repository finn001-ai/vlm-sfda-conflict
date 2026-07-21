#!/usr/bin/env bash
set -euo pipefail

for path in \
  data/VISDA-C/train_list.txt \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt; do
  if [ ! -f "$path" ]; then
    echo "Missing VisDA-C metadata: $path" >&2
    exit 1
  fi
done

dest="source/uda/VISDA-C/T"
for component in F B C; do
  if [ -f "$dest/source_${component}.pt" ]; then
    echo "Existing VisDA-C source checkpoint found in $dest; refusing to overwrite" >&2
    exit 1
  fi
done

echo "==> Training VisDA-C source model: train domain (ResNet-101, seed 2020)"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/source.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  SETTING.SEED 2020 SETTING.S 0 SETTING.T 1

mkdir -p "$dest"
for component in F B C; do
  source_path="source/source_${component}.pt"
  if [ ! -f "$source_path" ]; then
    echo "Source training did not produce $source_path" >&2
    exit 1
  fi
  mv "$source_path" "$dest/source_${component}.pt"
done

echo "Saved VisDA-C source checkpoints to $dest"
sha256sum "$dest"/source_{F,B,C}.pt
