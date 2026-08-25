# Cloud Start

This repo is prepared for cloud-side training and diagnostics. The local Mac is
only for code editing and lightweight checks.

## 1. Clone and create environment

```bash
git clone <your-github-repo-url>
cd <repo>/duet-sfda-main
conda env create -f environment.yml
conda activate sfa
```

If the full `environment.yml` is too old for the cloud image, start from it and
install missing packages as errors appear. The first required package already
seen locally is `iopath`.

## 2. Prepare Office-Home

Put Office-Home under:

```text
duet-sfda-main/data/office-home/
  Art/
  Clipart/
  Product/
  RealWorld/
```

Then build DUET list files:

```bash
python tools/prepare_office_home_lists.py --root data/office-home
```

This creates:

```text
data/office-home/Art_list.txt
data/office-home/Clipart_list.txt
data/office-home/Product_list.txt
data/office-home/RealWorld_list.txt
data/office-home/classname.txt
```

## 3. Prepare source checkpoints

Download or train source checkpoints, then place them like:

```text
duet-sfda-main/source/uda/office-home/A/source_F.pt
duet-sfda-main/source/uda/office-home/A/source_B.pt
duet-sfda-main/source/uda/office-home/A/source_C.pt
```

For source domain `Clipart`, `Product`, or `RealWorld`, use folders `C`, `P`,
or `R`.

## 4. Run the first diagnostic

Start with one pair, `Art -> Clipart`:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_conflict_diagnostics.py \
  --cfg cfgs/office-home/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  SETTING.S 0 SETTING.T 1 GPU_ID 0
```

Outputs:

```text
output/uda/office-home/AC/plmatch/diagnostics/AC_conflicts.csv
output/uda/office-home/AC/plmatch/diagnostics/AC_conflicts_summary.json
output/uda/office-home/AC/plmatch/diagnostics/AC_conflicts_summary.md
```

The first research check is:

```text
conflict_rate
conflict_source_correct_clip_wrong
conflict_source_wrong_clip_correct
useful_conflict_rate_among_conflicts
```

If useful conflicts are non-trivial, the paper problem is real enough to move
from diagnosis to a conflict-aware reliability module.
