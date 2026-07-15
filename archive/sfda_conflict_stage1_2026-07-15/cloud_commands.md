# Cloud Commands

These commands were used or prepared for cloud-side reproduction.

## Pull Latest Code

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
```

## Prepare Office-Home List Files

Expected dataset layout:

```text
duet-sfda-main/data/office-home/
  Art/
  Clipart/
  Product/
  RealWorld/
```

Generate DUET list files:

```bash
python tools/prepare_office_home_lists.py --root data/office-home
```

## Train Source Checkpoints

Train all four Office-Home source domains:

```bash
bash tools/train_office_home_sources.sh
```

Train selected source domains only:

```bash
bash tools/train_office_home_sources.sh 1 2 3
```

The script moves checkpoints into the expected structure:

```text
source/uda/office-home/A/source_F.pt
source/uda/office-home/A/source_B.pt
source/uda/office-home/A/source_C.pt
```

Use `C`, `P`, and `R` for Clipart, Product, and RealWorld source domains.

## Run Conflict Diagnostics

Run all 12 Office-Home pairs:

```bash
bash tools/run_office_home_conflict_diagnostics.sh
```

Run only one source domain, for example Art:

```bash
bash tools/run_office_home_conflict_diagnostics.sh 0
```

Run one pair manually:

```bash
python tools/export_conflict_diagnostics.py \
  --cfg cfgs/office-home/plmatch.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  SETTING.S 0 SETTING.T 1
```

## Output Files

For `Art -> Clipart`, outputs are:

```text
output/uda/office-home/AC/plmatch/diagnostics/AC_conflicts.csv
output/uda/office-home/AC/plmatch/diagnostics/AC_conflicts_summary.json
output/uda/office-home/AC/plmatch/diagnostics/AC_conflicts_summary.md
```

Equivalent paths are created for the other task names.
