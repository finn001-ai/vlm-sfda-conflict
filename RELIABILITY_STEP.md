# Step 2: Conflict Reliability Analysis

This step evaluates whether oracle-free signals can identify which side of a
conflict sample is more reliable.

The input is the per-sample CSV exported by:

```bash
bash duet-sfda-main/tools/run_office_home_conflict_diagnostics.sh
```

## Run on Cloud

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main

python tools/analyze_conflict_reliability.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*_conflicts.csv' \
  --out-dir output/uda/office-home/conflict_reliability
```

Outputs:

```text
output/uda/office-home/conflict_reliability/conflict_reliability_analysis.csv
output/uda/office-home/conflict_reliability/conflict_reliability_analysis.json
output/uda/office-home/conflict_reliability/conflict_reliability_analysis.md
```

## What It Tests

The script evaluates conflict samples only:

```text
source_pred != clip_pred
```

It compares these selectors:

| Selector | Meaning |
|---|---|
| `always_source_acc` | Always trust source model on conflict samples |
| `always_clip_acc` | Always trust CLIP on conflict samples |
| `higher_conf_acc` | Trust the side with higher confidence |
| `classwise_acc` | Trust confidence weighted by class-wise reliability estimated from agreement samples |
| `classwise_reject_acc` | Same as class-wise reliability, but rejects low-gap conflicts |
| `classwise_reject_coverage` | Percentage of conflict samples not rejected |

Ground-truth labels are used only for evaluation, not for constructing the
selectors.

## Optional Reject Thresholds

Try thresholds to see whether accuracy improves while retaining useful
coverage:

```bash
python tools/analyze_conflict_reliability.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*_conflicts.csv' \
  --out-dir output/uda/office-home/conflict_reliability_gap005 \
  --score-gap 0.05

python tools/analyze_conflict_reliability.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*_conflicts.csv' \
  --out-dir output/uda/office-home/conflict_reliability_confgap005 \
  --conf-gap 0.05
```

## Decision Rule

If:

```text
classwise_acc > higher_conf_acc
```

or:

```text
classwise_reject_acc improves with acceptable coverage
```

then class-wise reliability is a valid first method component.

If confidence and class-wise reliability are weak, move next to prototype
consistency.
