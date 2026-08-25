# Step 3: Triangulated Conflict Analysis

This step tests a stronger conflict-learning idea than confidence or prototype
consistency alone.

Core question:

> When source and CLIP disagree, can target feature structure support
> candidate-set learning or more reliable conflict selection?

## Important

This step requires the latest `export_conflict_diagnostics.py`, because it now
exports both:

```text
*_conflicts.csv
*_conflicts.npz
```

The `.npz` file contains target features and probability vectors.

## Cloud Commands

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
```

First rerun diagnostics so the new `.npz` files are generated:

```bash
bash tools/run_office_home_conflict_diagnostics.sh
```

Then run triangulated analysis:

```bash
python tools/analyze_triangulated_conflicts.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*_conflicts.csv' \
  --out-dir output/uda/office-home/triangulated_conflicts
```

Outputs:

```text
output/uda/office-home/triangulated_conflicts/triangulated_conflict_analysis.csv
output/uda/office-home/triangulated_conflicts/triangulated_conflict_analysis.json
output/uda/office-home/triangulated_conflicts/triangulated_conflict_analysis.md
```

## What It Evaluates

The script compares:

| Metric | Meaning |
|---|---|
| `candidate_set_recall` | Whether true label is in `{source_pred, clip_pred}` |
| `always_clip_acc` | Trust CLIP on all conflicts |
| `higher_conf_acc` | Trust higher-confidence side |
| `prototype_only_acc` | Trust side closer to agreement-sample class prototype |
| `neighborhood_only_acc` | Trust side supported by nearby agreement samples |
| `triangulated_acc` | Combine confidence + prototype + neighborhood evidence |
| `triangulated_reject_acc` | Triangulated evidence with low-gap conflicts rejected |
| `triangulated_reject_coverage` | Non-rejected conflict coverage |

Ground-truth labels are used only for evaluation.

## Threshold Runs

Try reject thresholds:

```bash
python tools/analyze_triangulated_conflicts.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*_conflicts.csv' \
  --out-dir output/uda/office-home/triangulated_conflicts_gap005 \
  --gap 0.05

python tools/analyze_triangulated_conflicts.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*_conflicts.csv' \
  --out-dir output/uda/office-home/triangulated_conflicts_gap010 \
  --gap 0.10
```

Try changing evidence weights:

```bash
python tools/analyze_triangulated_conflicts.py \
  --glob 'output/uda/office-home/*/plmatch/diagnostics/*_conflicts.csv' \
  --out-dir output/uda/office-home/triangulated_conflicts_structure_heavy \
  --w-conf 0.5 --w-proto 1.0 --w-neigh 1.5
```

## How To Interpret

If `candidate_set_recall` is high but hard-selection accuracy is not, the next
method should use candidate-set supervision:

```text
L_candidate = -log(p(source_pred) + p(clip_pred))
```

If `triangulated_acc` or `triangulated_reject_acc` beats `always_clip_acc`, then
triangulated evidence can become the hard pseudo-label selector.

If it does not beat `always_clip_acc`, the method should avoid hard selection
and focus on candidate-set learning plus harmful-conflict rejection.
