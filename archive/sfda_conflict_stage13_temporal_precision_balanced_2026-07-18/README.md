# Stage 13: Target-Prior Balanced Temporal Precision

Date: 2026-07-18

## Rationale

Stage 11 remains the strongest target-Clipart training result so far. Stage 12
shows that target prototype logit adaptation is not robust:

| Task | temporal_precision_proto | temporal_precision_residual | DUET paper |
|---|---:|---:|---:|
| A->C | 73.06 | 73.38 | 73.60 |
| P->C | 72.90 | 73.06 | 73.70 |
| R->C | 73.47 | 73.36 | 74.00 |

The next change stays with the successful stage11 mechanism but makes its
pseudo-label pool more target-aware. Temporal precision memory improves label
quality, but it does not explicitly control class coverage. A biased stable
pseudo-label pool can still train a target decision boundary that is too
source-shaped.

## Method

After stage11 stable temporal memory selects candidate supervised labels:

```text
label must be source/CLIP agreement
same label must persist for PL_STABLE_CYCLES
memory is reversible
```

Stage13 applies a target-prior class balance before supervised CE:

```text
target prior = mean calibrated both_prior teacher over all target samples
budget = PL_BALANCE_COVERAGE * number_of_target_samples
within each class, keep stable labels up to target-prior quota
rank only within stable labels
```

This is not a loss change and not a graph rule. It changes the composition of
the supervised pseudo-label pool using a target-level prior rather than
per-sample confidence adjudication.

## Default Config

```text
MODEL.METHOD = temporal_precision_balanced
DCCL.PL_MEMORY = stable
DCCL.PL_CLASS_BALANCE = True
DCCL.PL_BALANCE_COVERAGE = 0.75
DCCL.PL_BALANCE_MIN_PER_CLASS = 1
```

Stage10 graph-temporal residual regularization remains enabled with the same
small coefficient as stage11.

## Cloud Command

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_balanced_clipart.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_balanced_clipart_accuracy.csv
output/uda/office-home/temporal_precision_balanced_dynamics_probe.json
```

## Gate

Proceed only if:

```text
mean target-Clipart accuracy beats temporal_precision_residual mean 73.27
at least one task reaches or exceeds its DUET paper value
no task drops below both_prior
```

If this fails, archive it and do not continue pseudo-label admission variants
without a new diagnostic.
