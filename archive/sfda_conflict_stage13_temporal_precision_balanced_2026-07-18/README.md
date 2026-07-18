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

## Result

Cloud target-Clipart training has been observed.

| Task | temporal_precision_balanced | temporal_precision_residual | both_prior | DUET paper | Delta vs stage11 | Delta vs DUET |
|---|---:|---:|---:|---:|---:|---:|
| A->C | 73.22 | 73.38 | 72.78 | 73.60 | -0.16 | -0.38 |
| P->C | 72.60 | 73.06 | 72.81 | 73.70 | -0.46 | -1.10 |
| R->C | 73.42 | 73.36 | 72.97 | 74.00 | +0.06 | -0.58 |

Mean target-Clipart accuracy is 73.08, below the stage11
`temporal_precision_residual` mean of 73.27. P->C also drops below
`both_prior`. The stage therefore fails its gate.

The temporal dynamics probe still passes all three Clipart tasks, so this is a
training-accuracy failure rather than a diagnostic-signal failure.

Conclusion:

```text
do not tune PL_BALANCE_COVERAGE or class-balance variants
stop pseudo-label admission variants unless a new diagnostic motivates them
```

Temporal precision memory remains the strongest mechanism. The next direction
should address target-domain decision boundary adaptation without perturbing
the pseudo-label pool composition.
