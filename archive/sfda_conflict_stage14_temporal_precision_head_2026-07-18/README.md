# Stage 14: Temporal Precision Target Head

Date: 2026-07-18

## Rationale

Stage 11 remains the strongest target-Clipart result. Stage 12 and stage 13
failed:

| Variant | Failure |
|---|---|
| target prototypes | prototype logits hurt A->C and P->C |
| target-prior balanced labels | class-balance admission hurts P->C |

The common issue is that indirect decision-boundary corrections are brittle.
Stage13 also shows that a passing temporal dynamics probe is not sufficient if
the final target training accuracy drops.
This stage adapts the target decision boundary explicitly while keeping the
source classifier frozen as an anchor.

## Method

Create a target classifier head:

```text
target_head = copy(source_classifier)
source_classifier remains frozen
```

After the temporal precision warmup cycle, logits are:

```text
logits = (1 - TARGET_HEAD_MIX) * source_logits
       + TARGET_HEAD_MIX * target_head_logits
```

The target head is trained together with the feature extractor from the same
stage11 temporal precision pseudo labels. This is a model-architecture change,
not a new loss term.

## Default Config

```text
MODEL.METHOD = temporal_precision_head
DCCL.PL_MEMORY = stable
DCCL.TARGET_HEAD_ADAPT = True
DCCL.TARGET_HEAD_MIX = 0.3
DCCL.TARGET_HEAD_START_CYCLE = 1
DCCL.TARGET_HEAD_LR_MULT = 1.0
```

## Cloud Command

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_clipart.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_clipart_accuracy.csv
output/uda/office-home/temporal_precision_head_dynamics_probe.json
```

## Gate

Proceed only if:

```text
mean target-Clipart accuracy beats temporal_precision_residual mean 73.27
at least one task reaches or exceeds its DUET paper value
no task drops below both_prior
```

If this fails, archive it and stop target-head adaptation variants unless a
new diagnostic shows the head is underfitting rather than drifting.

## Result

Cloud target-Clipart training has been observed.

| Task | temporal_precision_head | temporal_precision_residual | both_prior | DUET paper | Delta vs stage11 | Delta vs DUET |
|---|---:|---:|---:|---:|---:|---:|
| A->C | 73.65 | 73.38 | 72.78 | 73.60 | +0.27 | +0.05 |
| P->C | 73.22 | 73.06 | 72.81 | 73.70 | +0.16 | -0.48 |
| R->C | 73.95 | 73.36 | 72.97 | 74.00 | +0.59 | -0.05 |

Mean target-Clipart accuracy is 73.6067, above the stage11
`temporal_precision_residual` mean of 73.2667 by +0.34. No Clipart task drops
below `both_prior`. A->C exceeds the DUET paper value, and R->C is within 0.05
of DUET.

The temporal dynamics probe passes all three Clipart tasks:

| Task | cycle4 teacher | cycle4 mix | cycle4 valid labels | cycle4 valid label accuracy |
|---|---:|---:|---:|---:|
| A->C | 73.6999 | 72.8522 | 3198 | 84.9906 |
| P->C | 73.5395 | 72.4857 | 3199 | 84.8078 |
| R->C | 73.8832 | 73.0584 | 3240 | 84.7531 |

Conclusion:

```text
target-head adaptation is the first post-stage11 change with a clear Clipart gain
it should be expanded to all 12 Office-Home tasks before tuning variants
do not treat this as proof of full DUET-level performance yet
```

Next command should run the same config on all Office-Home tasks and extract a
full 12-task table. Only after that result is known should TARGET_HEAD_MIX or
start-cycle variants be considered.

## Full 12-Task Follow-Up

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_all.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_all_accuracy.csv
output/uda/office-home/temporal_precision_head_all_dynamics_probe.json
```

Full-task gate:

```text
mean over 12 tasks must beat DUET paper mean 84.7167
no severe collapse on non-Clipart targets
Clipart gains should remain positive relative to stage11
```
