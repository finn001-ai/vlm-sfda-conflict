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
