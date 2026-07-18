# Stage 12: Temporal Precision With Target Prototypes

Date: 2026-07-18

## Rationale

Stage 11 passed its gate:

| Task | temporal_precision_residual | both_prior | DUET paper |
|---|---:|---:|---:|
| A->C | 73.38 | 72.78 | 73.60 |
| P->C | 73.06 | 72.81 | 73.70 |
| R->C | 73.36 | 72.97 | 74.00 |

It fixed much of the pseudo-label precision problem, but the method still
does not reach DUET. The next bottleneck is likely target-domain decision
boundary mismatch: the classifier remains the frozen source classifier while
the target feature distribution has shifted.

## Method

Build target-domain prototypes from high-precision temporal pseudo labels:

```text
features = target bottleneck features
labels = stable temporal pseudo labels
mask = stage11 temporal precision mask
```

During the next label-estimation/training/evaluation pass, source classifier
logits are lightly adjusted by normalized prototype logits:

```text
logits = source_logits + PROTO_MIX * standardized_proto_logits
```

This is a decision-boundary adapter, not another loss. It keeps the stage11
training losses and pseudo-label memory unchanged.

## Default Config

```text
MODEL.METHOD = temporal_precision_proto
DCCL.PL_MEMORY = stable
DCCL.PROTO_ADAPT = True
DCCL.PROTO_MIX = 0.15
DCCL.PROTO_TEMPERATURE = 0.2
DCCL.PROTO_MIN_PER_CLASS = 3
```

## Implementation

```text
duet-sfda-main/cfgs/office-home/temporal_precision_proto.yaml
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/tools/run_office_home_temporal_precision_proto_clipart.sh
```

## Cloud Command

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_proto_clipart.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_proto_clipart_accuracy.csv
output/uda/office-home/temporal_precision_proto_dynamics_probe.json
```

## Gate

Proceed only if:

```text
mean target-Clipart accuracy beats temporal_precision_residual mean 73.27
at least one task reaches or exceeds its DUET paper value
no task drops below both_prior
```

If this fails, archive the result and move away from target prototype
adaptation.

## Result

Cloud target-Clipart training has been observed.

| Task | temporal_precision_proto | temporal_precision_residual | both_prior | DUET paper | Delta vs stage11 | Delta vs DUET |
|---|---:|---:|---:|---:|---:|---:|
| A->C | 73.06 | 73.38 | 72.78 | 73.60 | -0.32 | -0.54 |
| P->C | 72.90 | 73.06 | 72.81 | 73.70 | -0.16 | -0.80 |
| R->C | 73.47 | 73.36 | 72.97 | 74.00 | +0.11 | -0.53 |

Mean target-Clipart accuracy is 73.14, below the stage11
`temporal_precision_residual` mean of 73.27. No task reaches the DUET paper
number. The prototype adapter therefore fails the stage gate.

Conclusion:

```text
do not tune PROTO_MIX/temperature/min_per_class variants
move away from target prototype adaptation
```

The useful signal remains stage11's temporal precision memory. The next
direction should improve the pseudo-label pool's target-class coverage and
class balance rather than perturb the classifier logits.
