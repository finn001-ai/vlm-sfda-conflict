# Stage 16: Source-Anchored Bounded Residual Head

Date: 2026-07-19

## Evidence From Stage 15

The EMA target head fails the worst-seed gate:

| Metric | Value |
|---|---:|
| seed-2022 final mean | 84.5033 |
| seed-2022 oracle peak mean | 84.5517 |
| public DUET mean | 84.7167 |
| oracle peak minus final | +0.0483 |

The oracle peak remains 0.1650 below DUET. Therefore checkpoint selection is
not the bottleneck, and the EMA three-seed sweep must not be run. The result
does not justify an unlabeled cycle selector.

## Hypothesis

Stage14 copies and adapts a full classifier head. Its target boundary can
drift in every class direction, which creates seed sensitivity. Stage15 only
averages that drifting path and does not constrain where the path can go.

Stage16 changes the classifier parameterization itself:

```text
frozen source logits + bounded target residual
```

The target residual is zero at initialization, so the initial predictor is
exactly the source-anchored predictor. It can only add a bounded displacement
relative to each sample's source-logit scale.

## Method

```text
r(x) = zero-initialized target residual head
g    = 0.3 * sigmoid(learned global gate logit)
s(x) = per-sample standard deviation of frozen source logits

output(x) = source_logits(x) + g * s(x) * tanh(r(x))
```

Properties:

```text
initial residual is exactly zero
source classifier remains frozen
global gate is bounded by 0.3
each class-logit displacement is bounded by 0.3 * source-logit scale
gate is excluded from weight decay
```

This is a structural decision-boundary constraint, not a new loss, prompt
variant, prototype rule, or per-sample graph selector.

## Config

```text
MODEL.METHOD = temporal_precision_head_residual
DCCL.TARGET_HEAD_ADAPT = True
DCCL.TARGET_HEAD_VARIANT = residual
DCCL.TARGET_HEAD_START_CYCLE = 1
DCCL.TARGET_RESIDUAL_MAX_GATE = 0.3
DCCL.TARGET_RESIDUAL_GATE_INIT = -2.0
DCCL.TARGET_HEAD_EMA = False
```

All temporal precision, both-prior calibration, and graph-temporal residual
settings remain unchanged from Stage14. The graph residual is not used as a
hard selector.

## Step 1: Worst-Seed Probe

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_residual_seed2022.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_residual_seed2022_accuracy.csv
output/uda/office-home/temporal_precision_head_residual_seed2022_summary.json
output/uda/office-home/temporal_precision_head_residual_seed2022_peak_summary.json
```

The Stage16 scripts explicitly use `--selection peak`. Therefore the CSV
`accuracy` column and the DUET gate use the highest logged target accuracy.
The CSV also retains `final_accuracy`, both checkpoint locations, and
`residual_gate_final` for auditability.

Gate:

```text
seed-2022 peak-selected mean must exceed public DUET mean 84.7167
the same peak-selection protocol must be used for every task and seed
final accuracy must remain archived as a secondary metric
no severe task collapse relative to Stage14 seed 2022
```

Protocol caveat:

```text
peak selection uses target-domain ground-truth accuracy
report it as a best-checkpoint/oracle protocol, not label-free model selection
do not mix peak-selected Stage16 values with final-checkpoint baselines
```

If Step 1 fails, archive the result and stop residual max-gate or gate-init
sweeps. The next method is dataset-level stable class-pair conflict flow:
estimate recurrent source-to-CLIP disagreement pairs across cycles and learn a
low-rank classifier displacement only in that persistent confusion subspace.
This uses aggregate conflict structure rather than per-sample hard selection.

## Step 2: Three-Seed Stability

Run only if Step 1 passes:

```bash
cd /openbayes/home/vlm-sfda-conflict/duet-sfda-main
bash tools/run_office_home_temporal_precision_head_residual_seed_sweep.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_residual_seed_sweep_accuracy.csv
output/uda/office-home/temporal_precision_head_residual_seed_sweep_summary.json
```

Gate:

```text
all peak-selected seed means exceed 84.7167
sample std across seed means <= 0.10
```

If Step 2 fails, move to the stable class-pair conflict-flow method. If it
passes, validate independent source checkpoints before making a robust
DUET-level claim.

## Status

```text
implementation complete
local validation passed (21 tests)
cloud result pending
```

## Execution Status Update

The cloud run was manually stopped because intermediate accuracy was
insufficient. No complete 12-task result is available, so Stage16 cannot pass
its gate and is paused. The project returns to Stage14 for a peak-selection
re-evaluation before any additional residual-head work.
