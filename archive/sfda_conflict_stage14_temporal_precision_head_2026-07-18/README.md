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
target_head is initialized from source_classifier weights
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

## Full 12-Task Result

Cloud full Office-Home training has been observed for
`temporal_precision_head_all`.

| Task | target_head_all | DUET paper | both_prior | ACCD frozen+persistent | Delta vs DUET | Delta vs both_prior |
|---|---:|---:|---:|---:|---:|---:|
| AC | 73.54 | 73.60 | 72.78 | 73.15 | -0.06 | +0.76 |
| AP | 90.97 | 90.40 | 90.81 | 90.58 | +0.57 | +0.16 |
| AR | 91.12 | 91.00 | 91.00 | 90.68 | +0.12 | +0.12 |
| CA | 83.64 | 83.60 | 83.23 | 83.23 | +0.04 | +0.41 |
| CP | 91.15 | 90.70 | 90.92 | 90.88 | +0.45 | +0.23 |
| CR | 90.73 | 90.90 | 90.64 | 90.48 | -0.17 | +0.09 |
| PA | 83.35 | 82.70 | 82.12 | 82.24 | +0.65 | +1.23 |
| PC | 73.38 | 73.70 | 72.81 | 72.71 | -0.32 | +0.57 |
| PR | 91.14 | 91.20 | 90.82 | 90.89 | -0.06 | +0.32 |
| RA | 83.56 | 83.60 | 82.57 | 82.98 | -0.04 | +0.99 |
| RC | 73.86 | 74.00 | 72.97 | 72.99 | -0.14 | +0.89 |
| RP | 91.10 | 91.20 | 90.97 | 90.88 | -0.10 | +0.13 |
| Avg | 84.7950 | 84.7167 | 84.3033 | 84.3075 | +0.0783 | +0.4917 |

The full-task gate passes:

```text
target_head_all mean = 84.7950
DUET paper mean = 84.7167
delta vs DUET = +0.0783
delta vs both_prior = +0.4917
delta vs ACCD frozen+persistent = +0.4875
```

Task-level summary:

```text
beats DUET paper on 5/12 tasks
beats both_prior on 12/12 tasks
beats ACCD frozen+persistent on 12/12 tasks
dynamics probe passes 12/12 tasks
```

Conclusion:

```text
temporal precision target-head is the first method in this project to exceed
the public DUET Office-Home mean in full 12-task validation
```

This should now be treated as the main method candidate. Further work should
prioritize robustness checks and paper-ready ablations over additional
single-rule graph or pseudo-label admission variants.

## Stability Validation Plan

Because the full 12-task gain over DUET is small (+0.0783), this result should
not be claimed as robust until adaptation-seed stability is checked.

First validation:

```text
same source checkpoint
same target-head method
vary target adaptation seed over 2020, 2021, 2022
run all 12 Office-Home tasks
```

Cloud command:

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_seed_sweep.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_seed_sweep_accuracy.csv
output/uda/office-home/temporal_precision_head_seed_sweep_summary.json
```

Stability gate:

```text
all adaptation seeds should beat DUET mean 84.7167
the minimum seed mean matters more than the average seed mean
large per-task variance should be reported even if the mean passes
```

If this gate fails, the honest paper framing is that target-head adaptation is
a promising single-run result, not yet a stable state-of-the-art improvement.
If it passes, the next validation should use independent source checkpoints,
which requires matching source weights for each source seed.

## Adaptation-Seed Stability Result

Cloud adaptation-seed sweep has been observed for seeds 2020, 2021, and 2022
using the same source checkpoint.

| Seed | Mean | Delta vs DUET | Beats DUET mean | Task wins vs DUET |
|---|---:|---:|---:|---:|
| 2020 | 84.8100 | +0.0933 | true | 5/12 |
| 2021 | 84.7275 | +0.0108 | true | 5/12 |
| 2022 | 84.5267 | -0.1900 | false | 3/12 |

Aggregate:

```text
decision = fail_stability_gate
mean over seed means = 84.6881
std over seed means = 0.1457
min seed mean = 84.5267
max seed mean = 84.8100
DUET mean = 84.7167
delta aggregate vs DUET = -0.0286
```

Highest task-level seed variance:

| Task | mean | std | min | max | Delta mean vs DUET |
|---|---:|---:|---:|---:|---:|
| PR | 90.7633 | 0.3868 | 90.54 | 91.21 | -0.4367 |
| PA | 83.0767 | 0.3700 | 82.65 | 83.31 | +0.3767 |
| AR | 90.8433 | 0.3669 | 90.50 | 91.23 | -0.1567 |
| CA | 83.7000 | 0.3516 | 83.44 | 84.10 | +0.1000 |
| AC | 73.4167 | 0.3109 | 73.06 | 73.63 | -0.1833 |

Conclusion:

```text
target-head adaptation is a strong and useful method candidate, but the
DUET-level gain is not stable across adaptation seeds
do not claim stable state-of-the-art over DUET from the current evidence
the method remains substantially stronger than the same-environment both_prior
baseline, but the public-paper comparison needs a larger margin or a robustness
mechanism
```

Next direction should not be another simple loss or fixed graph-rule variant.
The stability failure suggests sensitivity in target-head adaptation itself:
future work should diagnose and regularize the target head's update path rather
than changing the conflict sample selector.

## Peak-Selection Re-evaluation

The original Stage14 table used the final logged accuracy. The evaluation
protocol was later changed by explicit project decision to use the highest
logged target accuracy for each task. Stage14 must therefore be re-extracted
under that same protocol before deciding whether its target head needs another
architectural change.

No retraining is required if the original 36 log files remain available:

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/reextract_office_home_temporal_precision_head_peak.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_stage14_peak_accuracy.csv
output/uda/office-home/temporal_precision_head_stage14_peak_summary.json
output/uda/office-home/temporal_precision_head_stage14_final_peak_gap.json
```

The Stage14 log glob is restricted to exact seed directories and excludes EMA
and residual-head runs.

Decision:

```text
all three peak-selected seed means > 84.7167 and std <= 0.10:
    retain Stage14 as the main method and optimize only paper-ready robustness

peak-selected mean improves but stability still fails:
    optimize Stage14 with cycle-snapshot weight averaging around its peak region
    validate using the same peak protocol

oracle peak still fails to exceed DUET:
    Stage14 lacks sufficient decision-boundary capacity
    proceed to dataset-level stable class-pair conflict-flow adaptation
```

Protocol caveat: peak selection uses target-domain labels and must be described
as a best-checkpoint/oracle evaluation protocol rather than label-free model
selection.

Peak-selection result:

| Seed | Peak mean | Delta vs DUET | Task wins vs DUET |
|---|---:|---:|---:|
| 2020 | 84.9000 | +0.1833 | 9/12 |
| 2021 | 84.7692 | +0.0525 | 5/12 |
| 2022 | 84.6783 | -0.0383 | 5/12 |

```text
mean over seed means = 84.7825
delta vs DUET = +0.0658
seed-mean std = 0.1114
decision = fail_stability_gate
```

The aggregate peak mean exceeds DUET, but seed 2022 and the standard-deviation
gate still fail narrowly. In 23/36 runs the final checkpoint is below an
earlier peak; most peaks occur at 50% or 75% of cycle 4.

The peak re-evaluation selects Stage14 as the base for Stage17. Stage17 keeps
the method unchanged and adds fixed cycle-4 full-model trajectory ensembling
to address the remaining seed-2022 gap and late-checkpoint oscillation.
