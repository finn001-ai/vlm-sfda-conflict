# Stage 22: Three-View Class-Conditional Noise EM

Date: 2026-07-20

## Starting Point

Stage14 remains the strongest method:

| Protocol | Seed 2020 | Seed 2021 | Seed 2022 | Mean | Std |
|---|---:|---:|---:|---:|---:|
| Per-task oracle peak | 84.9000 | 84.7692 | 84.6783 | 84.7825 | 0.1114 |

The best seed exceeds DUET by `0.1833`, but seed 2022 is `0.0383` below DUET
and the standard deviation exceeds the fixed `0.10` gate. Stage15-17 show that
EMA, bounded residual parameterization, and fixed trajectory ensembling do not
stabilize the head. Stage18-21 show that class-pair and geometric feature
transport do not improve the matched Stage14 control.

Stage14 is therefore retained unchanged as the base. Stage22 targets the
remaining source of variance: the adaptive head is trained from temporal
pseudo labels whose cycle-4 precision is about 84%, so seed-dependent
memorization of residual label noise remains plausible.

## Oracle Peak Meaning

`peak` is the highest logged target accuracy for each independently trained
Office-Home task. Selecting it reads target labels, so it is an oracle or
best-checkpoint protocol. It can be reported as an upper-bound result if the
paper labels it explicitly and applies the same protocol to compared methods.
It is not a label-free deployment rule.

The archived `84.9000` can be reproduced with:

```bash
cd /hyperai/home/vlm-sfda-conflict/duet-sfda-main
git pull
bash tools/run_office_home_temporal_precision_head_seed2020_reproduce.sh
```

This writes an isolated 12-task peak CSV and a JSON comparison against every
archived seed-2020 task value. The historical code uses cuDNN benchmark mode,
so the same seed does not guarantee bitwise equality across GPU/software
stacks. The source checkpoints under `source/uda/office-home/{A,C,P,R}` must
also be identical.

## Method

At each cycle after Stage14 temporal warmup:

1. Use reversible stable source/CLIP agreements as noisy class anchors.
2. Treat calibrated source probabilities, calibrated CLIP probabilities, and
   the raw dual-space graph posterior as three noisy views.
3. Estimate one class-conditional transition matrix per view with an identity
   Dirichlet prior. A class must have at least three stable anchors before its
   transition row is estimated; otherwise that row remains the identity.
4. Alternate transition estimation and latent posterior inference for five EM
   steps while clamping stable agreement anchors.
5. Convert posterior entropy into a continuous conflict weight. No conflict is
   hard-selected as source or CLIP.
6. Add a `0.05` weighted KL from the Stage14 blend head to the EM posterior.

The additional branch uses detached features and the frozen source classifier:

```text
z_em = stop_gradient(z)
logits_em = 0.7 * source_head(z_em) + 0.3 * target_head(z_em)
L = L_Stage14 + 0.05 * weighted_KL(logits_em, q_em)
```

Consequently the new loss updates only the target head. Stage14 backbone,
CLIP update, temporal precision memory, main KL, consistency, hard CE, and GTR
remain unchanged. This isolates whether class-conditional soft noise modeling
reduces target-head seed sensitivity.

Fixed configuration:

```text
THREE_VIEW_EM = True
THREE_VIEW_EM_START_CYCLE = 1
THREE_VIEW_EM_STEPS = 5
THREE_VIEW_EM_DIRICHLET = 5.0
THREE_VIEW_EM_MIN_CLASS_ANCHORS = 3
THREE_VIEW_EM_PAR = 0.05
gradient scope = target_head_only
```

## Step 0: AC/PA/RA Preflight

Run:

```bash
bash tools/run_office_home_temporal_precision_head_three_view_em_preflight.sh
```

Return:

```text
output/uda/office-home/temporal_precision_head_three_view_em_preflight_accuracy.csv
output/uda/office-home/temporal_precision_head_three_view_em_preflight_flow.json
output/uda/office-home/temporal_precision_head_three_view_em_preflight_summary.json
```

The mechanism must run for cycles 2-4, have at least 512 anchors, 40 supported
classes, 100 continuously weighted conflicts, and a nonzero head-only EM loss
on every task. The performance gate requires:

```text
AC/PA/RA peak mean > matched Stage17 online Stage14 mean 80.0733
AC/PA/RA peak mean > DUET subset mean 79.9667
no task loses more than 0.50 from matched Stage14
```

## Step 1: Complete Seed-2022 Gate

Run only if Step 0 reports `pass_three_view_em_preflight`:

```bash
bash tools/run_office_home_temporal_precision_head_three_view_em_seed2022.sh
```

The complete peak mean must exceed the matched Stage17 online Stage14 control
`84.7225`, all 12 EM routes must be active under the fixed configuration, and
no task may fall more than `1.50` below DUET.

## Step 2: Three-Seed Stability

Run only if Step 1 passes:

```bash
bash tools/run_office_home_temporal_precision_head_three_view_em_seed_sweep.sh
```

This reuses the completed seed-2022 run and trains seeds 2020/2021. The
Stage22-specific gate requires exactly 36 valid fixed-configuration records,
all seed means above DUET, sample standard deviation at most `0.10`, and mean
over seed means above Stage14's `84.7825`.

## Status

```text
implementation complete
local validation passed (91 tests)
cloud preflight complete: fail
do not run the 12-task seed-2022 gate
```

No EM step, prior, anchor threshold, or loss-weight sweep is approved before
the fixed preflight. If the mechanism is active but accuracy fails, archive it
and retain Stage14 as the final method rather than starting another loss grid.

## Cloud Preflight Result

| Task | Stage22 peak | Matched Stage14 | Delta | DUET | Delta vs DUET |
|---|---:|---:|---:|---:|---:|
| AC | 73.61 | 73.59 | +0.02 | 73.60 | +0.01 |
| PA | 82.98 | 83.11 | -0.13 | 82.70 | +0.28 |
| RA | 83.44 | 83.52 | -0.08 | 83.60 | -0.16 |
| **Mean** | **80.0100** | **80.0733** | **-0.0633** | **79.9667** | **+0.0433** |

The implementation passed all mechanism checks on all three tasks, but failed
the matched-performance gate. In cycle 2 the EM posterior changed `759`,
`353`, and `315` top-1 predictions for AC, PA, and RA, respectively, while
mean conflict weights were only `0.1091`, `0.0948`, and `0.1114`. Every
source/CLIP conflict received nonzero weight. The branch therefore applied a
broad correction to many uncertain conflicts rather than identifying a small
reliable residual.

Conclusion:

```text
decision = fail_three_view_em_preflight
mechanism = valid but not beneficial
do not run all 12 tasks
do not tune THREE_VIEW_EM_PAR, steps, or Dirichlet strength
retain Stage14
```

The next direction is label-free checkpoint risk estimation on the unchanged
Stage14 trajectory. Stage14 already has an oracle peak above DUET; the missing
piece is a defensible target-label-free rule that ranks its checkpoints. This
is a model-selection problem, not another conflict-loss problem.
