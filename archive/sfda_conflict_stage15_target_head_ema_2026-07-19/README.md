# Stage 15: Temporal Precision EMA Target Head

Date: 2026-07-19

## Starting Evidence

Stage14 exceeded the public DUET mean in one full run, but failed the
adaptation-seed stability gate:

| Metric | Stage14 target head | Public DUET |
|---|---:|---:|
| seed 2020 mean | 84.8100 | 84.7167 |
| seed 2021 mean | 84.7275 | 84.7167 |
| seed 2022 mean | 84.5267 | 84.7167 |
| mean over seeds | 84.6881 | 84.7167 |
| seed-mean std | 0.1457 | - |

This identifies target decision-boundary path sensitivity as the current
bottleneck. It does not support another pseudo-label selector, fixed graph
rule, prototype rule, prompt change, or simple loss coefficient variant.

## Method

Maintain two target classifier heads:

```text
online target head: receives gradients from target adaptation
EMA target head: no gradients; updated after every optimizer step
```

The update is:

```text
theta_ema <- 0.99 * theta_ema + 0.01 * theta_online
```

The online head is used inside the training loss. The EMA head is used for
cycle-level pseudo-label estimation and reported inference accuracy. The
source classifier remains frozen and is still mixed into the logits with the
unchanged weight of 0.7.

This changes the temporal decision-boundary estimator, not the loss or the
conflict-sample hard-selection rule.

## Config

```text
MODEL.METHOD = temporal_precision_head_ema
DCCL.TARGET_HEAD_ADAPT = True
DCCL.TARGET_HEAD_MIX = 0.3
DCCL.TARGET_HEAD_EMA = True
DCCL.TARGET_HEAD_EMA_MOMENTUM = 0.99
```

No other Stage14 method coefficient is changed.

## Step 1: Worst-Seed Gate

Run the previously failing adaptation seed over all 12 tasks:

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_ema_seed2022.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_ema_seed2022_accuracy.csv
output/uda/office-home/temporal_precision_head_ema_seed2022_summary.json
```

Gate:

```text
seed 2022 mean must exceed public DUET mean 84.7167
no task may show a severe collapse relative to Stage14 seed 2022
```

If Step 1 fails, archive the result and stop EMA momentum variants. The next
method will be a source-anchored residual classifier: a zero-initialized
residual branch added to the frozen source logits with an explicit bounded
global gate. That tests controlled target displacement instead of temporal
averaging.

## Step 2: Three-Seed Stability Gate

Run only if Step 1 passes:

```bash
cd /openbayes/home/vlm-sfda-conflict/duet-sfda-main
bash tools/run_office_home_temporal_precision_head_ema_seed_sweep.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_ema_seed_sweep_accuracy.csv
output/uda/office-home/temporal_precision_head_ema_seed_sweep_summary.json
```

Gate:

```text
all three adaptation-seed means must exceed 84.7167
sample std across seed means must be <= 0.10
```

If Step 2 fails, archive it and move to the source-anchored residual
classifier. Do not tune EMA momentum against target accuracy.

If Step 2 passes, the next validation is independent source-checkpoint seeds.
If matching source checkpoints are unavailable, the next deliverable is the
paper ablation separating temporal precision memory, online target head, and
EMA inference teacher.

## Status

```text
implementation complete
local tests passed
seed-2022 cloud result observed: fail
```

## Seed-2022 Result

| Metric | EMA head | Stage14 online head | Public DUET |
|---|---:|---:|---:|
| seed 2022 mean | 84.5033 | 84.5267 | 84.7167 |
| delta vs DUET | -0.2133 | -0.1900 | - |
| task wins vs DUET | 3/12 | 3/12 | - |

EMA reduces the seed-2022 mean by 0.0234 relative to the online target head.
The Step 1 gate fails, so the three-seed EMA sweep must not be run and EMA
momentum must not be tuned.

The original extractor reports the final logged checkpoint. A newly identified
question is whether earlier checkpoints have meaningful unused headroom. The
maximum target accuracy is not a valid SFDA checkpoint selector because it
uses target labels, but it is useful as an oracle diagnostic.

Run without retraining:

```bash
cd /openbayes/home/vlm-sfda-conflict/duet-sfda-main
git pull
bash tools/run_office_home_temporal_precision_head_ema_peak_diagnostic.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_ema_seed2022_final_peak.csv
output/uda/office-home/temporal_precision_head_ema_seed2022_final_peak_summary.json
```

Decision after the diagnostic:

```text
oracle peak mean <= 84.7167:
    proceed to the source-anchored zero-initialized residual classifier

oracle peak mean > 84.7167 while final mean fails:
    develop a label-free cycle selector using prediction stability and entropy
    do not report oracle peak as the main paper result
```

## Oracle Peak Diagnostic Result

| Metric | Value | Delta vs DUET |
|---|---:|---:|
| final mean | 84.5033 | -0.2133 |
| oracle peak mean | 84.5517 | -0.1650 |
| oracle peak minus final | +0.0483 | - |

Only 7/12 tasks have a peak above the final checkpoint, and the largest
per-task gap is 0.13 on C->A. All oracle peaks occur within cycle 4. The
available headroom is too small to explain the DUET gap.

Conclusion:

```text
decision = no_oracle_headroom
do not develop an unlabeled cycle selector for this EMA path
move to the source-anchored bounded residual classifier in Stage16
```
