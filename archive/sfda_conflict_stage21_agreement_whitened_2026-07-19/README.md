# Stage 21: Global Agreement-Whitened Transport

Date: 2026-07-19

## Starting Evidence

Stage20 conditional covariance transport had valid geometry on AC/PA/RA, with
`50-60` active classes and `62.06%-91.46%` conflict coverage. Nevertheless,
every task regressed from matched Stage14 and every mean relative shift reached
the fixed `5%` cap. The failure therefore concerns transport direction, not
coverage.

Stage21 removes both Stage20 assumptions that remain untested:

1. transport is global rather than restricted to initial source/CLIP pairs;
2. strength is selected from held-out agreement evidence and may be exactly
   zero rather than fixed at `5%`.

This is a major method change and therefore receives a new stage. It does not
modify the prompt, graph, target head, pseudo-label memory, or Stage14 GTR.

## Method

### Frozen agreement split

At cycle 1, take all calibrated source/CLIP agreements and deterministically
split each represented class into `80%` fitting anchors and `20%` held-out
anchors. Target labels are never read. The fitting set and source classifier
are frozen.

### Shrinkage-whitened alignment

Normalize target features and source-classifier weight vectors. The reference
point paired with agreement feature `z_i` is the normalized source classifier
weight of its source/CLIP consensus class. Estimate shrinkage covariances for
both distributions, whiten them, and solve the orthogonal Procrustes problem
between paired whitened points. The final affine map is:

```text
T(z) = ((normalize(z) - mu_z) C_z^-1/2 R C_w^1/2 + mu_w)
```

where `R` is obtained from the SVD of the paired whitened cross-covariance.
This combines global second-order geometry with class correspondence without a
per-sample conflict selector.

### Label-free strength selection

The fixed candidate relative-shift bounds are:

```text
0, 0.005, 0.0125, 0.025, 0.0375, 0.05
```

For each nonzero candidate, evaluate class-balanced source-classifier CE on
the held-out agreement set. A candidate is valid only if it improves held-out
CE by at least `0.001` and does not reduce held-out consensus accuracy. Select
the smallest valid nonzero candidate; otherwise select zero and recover an
exact Stage14 identity path. This minimum-change rule prevents the classifier
reference from automatically driving selection to the largest bound. The
selected transform then applies globally from cycle 2. No target accuracy or
prompt information enters fitting or selection.

## Fixed Configuration

```text
COV_TRANSPORT_ADAPT = True
COV_TRANSPORT_MODE = global_whitened
COV_TRANSPORT_START_CYCLE = 1
COV_TRANSPORT_MAX_GATE = 0.05
COV_GLOBAL_MIN_ANCHORS = 512
COV_GLOBAL_SHRINKAGE = 0.1
COV_GLOBAL_HOLDOUT_RATIO = 0.2
COV_GLOBAL_MIN_IMPROVEMENT = 0.001
PAIR_FEATURE_ADAPT = False
TARGET_HEAD_VARIANT = blend
TARGET_HEAD_MIX = 0.3
GTR_PAR = 0.05
```

No parameter sweep is approved before the fixed preflight.

## Step 0: AC/PA/RA Preflight

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_agreement_whitened_preflight.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_agreement_whitened_preflight_accuracy.csv
output/uda/office-home/temporal_precision_head_agreement_whitened_preflight_flow.json
output/uda/office-home/temporal_precision_head_agreement_whitened_preflight_summary.json
```

Mechanism gate on every task:

```text
agreements >= 512
held-out agreements >= 100
active agreement classes >= 40
0 < selected strength <= 0.05
held-out balanced CE improvement >= 0.001
held-out consensus accuracy does not decrease
0 < mean global shift <= selected strength
```

Performance gate:

```text
AC/PA/RA peak mean > matched Stage14 mean 80.0733
AC/PA/RA peak mean > DUET subset mean 79.9667
no task loses more than 0.50 from matched Stage14
```

All accuracy summaries use target-label-selected `peak`, as requested. Labels
are used only by this post-training performance gate.

## Step 1: Complete Seed-2022 Gate

Run only after `pass_whitened_preflight`:

```bash
bash tools/run_office_home_temporal_precision_head_agreement_whitened_seed2022.sh
```

The complete peak mean must exceed the matched `84.7225`; all 12 transport
diagnostics must pass and no task may lose more than `1.50` from DUET.

## Step 2: Three-Seed Stability

Run only after the complete seed-2022 gate passes:

```bash
bash tools/run_office_home_temporal_precision_head_agreement_whitened_seed_sweep.sh
```

The stability gate remains: all seed means above DUET, sample standard
deviation at most `0.10`, and mean over seeds above `84.7825`.

## Failure Route

If the fixed preflight fails, archive whether the held-out selector chose zero
or selected a harmful nonzero map. Do not sweep shrinkage, holdout ratio, or
gate. Close geometric feature transport and implement three-view
class-conditional noise EM consensus: treat source, CLIP, and graph-temporal
posteriors as noisy annotators; estimate aggregate class transition matrices;
infer soft latent labels over all samples; and train with those posteriors
without per-sample hard source-vs-CLIP selection.

## Status

```text
implementation complete
local validation passed (80 tests)
cloud preflight pending
```
