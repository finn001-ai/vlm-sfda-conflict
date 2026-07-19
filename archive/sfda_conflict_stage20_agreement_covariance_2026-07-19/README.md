# Stage 20: Agreement-Anchor Conditional Covariance Transport

Date: 2026-07-19

## Re-evaluated Status

Stage20 is implemented but paused before cloud execution. The earlier decision
to close every learned pair-router variant was too broad after the project
scope was clarified: graph information and a loss are allowed when their role
is mechanistically justified; only prompt-only adjustment is excluded.

Stages 8-11 show that graph evidence is useful while direct teacher replacement
is not. Stage19 also never isolated its pair router from generic CE, ordinary
KL, and consistency gradients. Stage19-G now tests that missing attribution by
training the bounded pair router only through stable graph-temporal evidence.
Run this Stage20 preflight only if Stage19-G activates correctly but fails its
accuracy gate. The covariance implementation and fixed gate remain unchanged.

## Starting Evidence

Stage19 learned a feature router in aggregate class-pair directions. Its valid
seed-2022 run reached `84.6908`, or `-0.0258` below DUET. The rank-protected
Stage19-C preflight then produced:

| Task | Stage19-C | Stage19 | Matched online Stage14 |
|---|---:|---:|---:|
| CA | 83.60 | 83.60 | 83.68 |
| PA | 82.61 | 82.49 | 83.11 |
| RA | 82.98 | 83.07 | 83.52 |
| Mean | 83.0633 | 83.0533 | 83.4367 |

The adapter was an exact identity on all three preflight tasks, but the mean
recovered only `+0.01`. Low active rank was therefore correlated with weak
tasks but was not their main causal explanation. Rank thresholds, gate
scaling, and learned pair routing are closed.

## Method

Stage20 retains the complete Stage14 training path and replaces the learned
router with one frozen, parameter-free geometric operator.

### Frozen agreement geometry

At the first adaptation cycle, take only samples whose calibrated source and
CLIP top-1 predictions agree. For every class with at least eight agreements:

1. Compute the class mean in task-feature space.
2. Compute the class-centered covariance SVD.
3. Retain its first four orthonormal covariance directions.
4. Estimate the residual variance outside that affine subspace.

The means, covariance bases, residual variances, initial source/CLIP conflict
pairs, and initial calibrated mix probabilities are frozen. Later model
agreements cannot enter the anchor bank.

### Soft conditional transport

For an initial conflict with candidates `(a, b)`, both class geometries must
exist. Let `P_c(z)` be the projection of feature `z` onto class `c`'s affine
covariance subspace, and let `e_c(z)` be residual mean-square error normalized
by the class residual variance. The two soft weights are:

```text
q_c proportional to p_mix(c) * exp(-0.5 * e_c(z))
```

The candidate-conditioned target and residual are:

```text
z_target = q_a * P_a(z) + q_b * P_b(z)
delta_raw = (p_mix(a) + p_mix(b)) * (z_target - z)
```

Probability outside the two candidates therefore remains on the identity
path. The final residual is clipped to at most `0.05 * ||z||`. Agreements,
conflicts missing either class geometry, and all samples in cycle 1 are exact
identities.

The operator has no trainable parameters. Gradients pass through it to the
Stage14 backbone and bottleneck, but no new loss is introduced.

## Distinction From Closed Families

Stage20 is not:

- a confidence/prototype/neighbor hard selector;
- a graph diffusion or fixed graph-action rule;
- another DUET loss term or head variant;
- a prompt modification;
- a learned source-vs-CLIP router.

It evaluates whether agreement samples identify class-conditional target
geometry that can reshape conflicts without assigning them hard labels.

## Fixed Config

```text
DCCL.COV_TRANSPORT_ADAPT = True
DCCL.COV_TRANSPORT_START_CYCLE = 1
DCCL.COV_TRANSPORT_MIN_ANCHORS = 8
DCCL.COV_TRANSPORT_RANK = 4
DCCL.COV_TRANSPORT_MAX_GATE = 0.05
DCCL.PAIR_FEATURE_ADAPT = False
DCCL.TARGET_HEAD_VARIANT = blend
DCCL.TARGET_HEAD_MIX = 0.3
```

No parameter sweep is approved before the fixed preflight.

## Step 0: AC/PA/RA Preflight

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_covariance_preflight.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_covariance_preflight_accuracy.csv
output/uda/office-home/temporal_precision_head_covariance_preflight_flow.json
output/uda/office-home/temporal_precision_head_covariance_preflight_summary.json
```

Mechanism gate for every task:

```text
active agreement classes >= 20
eligible fixed-conflict coverage >= 25%
0 < mean relative shift <= 5%
```

Performance gate:

```text
AC/PA/RA peak mean > matched base mean 80.0733
AC/PA/RA peak mean > DUET subset mean 79.9667
no task loses more than 0.50 from its matched base
```

Target labels are used only to report the diagnostic peak gate; they are not
inputs to fitting or transport.

## Step 1: Complete Seed-2022 Gate

Run only after `pass_covariance_preflight`:

```bash
bash tools/run_office_home_temporal_precision_head_covariance_seed2022.sh
```

Bring back the four `temporal_precision_head_covariance_seed2022` CSV/JSON
files. The complete gate requires peak mean above `84.7225`, valid transport
geometry on all tasks, and worst task delta versus DUET above `-1.50`.

## Failure Route

If the fixed preflight fails, do not sweep covariance rank, anchor count, or
gate. The next mechanism is global agreement-whitened optimal transport:
estimate a single shrinkage-whitened map from the complete agreement
distribution to source-classifier geometry, apply it globally rather than by
two conflict candidates, and use a label-free held-out agreement objective to
choose the interpolation strength. That would test whether the two-candidate
conditioning, rather than covariance geometry itself, is the limiting
assumption.

## Status

```text
implementation complete
local validation passed (64 tests)
cloud execution deferred behind Stage19-G
```
