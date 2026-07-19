# Stage 19: Aggregate-Flow Class-Pair Feature Adapter

Date: 2026-07-19

## Starting Evidence

Stage18 produced a peak mean of 84.6075, but its pair-flow rank was zero on all
12 tasks. The run is invalid for testing class-pair adaptation: it intersected
an agreement-only pseudo-label mask with a conflict mask, so no evidence could
reach the adapter.

Stage14's archived temporal diagnostics provide a stronger starting point.
On initial target-Clipart conflicts, stable mix predictions cover 85-87% of
samples and outperform both final candidate models:

| Task | Stable mix | Stable source | Stable CLIP | Coverage |
|---|---:|---:|---:|---:|
| AC | 66.9681 | 65.6354 | 63.3508 | 86.0008 |
| PC | 67.4071 | 66.0402 | 65.0150 | 87.2531 |
| RC | 67.8677 | 67.2261 | 64.9556 | 85.1618 |

These label-based values are oracle diagnostics only. They justify using the
aggregate temporal mix distribution, but are not inputs to Stage19.

## Method

Stage19 preserves the complete Stage14 method, including its source classifier,
blend target head, temporal precision memory, optimization, and existing
losses. It adds one constrained representation module.

### Aggregate soft conflict flow

At cycle 1, freeze every sample's calibrated source/CLIP conflict pair. For a
fixed pair `(a, b)`, later cycles contribute the calibrated mix probability
mass on `a` and `b`:

```text
mass a -> b += p_mix(b)
mass b -> a += p_mix(a)
```

No sample is assigned a new hard training label by this mechanism. A class-pair
direction is eligible only when it has at least five units of soft mass in at
least two cycles. Opposing directions are netted, then the highest supported
non-redundant edges are selected as a forest, giving at most 16 linearly
independent class-confusion axes. The set freezes after activation.

### Feature-space intervention

For each selected direction `loser -> winner`, construct a fixed feature axis
from the frozen source classifier:

```text
d_pair = normalize(W_source[winner] - W_source[loser])
```

A zero-initialized linear router maps each feature to continuous coefficients
over these axes. The residual is norm-bounded:

```text
z_adapted = z + gate * ||z|| * bounded(router(z) @ D_pair)
gate <= 0.05
```

Both the frozen source classifier and Stage14 target head consume `z_adapted`.
At initialization, before flow activation, and whenever the router is zero,
the output is exactly Stage14. This is an architecture constraint, not a new
loss, prompt change, graph rule, or per-sample hard selector.

## Fixed Config

```text
DCCL.TARGET_HEAD_ADAPT = True
DCCL.TARGET_HEAD_VARIANT = blend
DCCL.TARGET_HEAD_MIX = 0.3
DCCL.PAIR_FEATURE_ADAPT = True
DCCL.PAIR_FEATURE_START_CYCLE = 1
DCCL.PAIR_FEATURE_LR_MULT = 1.0
DCCL.PAIR_FEATURE_MAX_GATE = 0.05
DCCL.PAIR_FEATURE_GATE_INIT = -2.0
DCCL.PAIR_FLOW_RANK = 16
DCCL.PAIR_FLOW_MIN_COUNT = 5
DCCL.PAIR_FLOW_MIN_CYCLES = 2
```

Do not run a rank, threshold, or gate grid against target labels.

## Step 0: AC Mechanism Preflight

Run one task before spending a full 12-task job:

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_pair_feature_preflight.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_pair_feature_preflight_accuracy.csv
output/uda/office-home/temporal_precision_head_pair_feature_preflight_summary.json
output/uda/office-home/temporal_precision_head_pair_feature_preflight_flow.json
```

Proceed only if the JSON says `pass_mechanism_preflight`. This requires a
nonzero pair-flow rank and nonzero router norm. AC accuracy is diagnostic at
this step and is not the full performance gate.

## Step 1: Seed-2022 Gate

```bash
cd /openbayes/home/vlm-sfda-conflict/duet-sfda-main
bash tools/run_office_home_temporal_precision_head_pair_feature_seed2022.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_pair_feature_seed2022_accuracy.csv
output/uda/office-home/temporal_precision_head_pair_feature_seed2022_summary.json
output/uda/office-home/temporal_precision_head_pair_feature_seed2022_peak_gap.json
output/uda/office-home/temporal_precision_head_pair_feature_seed2022_flow.json
```

The script explicitly extracts `--selection peak`. The gate requires:

```text
peak mean > 84.7225
active pair-flow basis on at least 10/12 tasks
nonzero trained router on at least 10/12 tasks
worst task delta vs DUET >= -1.50
```

If activation or router training fails, this is a mechanism failure and the
implementation must be audited. If the mechanism activates but accuracy fails,
archive the result and stop this learned-router family; do not tune its fixed
parameters.

## Step 2: Three-Seed Stability

Run only if Step 1 returns `pass_seed2022_gate`:

```bash
cd /openbayes/home/vlm-sfda-conflict/duet-sfda-main
bash tools/run_office_home_temporal_precision_head_pair_feature_seed_sweep.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_pair_feature_seed_sweep_accuracy.csv
output/uda/office-home/temporal_precision_head_pair_feature_seed_sweep_summary.json
output/uda/office-home/temporal_precision_head_pair_feature_seed_sweep_flow.json
```

The stability gate remains:

```text
all three peak-selected seed means > 84.7167
seed-mean sample standard deviation <= 0.10
mean over seed means > Stage14 peak mean 84.7825
```

## Failure Route

If Stage19 activates correctly but fails the accuracy gate, the next method is
agreement-anchor class-conditional covariance subspace transport. Reliable
source/CLIP agreements would estimate per-class means and low-rank covariance
geometry; conflict samples would contribute only soft candidate mass, and a
closed-form bounded transport would replace the learned router. This tests
manifold geometry rather than another loss, graph rule, or head variant. It
becomes a new stage only after implementation and an execution gate exist.

## Status

```text
implementation complete
local validation passed (48 tests)
AC mechanism preflight passed
seed-2022 cloud result observed: fail
```

## Seed-2022 Result

The adapter activated and trained on all 12 tasks, so this is a valid test of
the Stage19 mechanism rather than another inactive control.

| Metric | Stage19 | Required |
|---|---:|---:|
| Peak mean | 84.6908 | >84.7225 |
| Delta vs DUET | -0.0258 | >0 |
| Delta vs historical Stage14 seed 2022 | +0.0125 | diagnostic |
| Delta vs matched online Stage14 | -0.0317 | >0 |
| Wins/ties vs DUET | 6/12 | diagnostic |
| Active/trained tasks | 12/12 | >=10/12 |
| Worst task delta vs DUET | -0.53 | >=-1.50 |

Task results:

| Task | Peak | Delta vs DUET | Active rank |
|---|---:|---:|---:|
| AC | 73.77 | +0.17 | 12 |
| AP | 91.01 | +0.61 | 16 |
| AR | 91.05 | +0.05 | 13 |
| CA | 83.60 | 0.00 | 2 |
| CP | 90.92 | +0.22 | 16 |
| CR | 90.77 | -0.13 | 16 |
| PA | 82.49 | -0.21 | 2 |
| PC | 73.65 | -0.05 | 9 |
| PR | 90.73 | -0.47 | 14 |
| RA | 83.07 | -0.53 | 1 |
| RC | 73.97 | -0.03 | 11 |
| RP | 91.26 | +0.06 | 9 |

Peak selection recovers only 0.1175 mean points over the final checkpoints and
still misses DUET, so checkpoint timing is not the remaining explanation.

## Stage19-C: Subspace-Coverage Protection

The valid run exposes one architectural defect rather than a gate/gain
hyperparameter issue. The feature residual is norm-bounded after the active
directions are combined. Consequently, a one-direction basis on RA can receive
the same maximum relative displacement as a complete 16-direction basis on
AP. `CA`, `PA`, and `RA` are exactly the target-Art tasks and activate only
`2`, `2`, and `1` directions; relative to the matched online Stage14 control,
Stage19 changes them by `-0.08`, `-0.62`, and `-0.45` points.

Stage19-C adds one data-independent identifiability condition:

```text
PAIR_FEATURE_MIN_ACTIVE_RANK = PAIR_FLOW_RANK / 2 = 8
```

If fewer than eight independent directions exist, the representation adapter
is an exact identity, its effective gate is zero, and its router receives no
gradient. The complete Stage14 path remains active. At rank eight or above,
the Stage19 implementation is unchanged. This is not a rank sweep, graph
rule, per-sample selector, or loss change. Local validation passes all 54
tests; the standalone configuration also passes YAML/schema-value checks.

A historical replacement diagnostic uses the archived Stage19 results on the
nine rank-sufficient tasks and the matched online Stage14 values on the three
under-covered target-Art tasks:

```text
projected peak mean = 84.7867
delta vs DUET = +0.0700
```

This projection uses target accuracies and is not a result. It only justifies
the following three-task preflight.

### Target-Art Preflight

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_pair_feature_coverage_preflight.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_pair_feature_coverage_preflight_accuracy.csv
output/uda/office-home/temporal_precision_head_pair_feature_coverage_preflight_summary.json
output/uda/office-home/temporal_precision_head_pair_feature_coverage_preflight_flow.json
```

Run the complete gate only if the summary says `pass_coverage_preflight`. It
must verify exact fallback on `CA/PA/RA`, recover at least 0.20 target-Art mean
points over Stage19, and project above 84.7225.

### Complete Seed-2022 Gate

```bash
bash tools/run_office_home_temporal_precision_head_pair_feature_coverage_seed2022.sh
```

Bring back the four `pair_feature_coverage_seed2022` CSV/JSON files. A valid
pass requires peak mean above 84.7225 and exact agreement between active rank
and the coverage policy on all tasks. If either gate fails, close learned
pair routing and implement the already specified agreement-anchor
class-conditional covariance transport as Stage20.
