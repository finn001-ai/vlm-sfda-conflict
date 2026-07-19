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
cloud result pending
```
