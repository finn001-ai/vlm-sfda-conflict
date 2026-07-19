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

## Stage19-C Preflight Result

The target-Art preflight failed while the coverage mechanism behaved exactly
as specified:

| Task | Stage19-C peak | Stage19 peak | Matched online Stage14 | Rank | Adapter |
|---|---:|---:|---:|---:|---|
| CA | 83.60 | 83.60 | 83.68 | 2 | fallback |
| PA | 82.61 | 82.49 | 83.11 | 2 | fallback |
| RA | 82.98 | 83.07 | 83.52 | 1 | fallback |
| Mean | 83.0633 | 83.0533 | 83.4367 | - | - |

Diagnostics confirm `router_norm=0`, `pair_feature_effective=False`, and the
correct fallback decision on all three tasks. Nevertheless, the target-Art
mean recovers only `+0.01`, and replacing these three archived Stage19 rows
projects a full mean of only `84.6933`, still `-0.0292` below the required
`84.7225`.

Conclusion:

```text
decision = fail_coverage_preflight
do not run the 12-task Stage19-C script
the low active rank correlation was not the main causal explanation
close rank thresholds, gate scaling, and all learned pair-router variants
move to Stage20 agreement-anchor covariance transport
```

## Scope Re-evaluation: Stage19-G

The previous closure bundled two different claims: the Stage19-C rank fallback
failed, and all learned pair routing should stop. Only the first claim follows
from the preflight. The project scope was subsequently clarified: graph
structure and auxiliary losses remain admissible when they have a specific
mechanistic role; prompt-only adjustment is excluded.

Re-reading Stages 8-11 changes the next experiment:

- graph diffusion improved the offline teacher by `+1.24` to `+2.52` points;
- direct graph-teacher replacement failed in Stage9;
- low-weight graph-temporal residual training improved two of three Clipart
  tasks in Stage10 and remained part of the strongest Stage14 baseline;
- Stage19 learned its router from CE, ordinary KL, consistency, and GTR at the
  same time, so its aggregate pair directions had no mechanism-specific
  gradient attribution.

Stage20 covariance transport is therefore deferred before cloud execution.
The next controlled test remains in Stage19 and changes the training interface,
not the graph rule, prompt, gate, or target head.

### Stage19-G Method

The forward model keeps the Stage19 bounded feature residual. During the main
Stage14 loss path, that residual is detached:

```text
z_main = z + stop_gradient(delta_pair(z))
```

The backbone and blend head still train normally through the identity path,
but CE, ordinary KL, consistency, and the original full-model GTR cannot update
the pair router. A second branch uses detached backbone features and the frozen
source classifier:

```text
z_route = stop_gradient(z) + delta_pair(stop_gradient(z))
L_route = weighted_KL(source_head(z_route), stable_graph_temporal_teacher)
```

Only samples admitted by the existing reversible two-cycle GTR memory have
nonzero weights. `L_route` uses the already validated `GTR_PAR=0.05`; no new
loss weight is introduced. Its gradients reach only the pair router and gate,
and optimizer weight decay is disabled for those parameters in this mode.
The original Stage14 GTR remains unchanged for the backbone and blend head.

Fixed differences from the valid Stage19 run:

```text
PAIR_FEATURE_GRADIENT_MODE = gtr_only
PAIR_FEATURE_MIN_ACTIVE_RANK = 1
COV_TRANSPORT_ADAPT = False
all other Stage14/Stage19 values unchanged
```

This directly tests whether aggregate conflict directions become useful when
their coefficients are learned only from the graph-temporal evidence that
survived the earlier teacher-interface tests.

### Step 0: AC/PA/RA Preflight

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_pair_feature_gtr_preflight.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_pair_feature_gtr_preflight_accuracy.csv
output/uda/office-home/temporal_precision_head_pair_feature_gtr_preflight_summary.json
output/uda/office-home/temporal_precision_head_pair_feature_gtr_preflight_flow.json
```

All reported accuracies use target-label-selected `peak`, as requested. The
preflight passes only if every task has a valid `gtr_only` route, the AC/PA/RA
mean exceeds the matched online Stage14 mean `80.0733`, the projected complete
mean exceeds `84.7225`, and no task loses more than `0.30` from matched Stage14.
The projection reuses nine archived Stage19 rows and is only a compute gate.

### Step 1: Complete Seed-2022 Gate

Run only after `pass_gtr_preflight`:

```bash
bash tools/run_office_home_temporal_precision_head_pair_feature_gtr_seed2022.sh
```

Bring back the four `pair_feature_gtr_seed2022` CSV/JSON files. A pass requires
peak mean above `84.7225`, at least ten active/trained pair adapters, a valid
GTR-only route on all 12 tasks, and worst task delta versus DUET above `-1.50`.

### Step 2: Three-Seed Stability

Run only after `pass_gtr_seed2022_gate`:

```bash
bash tools/run_office_home_temporal_precision_head_pair_feature_gtr_seed_sweep.sh
```

The stability protocol remains unchanged: all three peak means above DUET,
sample standard deviation at most `0.10`, and mean over seeds above the Stage14
peak mean `84.7825`.

### Stage19-G Failure Route

If the route diagnostics fail, audit the implementation before interpreting
accuracy. If the route is valid but the AC/PA/RA or complete accuracy gate
fails, archive the result and do not sweep `GTR_PAR`, gate size, rank, or graph
parameters. Resume the already implemented Stage20 covariance preflight. That
is an orthogonal test of agreement geometry and remains the next proposal.

Stage19-G local status:

```text
implementation complete
local validation passed (70 tests)
cloud preflight complete: valid mechanism failure
```

## Stage19-G Preflight Result

The `gtr_only` mechanism activated correctly on all three tasks. This rules out
implementation inactivity as the reason for failure:

| Task | Peak | Stage19 | Matched Stage14 | Rank | GTR active | Router norm |
|---|---:|---:|---:|---:|---:|---:|
| AC | 73.61 | 73.77 | 73.59 | 12 | 151 | 0.035880 |
| PA | 82.53 | 82.49 | 83.11 | 2 | 67 | 0.015814 |
| RA | 82.94 | 83.07 | 83.52 | 1 | 63 | 0.003765 |

Aggregate gate:

```text
decision = fail_gtr_preflight
route diagnostics = pass (3/3)
Stage19 subset mean = 79.7767
matched online Stage14 subset mean = 80.0733
Stage19-G subset mean = 79.6933
delta vs matched Stage14 = -0.3800
projected complete mean = 84.6700
projected delta vs required 84.7225 = -0.0525
```

AC remained essentially at its matched baseline (`+0.02`), while PA and RA
each lost `0.58`. Isolating the router from generic losses therefore did not
repair the weak target-Art tasks. The original joint Stage19 subset mean was
also `0.0834` higher than Stage19-G, so generic-gradient interference was not
the missing causal explanation.

Conclusion:

```text
do not run the complete 12-task Stage19-G script
close gtr_only learned pair-feature routing without parameter sweeps
retain graph-temporal residual training inside the Stage14 baseline
run the fixed Stage20 agreement-covariance preflight next
```

This failure does not imply that graph information or losses are generally
invalid. It specifically rejects using stable GTR evidence as the sole trainer
of the current aggregate pair-direction router.
