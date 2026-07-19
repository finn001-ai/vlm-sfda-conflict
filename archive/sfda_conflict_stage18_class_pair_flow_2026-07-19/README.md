# Stage 18: Persistent Class-Pair Conflict Flow

Date: 2026-07-19

## Motivation

Stage17 failed on seed 2022:

| Method | Peak mean | Delta vs DUET |
|---|---:|---:|
| Stage17 trajectory ensemble | 84.6542 | -0.0625 |
| Matched online Stage14 | 84.7225 | +0.0058 |
| Historical Stage14 seed 2022 | 84.6783 | -0.0383 |

The trajectory ensemble lost 0.0683 points against its matched online control.
Fixed snapshot averaging is therefore closed; no further snapshot combination
sweep should be run.

Stage14 remains the base. Its failure pattern is concentrated in task-specific
class confusions, while earlier experiments showed that per-sample confidence,
prototype, neighbor, dual-graph, and fixed topology decisions do not identify
the correct conflict candidate reliably.

## Method

Stage18 changes the classifier parameterization instead of adding another loss
or hard selector.

At each adaptation cycle:

1. Take only source/CLIP conflicts resolved by Stage14 temporal precision
   memory.
2. Count the dataset-level direction `loser class -> resolved winner class`.
3. Keep directions supported by at least five samples in at least two cycles.
4. For each unordered class pair, retain only its net-dominant direction.
5. Greedily select at most 16 non-redundant directions. Their undirected edges
   form a forest, so every selected vector adds a linearly independent
   dimension to the class-confusion subspace; then freeze the basis.
6. Learn a zero-initialized feature-to-pair coefficient map. Its output can
   move logits only along the frozen winner-minus-loser directions.
7. Bound the correction by a learned gate with maximum 0.3 times each sample's
   source-logit scale.

The adapted logits are:

```text
source_logits + bounded_gate * source_scale * tanh(coeff(features) @ pair_basis)
```

The basis is dataset-level and persistent. It does not decide the label of an
individual sample, modify the prompt, introduce a graph rule, or add a loss.
Before the basis activates, and at initialization, the output is exactly the
source logits.

## Fixed Config

```text
DCCL.TARGET_HEAD_VARIANT = pair_flow
DCCL.PAIR_FLOW_RANK = 16
DCCL.PAIR_FLOW_MIN_COUNT = 5
DCCL.PAIR_FLOW_MIN_CYCLES = 2
DCCL.PAIR_FLOW_MAX_GATE = 0.3
DCCL.PAIR_FLOW_GATE_INIT = -2.0
DCCL.TARGET_HEAD_EMA = False
DCCL.TRAJECTORY_ENSEMBLE = False
```

All other optimization, Stage14 temporal precision, graph residual, CLIP, and
loss settings remain unchanged. Do not tune these five pair-flow parameters on
target labels before the fixed gate is evaluated.

## Step 1: Seed-2022 Gate

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_pair_flow_seed2022.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_pair_flow_seed2022_accuracy.csv
output/uda/office-home/temporal_precision_head_pair_flow_seed2022_summary.json
output/uda/office-home/temporal_precision_head_pair_flow_seed2022_peak_gap.json
```

The script explicitly uses `--selection peak`. The gate requires:

```text
peak mean > 84.7225 (DUET and the matched Stage17 online control)
active pair-flow basis on at least 10/12 tasks
worst task delta vs DUET >= -1.50
```

If the JSON says `fail_seed2022_gate`, stop and archive the result. Do not run
the seed sweep and do not make a rank/threshold/gate grid.

## Step 2: Three-Seed Stability

Run only after Step 1 passes:

```bash
cd /openbayes/home/vlm-sfda-conflict/duet-sfda-main
bash tools/run_office_home_temporal_precision_head_pair_flow_seed_sweep.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_pair_flow_seed_sweep_accuracy.csv
output/uda/office-home/temporal_precision_head_pair_flow_seed_sweep_summary.json
```

The stability gate requires:

```text
all three peak-selected seed means > 84.7167
seed-mean sample standard deviation <= 0.10
mean over seed means > Stage14 peak mean 84.7825
```

## Failure Route

If Stage18 fails, the next method is not another classifier loss, graph rule,
or class-pair hyperparameter sweep. Replace classifier-only adaptation with
class-pair-specific low-rank feature adapters, softly routed by the aggregate
conflict-flow posterior. That moves the intervention into representation space
while retaining the dataset-level evidence constraint. It becomes a new stage
only after the implementation and execution plan exist.

## Status

```text
implementation complete
local validation passed (35 tests)
seed-2022 cloud run observed
decision = invalid_mechanism_run
do not run the Stage18 seed sweep
```

## Seed-2022 Result And Root Cause

| Metric | Stage18 | Required |
|---|---:|---:|
| Peak mean | 84.6075 | >84.7225 |
| Delta vs DUET | -0.1092 | >0 |
| Delta vs historical Stage14 seed 2022 | -0.0708 | >0 |
| Active pair-flow tasks | 0/12 | >=10/12 |
| Worst task delta vs DUET | -0.94 | >=-1.50 |
| Peak minus final | +0.0250 | diagnostic only |

The classifier correction was exactly inactive on every task: active rank was
zero and the gate stayed at its initialization value `0.035761`.

Root cause:

```text
label_mask = temporally stable source/CLIP agreement samples
flow validity = label_mask AND (source_label != clip_label)
```

The two conditions are mutually exclusive at the point where the mask is
constructed. Therefore the flow counter received zero valid conflicts, no
class-pair basis was created, and the low-rank head never received a gradient.
The reported accuracy is an inactive source-anchored control, not evidence
against aggregate class-pair flow.

Conclusion:

```text
do not lower PAIR_FLOW_MIN_COUNT or PAIR_FLOW_MIN_CYCLES
do not rerun the classifier-only Stage18 implementation
retain the result as an implementation-invalidation record
move to the already specified representation-space route
```

The next implementation must preserve the complete Stage14 blend target head,
freeze each sample's initial source/CLIP conflict pair, aggregate the current
mix probability mass on those two fixed candidates, and use persistent net
class-pair directions to constrain a bounded low-rank feature adapter.
