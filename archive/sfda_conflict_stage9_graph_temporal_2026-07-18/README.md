# Stage 9: Graph-Fused Teacher With Temporal Validation

Date: 2026-07-18

## Decision

Two independent probes passed:

| Probe | Signal |
|---|---|
| temporal conflict dynamics | stable final-cycle teacher predictions beat final CLIP on initial conflicts |
| graph-teacher fusion | continuous graph/posterior fusion improves `both_prior` teacher top-1 |

They are combined conservatively:

```text
graph fusion enters training as a soft teacher
temporal dynamics remains a validation gate and diagnostic
```

No temporal hard-label loss is added yet. The temporal probe gain is real but
small, so using it as a hard-label selector would repeat the earlier failure
mode of turning a modest reliability signal into noisy supervision.

## Training Mechanism

At each adaptation cycle:

1. Run the existing `both_prior` calibration.
2. Build the existing source/CLIP agreement hard-label mask.
3. Run dual-space graph diffusion from agreement anchors.
4. Fuse:

```text
q_teacher = product_of_experts(q_both, q_graph, entropy_adaptive_weight)
```

5. Use `q_teacher` for:

```text
CLIP visual update target
task-model KL target
```

6. Keep hard pseudo-label admission unchanged.
7. Export temporal diagnostics for the fused teacher trajectory.

## Cloud Command

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_graph_temporal_clipart.sh
```

The script runs:

```text
A->C, P->C, R->C
```

and writes temporal diagnostics:

```text
output/uda/office-home/graph_temporal_dynamics_probe.json
```

## Gate For Full 12 Tasks

Compare final training accuracy against:

| Task | both_prior | DUET paper |
|---|---:|---:|
| A->C | 72.78 | 73.60 |
| P->C | 72.81 | 73.70 |
| R->C | 72.97 | 74.00 |

Expand to all 12 only if:

```text
at least two target-Clipart tasks improve over both_prior
and at least one materially closes the gap to the DUET paper number
```

If accuracy fails but temporal diagnostics remain strong, the next action may
consider a soft temporal consistency term. Do not add hard temporal labels
without a new gate.

## Implementation

```text
duet-sfda-main/cfgs/office-home/graph_temporal.yaml
duet-sfda-main/src/utils/conflict_diffusion.py
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/tools/run_office_home_graph_temporal_clipart.sh
```

## Status

Cloud target-Clipart training has been observed.

| Task | graph_temporal | both_prior | DUET paper | Delta vs both_prior | Delta vs DUET |
|---|---:|---:|---:|---:|---:|
| A->C | 73.15 | 72.78 | 73.60 | +0.37 | -0.45 |
| P->C | 72.30 | 72.81 | 73.70 | -0.51 | -1.40 |
| R->C | 72.71 | 72.97 | 74.00 | -0.26 | -1.29 |

Mean target-Clipart accuracy is 72.72. The expansion gate fails because only
one of three target-Clipart tasks improves over `both_prior`, and the average
does not close the DUET gap.

## Conclusion

Do not expand this exact graph-temporal teacher-fusion training configuration
to all 12 Office-Home tasks.

The result is useful as a negative control: graph-teacher fusion improves the
offline teacher probe, but feeding that fused teacher directly into both CLIP
visual update and task-model KL is not robust in training. This supports the
current interpretation that graph diffusion contains signal, but the training
injection needs to be weaker or more selectively applied than the default
continuous teacher replacement used here.

## Follow-up Probe

The next test keeps the graph-fused teacher out of CLIP visual adaptation and
uses it only as the task-model KL target:

```text
MODEL.METHOD = graph_temporal_kl_only
DCCL.GRAPH_TEACHER_FUSION = True
DCCL.GTF_APPLY_TO = kl
```

This is a weaker injection test, not a new stage. It directly checks whether
the previous failure came from graph teacher feedback into the CLIP visual
branch.

Cloud command:

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_graph_temporal_kl_only_clipart.sh
```

Bring back both generated files:

```text
output/uda/office-home/graph_temporal_kl_only_clipart_accuracy.csv
output/uda/office-home/graph_temporal_kl_only_dynamics_probe.json
```

Gate:

```text
expand only if at least two target-Clipart tasks improve over both_prior
and the mean is above the previous graph_temporal mean of 72.72
```

Observed KL-only temporal diagnostics:

| Task | Stable coverage | Stable teacher acc | Stable CLIP acc | Net gain vs final CLIP | p-value |
|---|---:|---:|---:|---:|---:|
| A->C | 86.83 | 67.25 | 64.10 | +68 | 0.000001 |
| P->C | 87.32 | 66.25 | 64.33 | +45 | 0.001563 |
| R->C | 87.13 | 65.95 | 63.46 | +52 | 0.000203 |

Decision:

```text
temporal diagnostics pass on 3/3 target-Clipart tasks
training expansion gate is still pending final accuracy CSV
```

The KL-only probe preserves the temporal signal and increases net gains over
the earlier temporal-only probe. This supports the hypothesis that the previous
graph-temporal failure was likely caused by graph-fused teacher feedback into
the CLIP visual branch, not by the graph signal itself. Do not expand until the
final training accuracy CSV confirms at least two target-Clipart improvements.
