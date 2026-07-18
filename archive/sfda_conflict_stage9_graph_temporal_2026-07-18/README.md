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

No cloud training result has been observed yet.
