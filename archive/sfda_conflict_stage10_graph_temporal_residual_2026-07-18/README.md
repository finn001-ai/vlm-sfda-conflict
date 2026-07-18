# Stage 10: Graph-Temporal Residual Regularization

Date: 2026-07-18

## Rationale

Stage 9 closed direct graph-fused teacher replacement:

| Variant | Result |
|---|---|
| graph_temporal | dynamics signal exists, but final target-Clipart training gate fails |
| graph_temporal_kl_only | dynamics pass, but final target-Clipart training gate fails |

The remaining signal is narrower:

```text
graph diffusion and temporal stability identify useful conflict decisions offline
direct teacher replacement does not convert the signal into robust training
```

This stage changes the training interface. The baseline `both_prior` teacher is
preserved for CLIP visual update and the main task-model KL. Graph-temporal
evidence is used only as a low-weight residual regularizer on stable conflict
samples.

## Method

At each adaptation cycle:

1. Run the existing `both_prior` calibration.
2. Keep the original pseudo-label mask, CLIP visual update target, and main KL
   target unchanged.
3. Run graph diffusion and build the entropy-adaptive fused teacher only for
   diagnostics and residual evidence.
4. Track temporally stable residual candidates where:

```text
source top-1 != CLIP top-1
fused-teacher top-1 == graph top-1
the same fused-teacher top-1 persists for GTR_STABLE_CYCLES
graph entropy confidence >= GTR_MIN_GRAPH_CONF
baseline support for that label is low enough
```

5. Add a small auxiliary KL term toward the fused teacher:

```text
loss = baseline_loss + GTR_PAR * weighted_KL(model_prediction, fused_teacher)
```

This is not a hard label, not candidate transport, and not teacher replacement.
The main teacher path remains the same as `both_prior`.

## Default Config

```text
MODEL.METHOD = graph_temporal_residual
DCCL.GRAPH_TEACHER_FUSION = True
DCCL.GTF_APPLY_TO = none
DCCL.GTR_PAR = 0.05
DCCL.GTR_STABLE_CYCLES = 2
DCCL.GTR_MEMORY = reversible
DCCL.GTR_MIN_GRAPH_CONF = 0.05
DCCL.GTR_MIN_DISAGREEMENT = 0.25
```

Implementation:

```text
duet-sfda-main/cfgs/office-home/graph_temporal_residual.yaml
duet-sfda-main/src/utils/conflict_diffusion.py
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/tools/run_office_home_graph_temporal_residual_clipart.sh
```

## Cloud Command

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_graph_temporal_residual_clipart.sh
```

Bring back both generated files:

```text
output/uda/office-home/graph_temporal_residual_clipart_accuracy.csv
output/uda/office-home/graph_temporal_residual_dynamics_probe.json
```

## Gate

Compare against:

| Task | both_prior | DUET paper |
|---|---:|---:|
| A->C | 72.78 | 73.60 |
| P->C | 72.81 | 73.70 |
| R->C | 72.97 | 74.00 |

Expand to all 12 only if:

```text
at least two target-Clipart tasks improve over both_prior
mean target-Clipart accuracy beats 72.72
at least one task materially closes the DUET paper gap
```

If this fails, archive the result and stop graph-temporal residual
regularization unless a new diagnostic shows a different failure mode.

## Result

Cloud target-Clipart training has been observed.

| Task | graph_temporal_residual | both_prior | DUET paper | Delta vs both_prior | Delta vs DUET |
|---|---:|---:|---:|---:|---:|
| A->C | 72.99 | 72.78 | 73.60 | +0.21 | -0.61 |
| P->C | 73.15 | 72.81 | 73.70 | +0.34 | -0.55 |
| R->C | 72.90 | 72.97 | 74.00 | -0.07 | -1.10 |

Mean target-Clipart accuracy is 73.01. This improves over both_prior on two
of three target-Clipart tasks and beats the previous `graph_temporal` mean of
72.72, but it remains below the DUET paper values.

Observed temporal diagnostics still pass on 3/3 tasks:

| Task | Stable coverage | Stable teacher acc | Stable CLIP acc | Net gain vs final CLIP |
|---|---:|---:|---:|---:|
| A->C | 88.07 | 66.56 | 64.09 | +54 |
| P->C | 88.70 | 66.74 | 65.05 | +40 |
| R->C | 87.79 | 66.97 | 63.69 | +69 |

The important failure mode is not lack of graph/temporal signal. It is noisy
pseudo-label admission: by the final cycle, valid pseudo labels grow to about
4000/4365 samples while their accuracy drops to about 76-77%. The next method
should change pseudo-label memory/admission rather than add another loss term.
