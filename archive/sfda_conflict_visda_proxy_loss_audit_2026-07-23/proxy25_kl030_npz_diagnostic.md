# KL 0.3 Temporal NPZ Diagnostic

Date: 2026-07-23

## Scope

This is a zero-training, oracle-labeled mechanism diagnostic over the 13,847
samples in the deterministic VisDA 25% adaptation subset. It analyzes the four
cycle-boundary NPZ files from the failed `KL_PAR=0.3` run.

The labels are used only to evaluate candidate signals. No validation label may
enter a training rule, class list, or threshold.

## Overall temporal result

| Metric | Value |
|---|---:|
| Initial source/CLIP conflicts | 6,911 |
| Final source/CLIP conflicts | 1,729 |
| Stable initial-conflict coverage | 92.52% |
| Stable teacher corrections over final CLIP | 308 |
| Stable teacher degradations from final CLIP | 184 |
| Net stable corrections | +124 |

Cycle-4 boundary accuracy on the proxy subset:

| Prediction | Micro accuracy | Macro accuracy |
|---|---:|---:|
| Task/source | 84.70 | 87.22 |
| CLIP | 87.04 | 89.47 |
| Probability-average mix | 87.59 | 89.99 |
| Graph teacher | 88.20 | 90.09 |

The graph-temporal teacher therefore contains useful information globally.
The failed end-to-end result is not evidence that its entire posterior is
random.

## Car/truck asymmetry

| True class | CLIP | Mix | Graph teacher | Teacher minus CLIP |
|---|---:|---:|---:|---:|
| Car | 75.46 | 76.42 | 83.08 | +7.62 |
| Person | 81.70 | 81.80 | 81.20 | -0.50 |
| Truck | 70.44 | 69.21 | 60.63 | -9.81 |

On stable samples that were source/CLIP conflicts in Cycle 1:

| True class | Selected | Corrections | Degradations | Net |
|---|---:|---:|---:|---:|
| Car | 1,001 | 167 | 9 | +158 |
| Person | 846 | 24 | 25 | -1 |
| Truck | 956 | 13 | 120 | -107 |

For true truck samples, the graph teacher predicts car on `21.77%`, versus
`12.47%` for CLIP. Its truck prediction rate falls to `60.63%`, versus
`70.44%` for CLIP. This directly explains why stronger GTR repeatedly raises
car while lowering truck.

The largest directed conflict pair is `task=car, CLIP=truck`:

```text
149 samples
72 true car
73 true truck
4 other
```

The two labels are almost perfectly balanced inside the same observable
conflict direction. Class identity or conflict direction alone therefore
cannot select the correct teacher side.

## Reliability-signal audit

For the general problem of choosing the CLIP side versus the task side on
one-sided final conflicts, the best inspected signal was graph-teacher
probability advantage:

```text
teacher_prob(CLIP label) - teacher_prob(task label)
ROC AUC = 0.763
```

However, on the critical `task=car, CLIP=truck` pair, its AUC falls to `0.586`.
The relative CLIP-versus-task margin reaches only `0.656`. This is not strong
enough to justify a hard conflict decision.

More importantly, among the 530 samples whose top-1 is actually changed by the
graph teacher:

```text
corrections = 277
degradations = 193
both predictions wrong = 60
oracle net correction = +84
```

The best inspected label-free signal for predicting whether the graph change
is a correction or degradation was:

```text
teacher confidence - base-mix confidence
ROC AUC = 0.587
```

Teacher margin, entropy confidence, temporal stability, posterior shift, and
source/CLIP-side support all remained close to random for this decision.

## Fixed-gate counterfactual

The strongest simple fixed gate applied graph changes only when teacher margin
exceeded base margin and the teacher label was stable from Cycle 3 to Cycle 4:

| Metric | Base mix | Gated teacher | Delta |
|---|---:|---:|---:|
| Macro accuracy | 89.99 | 90.19 | +0.20 |
| Hard-class mean | 75.81 | 75.92 | +0.11 |
| Car | 76.42 | 80.77 | +4.35 |
| Person | 81.80 | 81.60 | -0.20 |
| Truck | 69.21 | 65.39 | -3.82 |

This counterfactual still obtains its macro gain through a large car/truck
exchange. It fails the no-compensation mechanism gate and must not be promoted
to a training experiment.

## Decision

```text
reject simple confidence/margin-conditioned KL routing
reject another graph-teacher gate training run
do not hard-code car or truck
do not run KL 0.5
```

The next training evidence must be a same-environment, same-proxy original
PLMatch control. It determines whether DCCL is improving over its actual local
base or whether the remaining gap is primarily an environment/source-model
comparison problem.
