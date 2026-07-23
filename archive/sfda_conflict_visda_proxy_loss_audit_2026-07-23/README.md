# VisDA-C 25% Proxy Loss Audit

Date: 2026-07-23

## Purpose

This archive records the compute-gated VisDA-C loss and residual audit that
followed the completed Stage14 seed-2020 transfer. It uses a fixed,
class-proportional 25% target adaptation list while retaining evaluation on the
complete 55,388-image validation set.

The proxy is an internal, target-label-assisted development instrument. Its
accuracies are not final SFDA results and must not be compared directly with
the full-data Stage14 accuracy.

## Proxy construction

Implementation commit:

```text
4d55b89 Add VisDA proxy subset preflights
```

The deterministic list is generated without copying images:

```bash
python tools/prepare_visda_proxy_subset.py \
  --input data/VISDA-C/validation_list.txt \
  --output data/VISDA-C/validation_proxy25_seed2020_list.txt \
  --ratio 0.25 \
  --seed 2020 \
  --force
```

Expected sizes:

```text
full evaluation samples = 55,388
proxy adaptation samples = 13,847
```

The `target` and `test_aug` loaders use the proxy list. The final `test`
loader continues to use the complete target list.

## Reference configuration

Unless explicitly changed below:

```yaml
ACTIVE:
  CYCLE: 4
  CLS_PAR: 0.4
  CON_PAR: 0.2
  KL_PAR: 0.4

DCCL:
  ADAPTATION_LIST: data/VISDA-C/validation_proxy25_seed2020_list.txt
  CALIB_POWER: 0.5
  TARGET_HEAD_MIX: 0.3
  PL_STABLE_CYCLES: 2
  GTR_STABLE_CYCLES: 2
  GTF_APPLY_TO: none
```

The primary matched reference is P1 (`GTR_PAR=0`):

```text
final macro accuracy = 87.83
hard-class mean (car/person/truck) = 73.79
remaining-nine-class mean = 92.51
```

## Completed runs

### GTR coefficient audit

| Run | GTR | Car | Person | Truck | Hard mean | Other-9 mean | Final macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.05 | 76.95 | 80.27 | 63.82 | 73.68 | 92.52 | 87.81 |
| P1 | 0.00 | 75.36 | 80.68 | 65.34 | 73.79 | 92.51 | 87.83 |
| P2 | 0.10 | 78.17 | 80.05 | 62.69 | 73.64 | 92.55 | 87.82 |

Interpretation:

```text
increasing GTR -> car rises, truck falls
macro accuracy remains within 0.02 pp
```

GTR is active but acts mainly as a car/truck boundary redistribution. It does
not increase global discrimination on this proxy. Do not run a full VisDA GTR
coefficient job and do not continue to larger coefficients.

The P1 and P2 cloud commands accidentally reused the P0 method/output name.
Their configuration dumps and sequential logs verify the actual `0.0` and
`0.1` coefficients, but artifacts sharing that output directory must not be
treated as independently named runs.

### Combined CLS/CON/GTR run

Actual configuration:

```yaml
ACTIVE:
  CLS_PAR: 0.5
  CON_PAR: 0.3
  KL_PAR: 0.4
DCCL:
  GTR_PAR: 0.05
```

Final result:

| Car | Person | Truck | Hard mean | Other-9 mean | Final macro |
|---:|---:|---:|---:|---:|---:|
| 77.96 | 80.70 | 63.43 | 74.03 | 92.60 | 87.96 |

Matched checkpoint deltas versus P1:

```text
cycle-1 final: +0.54
cycle-2 final: +0.16
cycle-3 peak:  +0.19
cycle-3 final: +0.21
cycle-4 final: +0.13
```

The early advantage decays substantially, and the final result remains below
the predeclared `+0.15 pp` continuation gate. The run also retains the
car-up/truck-down exchange. It is a completed negative preflight, not evidence
for an eight-cycle job.

### Weak-teacher stop-gradient audit

Implementation and diagnostics commit:

```text
faa7bc6 Add stop-gradient consistency diagnostics
```

The experiment changed only the direct consistency gradient:

```text
legacy: weak and strong predictions both receive consistency gradients
candidate: weak probability is detached; strong prediction follows weak
```

Isolated configuration:

```yaml
ACTIVE:
  CLS_PAR: 0.4
  CON_PAR: 0.2
  KL_PAR: 0.4
DCCL:
  GTR_PAR: 0.0
  CONSISTENCY_STOP_GRAD: true
  LOSS_DIAG: true
```

Checkpoint comparison:

| Checkpoint | P1 legacy | Stop-gradient | Delta |
|---|---:|---:|---:|
| Cycle 1 final | 81.72 | 81.32 | -0.40 |
| Cycle 2 final | 85.31 | 85.11 | -0.20 |
| Cycle 3 peak | 87.17 | 86.99 | -0.18 |
| Cycle 3 final | 86.87 | 86.67 | -0.20 |
| Cycle 4 final | 87.83 | 87.66 | -0.17 |

Final class summary:

| Car | Person | Truck | Hard mean | Other-9 mean | Final macro |
|---:|---:|---:|---:|---:|---:|
| 74.40 | 79.90 | 66.17 | 73.49 | 92.38 | 87.66 |

Decision:

```text
fail proxy gate
do not use stop-gradient consistency
keep the legacy bidirectional consistency gradient
```

The implementation remains behind an opt-in configuration flag whose default
is `False`; therefore existing Office-Home and VisDA configurations retain
legacy behavior.

## Loss-magnitude diagnostics

The stop-gradient run produced the first per-cycle task-loss magnitude audit.
Shares are weighted loss-value shares across tracked terms, not strict
gradient-norm shares.

| Cycle | Consistency share | Stable CE share | CLIP KL share | GTR share |
|---:|---:|---:|---:|---:|
| 1 | 25.05% | 19.75% | 55.21% | 0% |
| 2 | 31.04% | 13.12% | 55.84% | 0% |
| 3 | 30.34% | 20.12% | 49.54% | 0% |
| 4 | 31.96% | 20.00% | 48.04% | 0% |

Although `CLS_PAR` and `KL_PAR` are both `0.4`, the CLIP KL contributes about
48-56% of the tracked weighted task-loss magnitude. With `KL_MODE=clip`,
`build_conflict_kl_target` applies the full CLIP distribution with unit weight
to every target sample; `KL_CANDIDATE` is inactive in this mode.

This identified the main KL coefficient as a scalar parameter worth auditing.
It did not by itself prove that KL gradients were excessive. The completed
`KL_PAR=0.3` result below shows that the large numeric share was not evidence
that the global CLIP constraint should be weakened.

## KL 0.3 audit

The predeclared isolated run completed normally:

```yaml
ACTIVE:
  CLS_PAR: 0.4
  CON_PAR: 0.2
  KL_PAR: 0.3
DCCL:
  GTR_PAR: 0.0
  CONSISTENCY_STOP_GRAD: false
  LOSS_DIAG: true
```

It failed every primary gate:

| Metric | P1 KL 0.4 | KL 0.3 | Delta |
|---|---:|---:|---:|
| Final macro | 87.83 | 87.61 | -0.22 |
| Hard mean | 73.79 | 73.44 | -0.35 |
| Other-9 mean | 92.51 | 92.34 | -0.17 |

All 16 evaluation checkpoints were below P1. The matched cycle-end trajectory
was:

| Checkpoint | P1 KL 0.4 | KL 0.3 | Delta |
|---|---:|---:|---:|
| Cycle 1 final | 81.72 | 81.18 | -0.54 |
| Cycle 2 final | 85.31 | 84.88 | -0.43 |
| Cycle 3 peak | 87.17 | 86.98 | -0.19 |
| Cycle 3 final | 86.87 | 86.66 | -0.21 |
| Cycle 4 final | 87.83 | 87.61 | -0.22 |

Final class changes show another car/truck boundary redistribution:

| Class | P1 KL 0.4 | KL 0.3 | Delta |
|---|---:|---:|---:|
| Car | 75.36 | 75.73 | +0.37 |
| Person | 80.68 | 80.68 | 0.00 |
| Truck | 65.34 | 63.91 | -1.43 |
| Bicycle | 86.01 | 85.29 | -0.72 |
| Skateboard | 94.56 | 94.04 | -0.52 |

The KL 0.3 loss-value shares remained substantial but lower:

| Cycle | Consistency share | Stable CE share | CLIP KL share |
|---:|---:|---:|---:|
| 1 | 28.49% | 21.17% | 50.34% |
| 2 | 34.98% | 14.41% | 50.61% |
| 3 | 33.28% | 22.95% | 43.77% |
| 4 | 34.68% | 23.08% | 42.25% |

At the Cycle-4 boundary, KL 0.3 had 48 fewer stable pseudo-labels, 52 more
source/CLIP conflicts, 25 fewer graph anchors, and a 0.16 pp lower pure task
output than P1. Pseudo-label precision was 0.05 pp higher, so the loss came
from weaker coverage and representation anchoring, not lower admitted-label
precision.

Interpretation:

```text
The full CLIP KL is acting as a useful stabilizer.
Its large loss-value share is not equivalent to excessive harmful gradient.
Lowering it globally releases a high-frequency car bias and damages truck.
```

VisDA compounds this effect because optimization averages over samples while
the reported metric averages over classes. The full validation set contains
10,401 car samples but only 5,548 truck samples and 4,000 person samples.
`both_prior` calibrates prediction marginals, but the consistency, stable CE,
and KL objectives are still reduced by sample count. A global coefficient
cannot repair this class/pair-asymmetric objective mismatch.

## Closed paths

Do not repeat or extend:

```text
GTR_PAR = 0 / 0.05 / 0.10 coefficient sweep
larger GTR coefficients
eight-cycle combined CLS=0.5, CON=0.3, GTR=0.05
weak-teacher stop-gradient consistency
direct graph-teacher main-KL injection
PL/GTR stability 3/3
global KL reduction from 0.4 to 0.3
blind continuation to KL 0.5
```

The earlier full-data stability-3 candidate used transferred Office-Home
parameters (`CALIB_POWER=0.8`, `TARGET_HEAD_MIX=0.5`) and `PL/GTR=3/3`. It
finished five cycles but remained below the matched canonical result; it must
not be launched for eight cycles.

## Next decision

No further DCCL training run is predeclared from this scalar audit. In
particular, do not infer that `KL_PAR=0.5` must help merely because `0.3`
hurts.

Before another method change:

1. Establish a same-environment, same-proxy PLMatch control. The current
   `91.4` VisDA number is an external reference rather than a locally archived
   matched run, so it cannot identify how much of the remaining gap is due to
   DCCL versus source weights/environment.
2. The KL 0.3 NPZ zero-training diagnostic is complete. It confirms that the
   graph teacher raises car while converting too many true trucks into car.
   The best simple label-free signal predicts beneficial versus harmful graph
   top-1 changes with ROC AUC only `0.587`.
3. Do not test a confidence/margin-conditioned KL or graph route. A fixed
   margin-and-stability gate still raises car by `4.35 pp` while lowering truck
   by `3.82 pp`; it fails the no-compensation gate.

The next training evidence is therefore the matched PLMatch control, not
another DCCL loss coefficient or graph gate. Full NPZ findings are in
`proxy25_kl030_npz_diagnostic.md`.

## Raw artifacts

```text
proxy25_gtr_sweep_raw.txt
proxy25_cls050_con030_gtr005_partial_raw.txt
proxy25_stopgrad_full_raw.txt
proxy25_kl030_full_raw.txt
proxy25_kl030_temporal_diagnostics.tar.gz
proxy25_kl030_temporal_dynamics.json
proxy25_kl030_npz_diagnostic.md
proxy25_results.csv
SHA256SUMS
```

The combined-run raw attachment ends at the first Cycle-4 checkpoint; its
remaining checkpoints and final class vector were supplied separately and are
recorded in the completed-run table above. `SHA256SUMS` preserves the hashes
of the copied raw attachments and temporal diagnostic bundle.
