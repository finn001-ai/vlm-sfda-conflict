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

This identifies the main KL coefficient as the next scalar parameter to audit.
It does not by itself prove that KL gradients are excessive.

## Closed paths

Do not repeat or extend:

```text
GTR_PAR = 0 / 0.05 / 0.10 coefficient sweep
larger GTR coefficients
eight-cycle combined CLS=0.5, CON=0.3, GTR=0.05
weak-teacher stop-gradient consistency
direct graph-teacher main-KL injection
PL/GTR stability 3/3
```

The earlier full-data stability-3 candidate used transferred Office-Home
parameters (`CALIB_POWER=0.8`, `TARGET_HEAD_MIX=0.5`) and `PL/GTR=3/3`. It
finished five cycles but remained below the matched canonical result; it must
not be launched for eight cycles.

## Next predeclared experiment

The next and only currently authorized proxy run changes:

```text
ACTIVE.KL_PAR: 0.4 -> 0.3
```

It restores all other P1 settings:

```yaml
ACTIVE:
  CYCLE: 4
  CLS_PAR: 0.4
  CON_PAR: 0.2
  KL_PAR: 0.3
DCCL:
  GTR_PAR: 0.0
  CONSISTENCY_STOP_GRAD: false
  LOSS_DIAG: true
```

Command:

```bash
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/temporal_precision_head.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD temporal_precision_head_visda_proxy25_kl030 \
  ACTIVE.CYCLE 4 \
  ACTIVE.CLS_PAR 0.4 \
  ACTIVE.CON_PAR 0.2 \
  ACTIVE.KL_PAR 0.3 \
  DCCL.ADAPTATION_LIST data/VISDA-C/validation_proxy25_seed2020_list.txt \
  DCCL.GTR_PAR 0.0 \
  DCCL.CONSISTENCY_STOP_GRAD False \
  DCCL.LOSS_DIAG True
```

Gate relative to P1:

```text
final macro >= 87.98
hard-class mean >= 74.29
cycles 3-4 retain a positive advantage
no compensating car/truck exchange
```

If KL `0.3` remains near `87.8` or fails the class gate, stop global scalar
loss-weight tuning. Do not automatically run KL `0.5`; first archive and
inspect the loss shares and classwise changes.

## Raw artifacts

```text
proxy25_gtr_sweep_raw.txt
proxy25_cls050_con030_gtr005_partial_raw.txt
proxy25_stopgrad_full_raw.txt
proxy25_results.csv
SHA256SUMS
```

The combined-run raw attachment ends at the first Cycle-4 checkpoint; its
remaining checkpoints and final class vector were supplied separately and are
recorded in the completed-run table above. `SHA256SUMS` preserves the hashes
of the three copied raw attachments.
