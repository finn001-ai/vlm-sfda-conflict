# VisDA-C Full Official-DUET-Path Control Audit

Date: 2026-07-24

## Why this is not merely a PLMatch baseline

The repository entrypoint is named `plmatch`, but `src/methods/oh/plmatch.py`
runs the complete released DUET cycle:

1. dual-perspective source-model/CLIP pseudo-label generation;
2. TMI-based CLIP vision optimization (`train_clip`);
3. target-model training with supervised pseudo-label CE, weak/strong
   consistency, and CLIP-guided KL, called PLMatch in the paper.

Therefore this experiment is a matched run of the official released DUET code
path, not an independent pre-DUET method called PLMatch.

Primary sources:

- Paper: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/6c8dfbbd1ba3e22339e58a336cbed52b-Abstract-Conference.html>
- Official code: <https://github.com/l3umblee/duet-sfda>
- Official code commit audited: `bd2644bf6a115ddb4bb64ec94fb121841c5783de`

## Run contract

```text
method/output name = plmatch_visda_full_seed2020
target samples     = complete 55,388-image validation set
evaluation samples = complete 55,388-image validation set
seed               = 2020
cycles             = 8
epochs per cycle   = 4
checkpoints        = 32
runtime            = 9545.92 seconds
environment        = PyTorch 2.8.0+cu128, CUDA 12.8
```

The custom method name changes only dispatch and the output directory. It
calls the same `PLMATCH.train_target` function.

## Result

```text
final mean per-class accuracy = 91.50
oracle peak accuracy          = 91.52
oracle peak checkpoint        = cycle 8, iteration 1732/3464
paper-reported DUET result    = 91.4
final minus paper             = +0.10 pp
```

The `0.10 pp` difference from the one-decimal paper result is not evidence of
a method modification. The released implementation enables cuDNN benchmark
mode without deterministic algorithms, and the reproduced environment differs
substantially from the paper environment. Source-checkpoint identity and
target-list ordering also remain reproduction variables unless hashed.

Cycle-end trajectory:

| Cycle | Official DUET path | DCCL Stage14 | DUET - DCCL |
|---:|---:|---:|---:|
| 1 | 85.76 | 86.01 | -0.25 |
| 2 | 88.84 | 88.43 | +0.41 |
| 3 | 89.78 | 89.36 | +0.42 |
| 4 | 90.33 | 89.98 | +0.35 |
| 5 | 90.65 | 90.34 | +0.31 |
| 6 | 91.13 | 90.70 | +0.43 |
| 7 | 91.31 | 90.57 | +0.74 |
| 8 | 91.50 | 91.04 | +0.46 |

This full-data control establishes that current DCCL Stage14 is below its
matched released DUET base by `0.46 pp` at the final checkpoint. The proxy
control's `+0.10 pp` PLMatch/DCCL difference understated the full-data gap.

## Dual-view inheritance and precision-coverage finding

DCCL does retain the released DUET dual-perspective pseudo-label mechanism:
one task/source-model view and one CLIP image-text view. Both methods use
branch agreement and an average of the two probability vectors. This inherited
dual-view construction must not be described as a DCCL novelty.

The effective DCCL run changes the common base with `both_prior` calibration,
two-cycle reversible stable memory, a `0.3` target-head blend from cycle 2,
and a `0.05` graph-temporal residual loss on selected conflicts. At the
matched pseudo-label refresh immediately before cycle-4 training:

| Metric | Official DUET path | DCCL Stage14 | DCCL - DUET |
|---|---:|---:|---:|
| global mixed-output accuracy | 88.94% | 87.98% | -0.96 pp |
| selected pseudo-label count | 53,372 | 47,393 | -5,979 |
| selected mixed-label accuracy | 90.42% | 93.46% | +3.04 pp |
| selected coverage | 96.36% | 85.57% | -10.80 pp |

DCCL therefore raises selected-label precision while sharply reducing
coverage, and its global mixed prediction is worse at the same checkpoint.
This is consistent with, but does not alone causally isolate, the final
`91.04` versus `91.50` gap because the DCCL additions are coupled in the run.

The next revision should preserve the official DUET dual-view path and apply
DCCL only to genuinely conflicting samples. The full reasoning, active-option
audit, and source-log locations are recorded in:

```text
dual_view_precision_coverage_audit.md
```

## Class comparison against DCCL Stage14 final

| Class | Official DUET path | DCCL | DUET - DCCL |
|---|---:|---:|---:|
| aeroplane | 98.71 | 98.60 | +0.11 |
| bicycle | 89.38 | 92.86 | -3.48 |
| bus | 88.98 | 89.00 | -0.02 |
| car | 80.61 | 78.47 | +2.14 |
| horse | 98.21 | 97.95 | +0.26 |
| knife | 97.78 | 97.98 | -0.20 |
| motorcycle | 95.93 | 94.03 | +1.90 |
| person | 85.12 | 83.62 | +1.50 |
| plant | 96.00 | 96.53 | -0.53 |
| skateboard | 97.37 | 97.37 | 0.00 |
| train | 95.21 | 94.78 | +0.43 |
| truck | 74.73 | 71.27 | +3.46 |
| car/person/truck mean | 80.15 | 77.79 | +2.37 |
| other-nine mean | 95.29 | 95.46 | -0.17 |
| macro | 91.50 | 91.04 | +0.46 |

Unlike the 25% proxy result, the full official path improves both car and
truck relative to DCCL. The full-data gap is concentrated in the difficult
classes rather than being only a car/truck exchange.

## Code and configuration audit

The local code was compared with the current official repository.

### Effective configuration

After removing the local optional empty `ACTIVE.ADAPTATION_LIST` field, the
local `cfgs/visda/plmatch.yaml` is structurally identical to the official
file:

```yaml
MODEL:
  ARCH: resnet101
SETTING:
  SEED: 2020
OPTIM:
  LR: 0.001
  MOMENTUM: 0.9
  WD: 0.001
TEST:
  BATCH_SIZE: 64
  MAX_EPOCH: 4
ACTIVE:
  CYCLE: 8
  CLS_PAR: 0.4
  CON_PAR: 0.2
  KL_PAR: 0.4
  ARCH: ViT-B/32
  FINE_LR: 1e-7
  Q_VALUE: 1.05
  BETA: 0.99
```

All printed `DCCL.*` fields are global configuration entries and are never
read by `src/methods/oh/plmatch.py`.

### Local PLMatch changes

Only two local compatibility changes affect the PLMatch source file and
dispatch:

1. method identifiers beginning with `plmatch_` dispatch to the same
   `PLMATCH.train_target`, allowing an isolated output directory;
2. an optional adaptation-list loader supports proxy training.

For this full run, `ACTIVE.ADAPTATION_LIST` is empty. Both
`cfg.t_dset_path` and `cfg.test_dset_path` resolve to the same
`validation_list.txt`, so changing `test_aug` from `txt_test` to `txt_tar` is
behaviorally identical. No loss, pseudo-label, CLIP-update, optimizer, model,
augmentation, or evaluation code differs from the official release.

### Paper-versus-official-code discrepancy

The paper's Appendix Table 4 states VisDA-C adaptation-index momentum
`mu=0.999`. The official released VisDA YAML and `conf.py` both set
`ACTIVE.BETA=0.99`; this value is passed directly as the momentum in
`tsallis_mutual_info`. The reproduced run uses `0.99`.

This discrepancy exists between the authors' paper and their released code; it
was not introduced by DCCL. Without author clarification, the defensible
description is:

```text
reproduced with the official released configuration (BETA=0.99)
```

Do not claim strict identity to every appendix value.

## Remaining provenance check

The terminal log proves the source checkpoint directory but not the checkpoint
bytes. Preserve:

```bash
sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt
sha256sum data/VISDA-C/validation_list.txt
```

These hashes are required to determine whether the source weights and list
ordering exactly match later runs or author-provided artifacts.

## Archived artifacts

```text
plmatch_visda_full_seed2020_raw.txt
plmatch_visda_full_seed2020_summary.json
dual_view_precision_coverage_audit.md
SHA256SUMS
```

The source attachment SHA-256 before whitespace normalization is:

```text
3641f6f121a3b361ad6e101464c9eb637d4278b85b917a3faeb5204ea6b834ed
```
