---
experiment_id: SFDA-B-260724-001
date: 2026-07-24
dataset: VisDA-C
task: train_to_validation
seed: 2020
comparison: official_released_DUET_vs_DCCL_Stage14
checkpoint: pre_cycle_4_pseudo_label_refresh
status: completed
decision: preserve_DUET_dual_view_and_restrict_DCCL_to_conflicts
---

# VisDA-C Dual-View Precision-Coverage Audit

## Question

Does DCCL retain the dual-perspective pseudo-label mechanism used by the
released DUET path, and what changes after that inherited mechanism?

## Code-path finding

DCCL retains both DUET prediction views:

1. the task/source-model probability;
2. the CLIP image-text probability.

Both methods use agreement between the two branch top-1 predictions and an
average of the two probability vectors. The dual-view mechanism is therefore
an inherited DUET component, not a new DCCL contribution.

The effective VisDA DCCL configuration changes what happens around this common
base:

| Mechanism | Released DUET path | Current DCCL Stage14 |
|---|---|---|
| branch probabilities | raw task and CLIP softmax | `both_prior` calibrates both branches |
| pseudo-label memory | monotonic: once selected, retained | two-cycle stable and reversible |
| task branch | frozen source classifier head | target head blend, mix `0.3`, active from cycle 2 |
| base losses | consistency + pseudo-label CE + CLIP KL | retained |
| graph-temporal loss | absent | conflict-only GTR with weight `0.05` |

Other optional DCCL families are inactive in this run:

```text
CAND_PAR=0
PL_EXPAND=none
PL_CLASS_BALANCE=False
PROTO_ADAPT=False
PAIR_FEATURE_ADAPT=False
COV_TRANSPORT_ADAPT=False
THREE_VIEW_EM=False
ACCD.ENABLED=False
GTF_APPLY_TO=none
```

`GTF_APPLY_TO=none` means that the graph teacher does not replace the main
CLIP KL target. It is still constructed for the separately weighted GTR term.

## Matched pre-cycle-4 evidence

The following values are taken immediately before cycle-4 target training on
the complete 55,388-image target set:

| Metric | Released DUET | DCCL Stage14 | DCCL - DUET |
|---|---:|---:|---:|
| global mixed-output accuracy | 88.94% | 87.98% | -0.96 pp |
| selected pseudo-label count | 53,372 | 47,393 | -5,979 |
| selected mixed-label accuracy | 90.42% | 93.46% | +3.04 pp |
| selected coverage | 96.36% | 85.57% | -10.80 pp |

Coverage is calculated as selected count divided by 55,388. Rounding explains
the displayed `-10.80 pp` difference.

## Interpretation

DCCL does not lack the dual-view pseudo-label mechanism. Its current additions
make the selected pseudo labels cleaner, but much more conservative:

- selected-label precision rises by `3.04 pp`;
- coverage falls by about `10.80 pp`, removing `5,979` training samples;
- the global mixed prediction is simultaneously `0.96 pp` worse.

This precision-coverage tradeoff is consistent with the final full-data result:

```text
released DUET final = 91.50
DCCL Stage14 final  = 91.04
gap                 = -0.46 pp
```

The comparison does not by itself isolate which one of `both_prior`, stable
memory, target-head blending, or GTR causes the loss. Those mechanisms are
coupled in the completed DCCL run. It does establish that the combined DCCL
intervention is not improving the inherited DUET base on VisDA-C.

## Paper and method consequence

Do not describe dual-perspective pseudo-label generation as a DCCL novelty.
It is inherited from the DUET framework. The defensible DCCL contribution must
be conflict identification and correction after the two views disagree.

The next method revision should be baseline-preserving:

1. keep the released DUET branch probabilities, agreement mask, monotonic
   pseudo-label memory, and base losses as the default path;
2. apply DCCL only to samples where the task and CLIP views conflict;
3. require a matched proxy gate before another complete eight-cycle run;
4. report precision, coverage, and final accuracy together, rather than
   optimizing selected-label precision alone.

## Source records

```text
archive/sfda_conflict_visda_full_duet_control_2026-07-24/
  plmatch_visda_full_seed2020_raw.txt

archive/sfda_conflict_visda_stage14_transfer_2026-07-21/
  temporal_precision_head_visda_seed2020_full_run.txt

duet-sfda-main/src/methods/oh/plmatch.py
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/cfgs/visda/temporal_precision_head.yaml
```
