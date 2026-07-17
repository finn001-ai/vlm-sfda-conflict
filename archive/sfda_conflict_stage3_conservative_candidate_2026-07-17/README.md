# Stage 3: Conservative Conflict Candidate Learning

Date: 2026-07-17

## Current Goal

The immediate experimental goal is not only to show that conflict samples exist,
but to test whether conservative conflict-candidate learning can improve over
DUET/PLMatch.

Target numbers reported for DUET on Office-Home:

| Task | DUET |
|---|---:|
| A->C | 73.6 |
| A->P | 90.4 |
| A->R | 91.0 |
| C->A | 83.6 |
| C->P | 90.7 |
| C->R | 90.9 |
| P->A | 82.7 |
| P->C | 73.7 |
| P->R | 91.2 |
| R->A | 83.6 |
| R->C | 74.0 |
| R->P | 91.2 |
| Avg | 84.7 |

## Observed Results So Far

| Task | PLMatch local/cloud | DCCL default | DCCL conservative |
|---|---:|---:|---:|
| A->C | 71.84 | 71.34 | 71.87 |
| A->P | unknown | 90.61 | 90.67 |
| A->R | unknown | 90.36 | 90.87 |

DCCL default means:

```text
CAND_PAR=0.05
PROMOTE_K=2
```

DCCL conservative means:

```text
CAND_PAR=0.01
PROMOTE_K=999
```

The current evidence does not support aggressive promotion of conflict samples
into hard pseudo-labels. Conservative candidate-set learning is more plausible:
conflict samples are used as soft candidate supervision, not converted into a
single hard label.

## Code Changes In This Stage

Added candidate reliability controls:

| Option | Meaning |
|---|---|
| `DCCL.CAND_START_CYCLE` | Starts conflict candidate learning only after a warm-up period. |
| `DCCL.CAND_TAU` | Only apply candidate loss when model probability mass on `{source_pred, clip_pred}` is at least this threshold. |
| `DCCL.CAND_WEIGHT` | Candidate loss weighting mode: `none`, `mass`, or `ramp`. |
| `DCCL.KL_MODE` | Controls how conflict samples participate in the main CLIP KL teacher. |
| `DCCL.KL_CANDIDATE` | Builds a conflict candidate teacher with `confidence` or `balanced` candidate weights. |
| `DCCL.PL_EXPAND` | Expands hard pseudo-label training beyond source/CLIP agreement. |
| `DCCL.PL_TOPK_PER_CLASS` | Minimum class-balanced top-k pseudo-labels per predicted class. |
| `DCCL.PL_MIN_CONF` | Confidence floor for class-balanced pseudo-label expansion. |

Added run scripts:

```text
duet-sfda-main/tools/run_office_home_plmatch_all.sh
duet-sfda-main/tools/run_office_home_dccl_conservative_smoke.sh
duet-sfda-main/tools/run_office_home_dccl_ac_sweep.sh
duet-sfda-main/tools/run_office_home_dccl_conservative_all.sh
duet-sfda-main/tools/run_office_home_dccl_scheme_trial_ac.sh
duet-sfda-main/tools/run_office_home_dccl_curriculum_ac.sh
duet-sfda-main/tools/run_office_home_dccl_balanced_pl_ac.sh
```

## 2026-07-17 Mid-run Correction

The A->C parameter sweep was stopped because it was becoming hyperparameter
tuning instead of method search.

Partial A->C results:

| Scheme | A->C |
|---|---:|
| PLMatch same environment | 72.03 |
| DCCL candidate loss, `CAND_PAR=0.005` | 71.87 |
| DCCL candidate loss, `CAND_PAR=0.01` | 71.98 |
| DCCL candidate loss, `CAND_PAR=0.02` | 71.71 |
| DCCL candidate gate, `CAND_TAU=0.3` | 72.03 |

Conclusion: adding or lightly gating candidate loss is not enough.

The next fast trials change the supervision mechanism:

| Trial | Meaning |
|---|---|
| `conflict_kl_off` | Remove CLIP KL supervision on conflict samples. |
| `conflict_candidate_kl` | Replace conflict CLIP KL with a `{source_pred, clip_pred}` candidate teacher. |
| `candidate_kl_plus_loss` | Candidate teacher plus low-weight candidate-set loss. |
| `candidate_kl_balanced` | Candidate teacher with balanced source/CLIP weights. |

Observed mechanism result:

| Trial | A->C |
|---|---:|
| `conflict_kl_off` | 67.31 |

Conclusion: CLIP KL on conflict samples is still necessary. The next scheme is
therefore not to remove CLIP supervision, but to delay conflict-candidate
learning until the target representation has stabilized.

Late curriculum did not improve A->C:

| Trial | A->C |
|---|---:|
| `CAND_START_CYCLE=2, CAND_PAR=0.01` | 72.00 |
| `CAND_START_CYCLE=1, CAND_PAR=0.01` | 71.71 |

The next larger scheme is class-balanced pseudo-label expansion. This changes
the hard pseudo-label admission rule itself instead of adding a loss to samples
that remain outside the main pseudo-label set.

## A->C Rapid Trial Summary

Same-environment PLMatch baseline:

| Method | A->C |
|---|---:|
| PLMatch | 72.03 |

Tested DCCL/candidate variants:

| Trial | A->C | Interpretation |
|---|---:|---|
| Candidate loss, `CAND_PAR=0.005` | 71.87 | No improvement. |
| Candidate loss, `CAND_PAR=0.01` | 71.98 | No improvement. |
| Candidate loss, `CAND_PAR=0.02` | 71.71 | Stronger candidate loss hurts. |
| Candidate gate, `CAND_TAU=0.3` | 72.03 | Matches baseline only. |
| Remove conflict KL | 67.31 | CLIP KL on conflicts is necessary. |
| Replace conflict KL with candidate KL | incomplete, poor early curve | Too destructive to the main teacher. |
| Late candidate, start cycle 2 | 72.00 | No improvement. |
| Late candidate, start cycle 1 | 71.71 | Hurts. |
| Balanced pseudo-label top30 | 71.71 | Expanding hard labels adds noise. |
| Balanced pseudo-label top45 | 71.96 | Still below baseline. |
| Balanced pseudo-label top45, min conf 0.2 | 71.71 | Still below baseline. |

Current conclusion:

```text
The simple conflict-candidate family is not enough for A->C.
The problem is not solved by:
1. adding candidate loss,
2. delaying candidate loss,
3. removing conflict CLIP KL,
4. replacing conflict CLIP KL,
5. broadening hard pseudo-label coverage by class-balanced top-k.
```

This is a useful negative result. It means the next step should not be another
small DCCL variant. The next method-level change should address teacher/source
calibration or representation alignment more directly.

Recommended next trial:

```text
Class-wise teacher calibration before fusion:
estimate per-class reliability of source and CLIP on agreement/high-confidence
target samples, then reweight source/CLIP logits before pseudo-label generation.
```

This changes the actual pseudo-label distribution before both hard-label
selection and KL supervision, rather than adding losses after labels are chosen.

Updated:

```text
duet-sfda-main/tools/extract_final_accuracy.py
```

The extractor now reports DCCL hyperparameters together with final accuracy.

## Next Cloud Order

First confirm baseline on the missing Art-source tasks:

```bash
bash tools/run_office_home_plmatch_smoke.sh
python tools/extract_final_accuracy.py --glob 'output/uda/office-home/*/*/*.txt' > output/uda/office-home/stage3_accuracy.csv
```

Then run A->C sweep:

```bash
bash tools/run_office_home_dccl_ac_sweep.sh
python tools/extract_final_accuracy.py --glob 'output/uda/office-home/*/*/*.txt' > output/uda/office-home/stage3_accuracy.csv
```

Decision rule:

1. If any A->C setting approaches or exceeds 73.6, use that setting for all 12 tasks.
2. If A->C remains around 71.8-72.0, do not claim A->C improvement; test whether the method improves average accuracy across 12 tasks.
3. If the average does not improve over DUET/PLMatch, the current method is not enough and needs a stronger conflict reliability model.

Run all 12 for the best conservative setting:

```bash
bash tools/run_office_home_dccl_conservative_all.sh 0.01 0.0 none 0.4
```

or replace arguments with the best A->C sweep result:

```bash
bash tools/run_office_home_dccl_conservative_all.sh CAND_PAR CAND_TAU CAND_WEIGHT TAU_LOW
```

## Paper Framing Update

Avoid saying the method is simply "DUET + loss".

The stronger claim should be:

> Existing VLM-guided SFDA methods treat source/VLM disagreement mainly as
> uncertainty. Our evidence suggests that many disagreement samples contain
> useful target-domain signal, but hard conflict resolution is unreliable.
> Conservative candidate-set supervision gives these samples a training role
> without forcing a premature single-label decision.

This framing is still only valid if the full 12-task experiments support it.
