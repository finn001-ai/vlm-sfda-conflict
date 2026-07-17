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
| `DCCL.CALIB_MODE` | Applies class-wise source/CLIP/mixed prior calibration before pseudo-label generation. |
| `DCCL.CALIB_POWER` | Strength of prior calibration. |
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
duet-sfda-main/tools/run_office_home_dccl_calibration_ac.sh
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

Implemented fast version:

```text
source_prior: calibrate source probability distribution by target class prior
clip_prior: calibrate CLIP probability distribution by target class prior
both_prior: calibrate both source and CLIP before agreement/fusion
mix_prior: calibrate only the fused source/CLIP distribution
```

Calibration A->C results:

| Trial | A->C |
|---|---:|
| PLMatch same environment | 72.03 |
| `source_prior` | 72.10 |
| `clip_prior` | 72.78 |
| `both_prior` | 72.74 |
| `mix_prior` | 72.28 |

Interpretation:

```text
Class-wise calibration is the first tested mechanism that clearly improves
A->C over the same-environment PLMatch baseline.
```

It still does not reach the DUET paper number of 73.6 on A->C, so the next
decision should be based on multi-task behavior. Run `clip_prior` first on
A->P and A->R, then expand to all 12 tasks if the Art-source average improves.

Art-source `clip_prior` results:

| Task | PLMatch same env | DUET paper | `clip_prior` |
|---|---:|---:|---:|
| A->C | 72.03 | 73.6 | 72.78 |
| A->P | 90.52 | 90.4 | 90.88 |
| A->R | 90.82 | 91.0 | 91.00 |
| Avg | 84.46 | 85.00 | 84.89 |

This improves all three same-environment PLMatch tasks, but is still slightly
below the DUET paper Art-source average because A->C remains under 73.6. The
next step is to run the remaining nine Office-Home tasks.

Full Office-Home `clip_prior` results:

| Task | DUET paper | `clip_prior` | Delta |
|---|---:|---:|---:|
| A->C | 73.6 | 72.78 | -0.82 |
| A->P | 90.4 | 90.88 | +0.48 |
| A->R | 91.0 | 91.00 | +0.00 |
| C->A | 83.6 | 83.27 | -0.33 |
| C->P | 90.7 | 90.81 | +0.11 |
| C->R | 90.9 | 90.57 | -0.33 |
| P->A | 82.7 | 82.20 | -0.50 |
| P->C | 73.7 | 72.44 | -1.26 |
| P->R | 91.2 | 90.80 | -0.40 |
| R->A | 83.6 | 82.53 | -1.07 |
| R->C | 74.0 | 72.69 | -1.31 |
| R->P | 91.2 | 90.79 | -0.41 |
| Avg | 84.72 | 84.23 | -0.49 |

Conclusion:

```text
clip_prior is a real mechanism-level improvement over same-environment PLMatch
on A-source tasks, but it is not enough as a universal 12-task method.
```

The negative pattern is target/source dependent:

```text
Art source improves.
Product/RealWorld -> Clipart remain weak.
RealWorld source is weak under clip_prior.
```

Next recommended direction:

```text
Source/target-conditioned calibration instead of one global calibration mode.
Use clip_prior where CLIP class prior is the likely bottleneck, but avoid or
change calibration on sources where it suppresses useful source structure.
```

Next implemented trial:

```text
auto_agree calibration selector
```

For each cycle, it evaluates `none`, `source_prior`, `clip_prior`,
`both_prior`, and `mix_prior` using only unsupervised agreement coverage and
agreement confidence, then selects one calibration mode before pseudo-label
generation. The intended test is whether it keeps the A-source gains from
`clip_prior` while avoiding P/R-source degradation.

Art-source `auto_agree` results:

| Task | PLMatch same env | fixed `clip_prior` | `auto_agree` |
|---|---:|---:|---:|
| A->C | 72.03 | 72.78 | 72.46 |
| A->P | 90.52 | 90.88 | 90.74 |
| A->R | 90.82 | 91.00 | 90.91 |
| Avg | 84.46 | 84.89 | 84.70 |

Interpretation:

```text
auto_agree preserves the positive direction on Art-source tasks, but it is
weaker than always using clip_prior. The selector may be too conservative or
may switch away from clip_prior in later cycles.
```

Before changing the selector, inspect the selected calibration mode per cycle
and run the weak-source tasks where fixed clip_prior degraded.

Updated:

```text
duet-sfda-main/tools/extract_final_accuracy.py
```

The extractor now reports DCCL hyperparameters together with final accuracy.

Weak-source `auto_agree` results:

| Task | DUET paper | fixed `clip_prior` | `auto_agree` | Delta vs fixed |
|---|---:|---:|---:|---:|
| P->A | 82.7 | 82.20 | 82.24 | +0.04 |
| P->C | 73.7 | 72.44 | 71.41 | -1.03 |
| P->R | 91.2 | 90.80 | 90.82 | +0.02 |
| R->A | 83.6 | 82.53 | 82.41 | -0.12 |
| R->C | 74.0 | 72.69 | 72.39 | -0.30 |
| R->P | 91.2 | 90.79 | 90.76 | -0.03 |

Interpretation:

```text
auto_agree is not a usable selector in its current form. It does not recover
the weak Product/RealWorld source tasks and it is especially wrong on P->C.
This is useful evidence: agreement coverage/confidence is not sufficiently
aligned with final adaptation accuracy, so the next step should be a task-level
calibration probe instead of further tuning auto_agree.
```

Next implemented probe:

```text
duet-sfda-main/tools/run_office_home_dccl_calibration_weak_probe.sh
```

It runs `none`, `source_prior`, `clip_prior`, `both_prior`, and `mix_prior` on
P->A, P->C, P->R, R->A, R->C, and R->P. The purpose is to identify whether the
weak tasks need no calibration, source-prior calibration, mixed calibration, or
a target-conditioned rule. This is a diagnostic matrix, not the final method.

Weak-source calibration probe results:

| Task | none | source_prior | clip_prior | both_prior | mix_prior | Best |
|---|---:|---:|---:|---:|---:|---|
| P->A | 82.16 | 82.49 | 82.20 | 82.08 | 82.24 | source_prior |
| P->C | 71.04 | 71.32 | 72.19 | 72.78 | 71.89 | both_prior |
| P->R | 90.87 | 90.87 | 90.93 | 90.89 | 90.82 | clip_prior |
| R->A | 82.41 | 82.41 | 82.49 | 82.53 | 82.45 | both_prior |
| R->C | 71.98 | 72.33 | 72.88 | 72.97 | 72.78 | both_prior |
| R->P | 90.43 | 90.58 | 90.79 | 90.97 | 90.47 | both_prior |

Main conclusions:

```text
1. The failed auto_agree selector is not selecting the mode that later gives
   the best accuracy.
2. both_prior is the most consistent weak-source mode: it is best on four of
   six weak tasks and materially improves the difficult target-Clipart tasks.
3. The next unified-method candidate should be dual prior calibration
   (both_prior), not task-wise oracle mode selection.
```

Next implemented run:

```text
duet-sfda-main/tools/run_office_home_dccl_both_prior_all.sh
```

This tests `both_prior` as a single fixed method on all 12 Office-Home tasks.
If it beats fixed `clip_prior` average and approaches the DUET paper average,
it becomes the next main method candidate. If it still falls short, use its
failure pattern to design a source/target-conditioned calibration rule.

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
