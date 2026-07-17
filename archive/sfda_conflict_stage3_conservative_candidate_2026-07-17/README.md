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
| `DCCL.CAND_TAU` | Only apply candidate loss when model probability mass on `{source_pred, clip_pred}` is at least this threshold. |
| `DCCL.CAND_WEIGHT` | Candidate loss weighting mode: `none`, `mass`, or `ramp`. |

Added run scripts:

```text
duet-sfda-main/tools/run_office_home_plmatch_all.sh
duet-sfda-main/tools/run_office_home_dccl_conservative_smoke.sh
duet-sfda-main/tools/run_office_home_dccl_ac_sweep.sh
duet-sfda-main/tools/run_office_home_dccl_conservative_all.sh
```

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
