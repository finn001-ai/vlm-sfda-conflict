# VisDA-C Structural Ablation Result

Date: 2026-07-24

## Scope

This archive closes the 25% proxy ablation of stable/reversible pseudo-label
memory and the adapted target head. The fixed contract was:

```text
adaptation samples = 13,847
evaluation samples = 55,388
cycles = 4
seed = 2020
CALIB_MODE = both_prior
CALIB_POWER = 0.5
GTR_PAR = 0
CLS/CON/KL = 0.4/0.2/0.4
```

The matched official-DUET proxy control finishes at `87.93`. The archived V0
reference (`stable memory + target head`, GTR=0) finishes at `87.83`.

## Result

| Variant | Memory | Target head | Final | Delta vs DUET | Hard mean delta | Other-9 delta | Decision |
|---|---|---|---:|---:|---:|---:|---|
| Official DUET | monotonic | no | 87.93 | 0.00 | 0.00 | 0.00 | reference |
| V0 archived | stable | yes | 87.83 | -0.10 | -1.9867 | +0.5222 | fail |
| V1 | monotonic | yes | 88.03 | +0.10 | -1.6700 | +0.6778 | fail |
| V2 | stable | no | 88.01 | +0.08 | -0.6867 | +0.3267 | fail |
| V3 | monotonic | no | 88.18 | +0.25 | -0.3633 | +0.4522 | fail |

All candidate final checkpoints equal their oracle peaks. The cycle-end
trajectories were:

| Method | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---:|---:|---:|---:|
| Official DUET | 81.67 | 85.55 | 87.15 | 87.93 |
| V1 | 81.73 | 85.96 | 87.13 | 88.03 |
| V2 | 82.05 | 85.10 | 87.18 | 88.01 |
| V3 | 82.07 | 85.77 | 87.36 | 88.18 |

V3 passes the macro-improvement threshold but fails both difficult-class
checks:

```text
car delta   = -4.15 pp
person delta = -1.18 pp
truck delta = +4.24 pp
hard-class mean delta = -0.3633 pp
```

Its `+0.25 pp` macro improvement therefore comes from the other nine classes
and a car-to-truck boundary redistribution. It does not demonstrate better
discrimination on the difficult classes.

The factorial final-accuracy effects are:

```text
monotonic minus stable memory, head on      = +0.20 pp
monotonic minus stable memory, head off     = +0.17 pp
target head effect, monotonic memory        = -0.15 pp
target head effect, stable memory           = -0.18 pp
```

Both stable/reversible admission and the adapted target head are harmful on
this proxy. Removing both gives the best candidate, but it leaves only global
`both_prior` calibration as the meaningful change from the released DUET
path. That calibration exchanges car against truck rather than adding a new
reliable conflict signal.

## Precision-Coverage Finding

The corrected cycle-4 diagnostics are:

| Method | Coverage | Selected mixed precision | Global mixed accuracy |
|---|---:|---:|---:|
| Official DUET | 94.8364 | 90.36 | 88.29 |
| V1 | 94.4248 | 89.70 | 87.65 |
| V2 | 81.8444 | 94.15 | 87.72 |
| V3 | 94.6342 | 89.87 | 87.72 |

Stable memory again raises precision by discarding supervision, while
monotonic memory restores coverage. None of the structural candidates raises
the global mixed prediction above DUET.

All three temporal dynamics JSON files report `pass_training_gate` and
positive net graph-teacher corrections (`+136`, `+126`, and `+141`). Their
end-to-end accuracy gates still fail. This is additional evidence that the
existing temporal/graph diagnostic signal does not reliably convert into
training gains.

## Parser Correction

The original generated gate stored the first DUET log accuracy (`89.44%`) as
`pseudo_label_precision`. In the released PLMatch log this first value is the
selected source/task-label accuracy; the comparable selected mixed-label
accuracy is the following `Mixed output with valid mask` value, `90.36%`.

The summarizer now records both fields separately. This correction changes no
training accuracy, class result, threshold check, or gate decision. The
corrected outputs in this archive are authoritative for precision/coverage
reporting.

## Decision

```text
decision = fail_proxy_gate
passing_variant = none
do not run any full-data structural variant
do not run an eight-cycle job
retain the released DUET path as the VisDA safety baseline
```

The stable-memory/target-head transfer family is closed on VisDA. A subsequent
method must preserve the released DUET path and introduce genuinely
independent information for conflict resolution. Recombining the existing
task, CLIP, calibration, and graph signals is not justified by these results.

## Archived Artifacts

```text
visda_structural_ablation_proxy25_gate_corrected.json
visda_structural_ablation_proxy25_results_corrected.csv
visda_structural_ablation_proxy25_results_corrected_per_class.csv
raw_user_artifacts.tar.gz
```

`raw_user_artifacts.tar.gz` contains all 18 files supplied by the user:
three complete logs plus the original generated gate, tables, summaries,
per-class files, accuracy trajectories, and dynamics JSON files.
