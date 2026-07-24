# VisDA-C Structural Ablation: Stable Memory × Target Head

Date: 2026-07-24

## Purpose

The matched full-data audit found that Stage14 finishes at `91.04`, below the
released DUET path at `91.50`. At the pre-cycle-4 refresh, Stage14 raises
selected-label precision from `90.42%` to `93.46%` but lowers coverage from
`96.36%` to `85.57%`. The accuracy gap begins when stable/reversible memory
and the adapted target head both become active in cycle 2.

This job isolates those two structural factors. It is a VisDA transfer
diagnostic under Stage14, not a new method stage and not a parameter sweep.
GTR is fixed to zero because the completed `0/0.05/0.10` audit found only a
car/truck redistribution.

## Fixed Contract

```text
adaptation set = deterministic class-proportional 25% proxy, seed 2020
adaptation samples = 13,847
evaluation samples = complete 55,388-image validation set
cycles = 4
checkpoints per run = 16
CALIB_MODE = both_prior
CALIB_POWER = 0.5
GTR_PAR = 0
CLS/CON/KL = 0.4/0.2/0.4
```

The existing matched official-DUET proxy control is required:

```text
method = plmatch_visda_proxy25_seed2020
final macro = 87.93
hard mean (car/person/truck) = 75.78
other-nine mean = 91.99
```

If it is missing, run:

```bash
bash tools/run_visda_plmatch_proxy25_control.sh
```

## Structural Matrix

The archived V0 result is `87.83` and is retained as a fixed reference. The
new cloud job runs V1-V3:

| Variant | PL memory | Target head | Run |
|---|---|---|---|
| V0 | stable/reversible | enabled | archived reference |
| V1 | monotonic | enabled | new |
| V2 | stable/reversible | disabled | new |
| V3 | monotonic | disabled | new |

## Cloud Command

```bash
cd /hyperai/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_visda_structural_ablation_proxy25.sh
```

The script can resume only from complete 16-checkpoint variant logs. It
refuses to reuse an incomplete or ambiguously duplicated output.

## Predeclared Gate

A variant passes only when every condition holds against the matched official
DUET proxy control:

```text
final macro improvement >= +0.15 pp
car/person/truck mean delta >= 0
other-nine mean delta >= -0.10 pp
no individual hard class delta < -0.50 pp
```

The final checkpoint is primary. Oracle peak is retained as a diagnostic and
must not be reported as label-free model selection. Precision, coverage, and
global mixed-output accuracy are reported jointly but are not independently
optimized.

Do not launch a complete-data experiment unless the unified JSON reports:

```text
decision = pass_proxy_gate
```

Even after a proxy pass, the next allowed job is one matched full-data
four-cycle preflight for the winning structure, not an eight-cycle run.

## Outputs To Return

```text
output/uda/VISDA-C/visda_structural_ablation_proxy25_gate.json
output/uda/VISDA-C/visda_structural_ablation_proxy25_results.csv
output/uda/VISDA-C/visda_structural_ablation_proxy25_results_per_class.csv
output/uda/VISDA-C/visda_structural_v{1,2,3}_*_summary.json
output/uda/VISDA-C/visda_structural_v{1,2,3}_*_per_class.csv
output/uda/VISDA-C/visda_structural_v{1,2,3}_*_dynamics.json
output/uda/VISDA-C/TV/temporal_precision_head_visda_proxy25_v*/*.txt
```

Also preserve the source-checkpoint and proxy-list SHA-256 files written by
the launcher.

## Result

The three cloud runs completed all 16 checkpoints and the unified gate failed:

| Variant | Final | Delta vs DUET | Hard mean delta | Other-9 delta |
|---|---:|---:|---:|---:|
| V1 monotonic + head | 88.03 | +0.10 | -1.6700 | +0.6778 |
| V2 stable + no head | 88.01 | +0.08 | -0.6867 | +0.3267 |
| V3 monotonic + no head | 88.18 | +0.25 | -0.3633 | +0.4522 |

V3 ranks first but loses `4.15 pp` on car and `1.18 pp` on person while
gaining `4.24 pp` on truck. It fails the hard-class-mean and individual-class
compensation gates. No full-data structural variant is allowed.

The result and raw bundle are archived in:

```text
archive/sfda_conflict_visda_structural_ablation_2026-07-24/
```
