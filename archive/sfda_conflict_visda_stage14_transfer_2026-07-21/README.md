# Stage14 Transfer to VisDA-C

Date: 2026-07-21

## Scope

This is a dataset transfer of the frozen Office-Home Stage14 mechanism, not a
new method stage. VisDA-C has one adaptation task: synthetic `train` to real
`validation`, with 12 classes and mean per-class accuracy as the primary
metric.

The conflict mechanism is unchanged: both-prior calibration, reversible
two-cycle temporal precision memory, the `0.3` blend target head, graph
teacher diagnostics with `GTF_APPLY_TO=none`, and the `0.05` graph-temporal
residual. Stage15-22 variants remain disabled.

Dataset-specific settings follow the repository's existing VisDA-C PLMatch
configuration:

```text
backbone = ResNet-101
optimizer LR = 0.001
cycles = 8
epochs per cycle = 4
GENT_PAR = 0.1
CLIP visual LR = 1e-7
Q_VALUE = 1.05
```

## Inputs

Required metadata and image lists:

```text
data/VISDA-C/train_list.txt
data/VISDA-C/validation_list.txt
data/VISDA-C/classname.txt
```

Required source checkpoints:

```text
source/uda/VISDA-C/T/source_F.pt
source/uda/VISDA-C/T/source_B.pt
source/uda/VISDA-C/T/source_C.pt
```

If source checkpoints are missing:

```bash
bash tools/train_visda_source.sh
```

Run Stage14 transfer:

```bash
bash tools/run_visda_temporal_precision_head.sh
```

## Outputs

```text
output/uda/VISDA-C/temporal_precision_head_visda_seed2020_accuracy.csv
output/uda/VISDA-C/temporal_precision_head_visda_seed2020_summary.json
output/uda/VISDA-C/temporal_precision_head_visda_seed2020_per_class.csv
output/uda/VISDA-C/temporal_precision_head_visda_seed2020_dynamics.json
output/uda/VISDA-C/temporal_precision_head_visda_seed2020_source_sha256.txt
```

Both final and oracle-peak mean per-class accuracy are retained. Peak uses the
validation labels and is diagnostic only. The first run has no pass/fail
claim; it must be compared with VisDA baselines using the same source weights
before seeds 2021 and 2022 are run.

## Status

```text
Stage14 seed-2020 run complete
final mean per-class accuracy = 91.04
oracle peak mean per-class accuracy = 91.07 (cycle 8, iter 1732/3464)
delta versus 91.4 reference = -0.33
temporal conflict training gate = pass
```

## Seed-2020 result

The run finished all 32 checkpoints. The peak is only `0.03` above the final
checkpoint, so this is not a late-training collapse. Peak accuracy progressed
from `90.15` at cycle 4 to `90.36`, `90.74`, `90.99`, and `91.07` through
cycles 5-8. The final increment was only `0.08`, making a cycle-only extension
unlikely to recover the full `0.33` reference gap efficiently.

The temporal mechanism remains supported: stable conflict decisions cover
`94.49%` of initial conflicts at `85.59%` accuracy, with 658 corrections and
225 degradations relative to final CLIP (`+433` net correct). The remaining
mean-class bottlenecks are truck (`71.77` peak), car (`78.18`), and person
(`82.80`).

Archived artifacts:

```text
temporal_precision_head_visda_seed2020_accuracy.csv
temporal_precision_head_visda_seed2020_summary.json
temporal_precision_head_visda_seed2020_per_class.csv
temporal_precision_head_visda_seed2020_dynamics.json
temporal_precision_head_visda_seed2020_full_run.txt
```

## Compute-gated next experiment

The first VisDA-specific Stage14 adjustment tests target-head blend `0.4`
instead of `0.3`. It uses four cycles and compares against the matched first
four cycles of the completed baseline. The full eight-cycle run is allowed
only when both conditions hold:

```text
matched peak improvement >= 0.25
candidate four-cycle peak + baseline late gain >= 91.4
```

Run:

```bash
bash tools/run_visda_temporal_precision_head_mix040_preflight.sh
```

Only after `temporal_precision_head_visda_mix040_preflight_gate.json` reports
`pass_full_training_gate`:

```bash
bash tools/run_visda_temporal_precision_head_mix040_seed2020.sh
```

If the preflight fails, do not run the full mix-0.4 job. The next distinct
proposal is a temporal-precision variant with `PL_STABLE_CYCLES=3` and
`GTR_STABLE_CYCLES=3`, evaluated with a later-cycle gate because its effect is
not observable reliably in the first two cycles.
