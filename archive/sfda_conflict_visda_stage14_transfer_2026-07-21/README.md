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
implementation complete
local validation passed (93 tests)
cloud seed-2020 run pending
```
