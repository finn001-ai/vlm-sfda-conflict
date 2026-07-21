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

### Dispatch fix

The first preflight launcher used a method identifier outside the entrypoint's
accepted DCCL prefix. It exited in zero seconds and produced an empty log; no
training was performed and this is not an experimental result. The launcher
now uses a `temporal_precision_head_seed...` identifier, the common dispatcher
accepts all `temporal_precision_head_...` variants, and both launchers refuse
to summarize a log without a `Task: TV` accuracy record.

## Mix-0.4 preflight result

The corrected four-cycle preflight completed normally and failed the full-run
gate:

```text
baseline matched peak = 90.15
mix-0.4 matched peak = 90.16
matched improvement = +0.01
projected full peak = 91.08
required full peak = 91.40
decision = fail_full_training_gate
```

The complete mix-0.4 run must not be launched. Its pseudo-label counts,
accuracies, and checkpoint trajectory are also nearly identical to the
baseline. A global blend coefficient therefore has no useful leverage in this
range; the target head is initialized from the source head and the two blends
do not induce materially different top-1 decisions.

Archived artifacts:

```text
temporal_precision_head_visda_mix040_preflight_accuracy.csv
temporal_precision_head_visda_mix040_preflight_summary.json
temporal_precision_head_visda_mix040_preflight_per_class.csv
temporal_precision_head_visda_mix040_preflight_dynamics.json
temporal_precision_head_visda_mix040_preflight_gate.json
```

Before another training run, analyze the existing eight-cycle baseline NPZs:

```bash
bash tools/run_visda_stage14_classwise_conflict_probe.sh
```

This is a zero-training diagnostic. It tests whether stable temporal
corrections are heterogeneous across predicted classes, while explicitly
retaining the warning that validation labels are used only for mechanism
diagnosis. If class-conditional routing is unsupported, the next training
proposal is the already specified `PL_STABLE_CYCLES=3` plus
`GTR_STABLE_CYCLES=3` variant.

## Classwise conflict result

The eight-cycle diagnostic supports class-conditional routing at the oracle
analysis level. Stable teacher corrections are highly heterogeneous:

```text
global stable teacher minus CLIP = +1.6685 pp; net corrections = +433
predicted classes passing the route gate = car, motorcycle, plant, truck
true car stable teacher minus CLIP = +8.9909 pp; net corrections = +376
true truck stable teacher minus CLIP = -1.7431 pp; net corrections = -70
true person stable teacher minus CLIP = +0.2671 pp; net corrections = +9
```

This result explains why a global target-head blend is ineffective: useful and
harmful conflict flows coexist across classes. The oracle class list cannot be
hard-coded because it uses validation labels. Before implementing another
training method, run a second zero-training gate over predeclared label-free
class proxies:

```bash
bash tools/run_visda_stage14_unlabeled_route_proxy_probe.sh
```

The proxy gate requires a single label-free statistic to have Spearman
correlation at least `0.5` with oracle class gain and top-4 overlap at least
`3/4`. A failed gate rejects class routing and sends the project to the
`PL/GTR_STABLE_CYCLES=3` proposal instead of fitting a proxy combination to
the twelve validation classes.
