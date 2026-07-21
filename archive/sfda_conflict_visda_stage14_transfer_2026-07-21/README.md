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

## Unlabeled route proxy result

Only `graph_intervention_rate` passes the predeclared proxy gate:

```text
Spearman versus oracle class gain = 0.580420
top-4 proxy classes = car, plant, motorcycle, horse
oracle top-4 overlap = 3/4
```

The signal is dominated by car (`12.39%` graph intervention versus roughly
`0.6%-1.6%` for the remaining classes), which is also the largest source of
oracle net corrections. The proxy misses truck and includes horse, so this is
evidence for a compute-gated preflight, not evidence for a full run.

The proposed class router is disabled by default and does not alter existing
Office-Home or VisDA configs. When explicitly enabled, it redistributes the
existing GTR sample weights according to per-predicted-class graph
intervention rates and renormalizes them so the total active GTR weight is
unchanged. It does not hard-code class identities or increase the global GTR
loss coefficient.

Run the four-cycle preflight:

```bash
bash tools/run_visda_temporal_precision_head_class_route_preflight.sh
```

Only after its gate reports `pass_full_training_gate` may the eight-cycle
script be launched:

```bash
bash tools/run_visda_temporal_precision_head_class_route_seed2020.sh
```

This remains a VisDA mechanism preflight under the Stage14 transfer archive.
It becomes a new project stage only if the preflight succeeds.

## Class intervention routing preflight result

The four-cycle run completed all 16 checkpoints, activated class routing, and
failed the predeclared full-training gate:

```text
baseline matched oracle peak = 90.15
class-route matched oracle peak = 90.13
matched improvement = -0.02
projected full oracle peak = 91.05
required full oracle peak = 91.40
decision = fail_full_training_gate
```

The final mean per-class accuracy was `89.98`; the oracle peak was `90.13` at
cycle 4, iteration 1732/3464. The temporal mechanism diagnostic still passed:
stable decisions covered `93.17%` of initial conflicts and improved over final
CLIP by 567 net correct samples. This is useful diagnostic evidence but did not
convert into end-to-end accuracy. The full eight-cycle class-routing job must
not be launched, and the class-routing variant is closed.

The archived gate's `next` string incorrectly says "mix-0.4 job" because the
shared gate summarizer retained its older default text. This wording defect
does not affect the recorded configuration checks, metrics, or
`fail_full_training_gate` decision. The generator has been corrected for
future class-routing runs.

Archived artifacts:

```text
temporal_precision_head_visda_class_route_preflight_accuracy.csv
temporal_precision_head_visda_class_route_preflight_summary.json
temporal_precision_head_visda_class_route_preflight_per_class.csv
temporal_precision_head_visda_class_route_preflight_dynamics.json
temporal_precision_head_visda_class_route_preflight_gate.json
```

## Next predeclared Stage14 temporal-stability preflight

The next experiment tests whether requiring three consecutive stable cycles
improves the precision/coverage tradeoff of both the pseudo-label memory and
the graph-temporal residual. This changes only:

```text
PL_STABLE_CYCLES: 2 -> 3
GTR_STABLE_CYCLES: 2 -> 3
```

It remains a Stage14 internal VisDA preflight. Because a three-cycle memory is
not meaningfully expressed in the earliest cycles, the preflight runs five
cycles and evaluates the later matched window, cycles 4-5. Run only:

```bash
bash tools/run_visda_temporal_precision_head_stability3_preflight.sh
```

The full eight-cycle script is present but gate-protected:

```bash
bash tools/run_visda_temporal_precision_head_stability3_seed2020.sh
```

The full run is allowed only if every predeclared check passes:

1. The baseline uses `PL/GTR_STABLE_CYCLES=2`, the candidate uses `3`, both
   target-head mixes are `0.3`, and the candidate completes five cycles.
2. The temporal dynamics file reports `pass_training_gate`.
3. The candidate cycles 4-5 oracle peak improves over the matched baseline
   cycles 4-5 oracle peak by at least `0.20` percentage points.
4. Adding the baseline post-cycle-5 late gain to the candidate matched oracle
   peak projects to at least `91.40`.

The gate is deliberately conservative and oracle-labeled: all accuracy peaks
use VisDA validation labels and must be reported as oracle peaks. If it fails,
do not run the full stability-3 job. Archive the result and compare the
pseudo-label precision/coverage trajectory with graph-temporal correction
coverage before deciding whether a one-axis `PL=3,GTR=2` or `PL=2,GTR=3`
diagnostic is mechanism-supported; do not launch either automatically.
