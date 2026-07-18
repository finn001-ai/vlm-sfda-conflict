# Stage 7: Temporal Conflict Dynamics Probe

Date: 2026-07-18

## Decision

Prompt/template engineering is not pursued because it changes the CLIP teacher
rather than studying conflict samples. Graph-prior calibration also failed its
gate and is stopped.

The next probe returns to the core conflict-sample question:

```text
Do source/CLIP conflict samples become more reliable when judged by adaptation
dynamics rather than static confidence, prototypes, neighbors, or graphs?
```

This stage does not introduce a new loss. It reruns the current `both_prior`
path and exports per-cycle predictions for conflict samples.

## What Is Tested

For the initial source/CLIP conflict set, the analyzer measures whether the
final prediction that remains stable across the last two adaptation cycles:

- covers a non-trivial fraction of conflicts;
- beats final-cycle CLIP on the same selected samples;
- produces positive net corrections over final-cycle CLIP;
- passes a one-sided paired correction test when SciPy is available.

Ground-truth labels are used only by the analyzer, not during training.

## Cloud Command

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_conflict_probe.sh
```

The script runs:

```text
A->C, P->C, R->C
```

and writes:

```text
output/uda/office-home/temporal_conflict_dynamics_probe.json
```

## Training Gate

Do not implement a temporal loss unless:

```text
decision = pass_training_gate
```

The default gate requires at least two target-Clipart tasks to pass. A task
passes only if stable temporal predictions cover at least 5% of initial
conflicts and beat final CLIP on the same selected subset with positive net
corrections.

## Boundary

This is not:

- a prompt/template change;
- an ACCD graph-rule variant;
- a static source-vs-CLIP hard selector;
- another candidate-loss or KL-weight trial.

It is a falsification probe for whether adaptation dynamics contain an
independent conflict reliability signal. If this fails, the next method should
not be built from sample-level conflict resolution.

## Implementation

```text
duet-sfda-main/cfgs/office-home/temporal_probe.yaml
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/tools/analyze_temporal_conflict_dynamics.py
duet-sfda-main/tools/run_office_home_temporal_conflict_probe.sh
```

## Status

The temporal dynamics probe passed on A->C, P->C, and R->C. The raw result is
archived as `temporal_conflict_dynamics_probe_result.json`.

| Task | Stable conflict coverage | Stable teacher acc. | Stable CLIP acc. | Net gain | p-value |
|---|---:|---:|---:|---:|---:|
| A->C | 87.3137 | 65.4982 | 64.2066 | +28 | 0.000881 |
| P->C | 87.6539 | 66.4681 | 65.2766 | +28 | 0.001873 |
| R->C | 86.5833 | 65.5919 | 64.1001 | +31 | 0.000225 |

This validates adaptation dynamics as a conflict reliability signal, but the
gain over final CLIP is small. It should be combined with a soft teacher
improvement rather than converted directly into hard labels.
