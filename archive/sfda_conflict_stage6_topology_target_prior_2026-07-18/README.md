# Stage 6: Graph-Target Distribution Alignment

Date: 2026-07-18

## Decision

Stage 5 `topo_prior` failed because it used the graph posterior mean as an
inverse prior. That branch is stopped. The corrected class-prior formulation is
tested as a separate method:

```text
topo_target_prior
```

For each teacher probability vector:

```text
q(y|x) proportional to p(y|x) * (pi_target(y) / pi_teacher(y))^gamma
```

where:

| Term | Meaning |
|---|---|
| `pi_teacher` | the teacher's own mean predicted class prior |
| `pi_target` | entropy-smoothed graph posterior mean |
| `gamma` | existing `CALIB_POWER`, fixed at `0.5` |

If the graph posterior is unreliable, entropy-adaptive smoothing keeps
`pi_target` close to uniform. This makes `topo_target_prior` a conservative
extension of `both_prior`; setting graph mix to zero recovers the `both_prior`
alignment target.

## Boundary

This is not an ACCD/DUET graph-rule retry:

- no per-sample graph label is used;
- no conflict sample is promoted, rejected, abstained, or transported;
- the graph contributes only a smoothed class-level target prior.

If the no-adaptation gate fails, stop the graph-prior family. Do not tune graph
thresholds, target mix, or loss weights.

## Scientific Gate

Run:

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_topo_target_prior_probe.sh
```

The probe compares `both_prior`, failed inverse `topo_prior`, and corrected
`topo_target_prior` on:

```text
A->C, P->C, R->C
```

Training is justified only if:

```text
decision = pass_training_gate
```

The default gate requires `topo_target_prior` to beat `both_prior` by at least
`0.05` points on at least two of the three tasks and have positive mean delta.
Ground-truth labels are used only for reporting.

## Training Command

If and only if the probe passes:

```bash
bash tools/run_office_home_topo_target_prior_clipart.sh
```

Target comparison:

| Task | both_prior training | DUET paper |
|---|---:|---:|
| A->C | 72.78 | 73.60 |
| P->C | 72.81 | 73.70 |
| R->C | 72.97 | 74.00 |

Do not run all 12 tasks unless at least two target-Clipart tasks improve over
`both_prior` and at least one materially closes the gap to the DUET paper
number.

## Implementation

```text
duet-sfda-main/src/utils/conflict_diffusion.py
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/cfgs/office-home/topo_target_prior.yaml
duet-sfda-main/tools/analyze_topology_target_prior_calibration.py
duet-sfda-main/tools/run_office_home_topo_target_prior_probe.sh
duet-sfda-main/tools/run_office_home_topo_target_prior_clipart.sh
```

## Status

The cloud probe was reported as:

```text
decision = fail_training_gate
```

No training run is justified. The graph-prior family is stopped: do not tune
graph thresholds, target-prior mixing, calibration power, or loss weights.
