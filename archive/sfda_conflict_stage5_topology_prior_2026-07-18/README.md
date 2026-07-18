# Stage 5: Topology-Prior Calibration

Date: 2026-07-18

## Decision

ACCD is stopped as the proposed method. The next graph-related action is not a
new per-sample conflict selector, hard-label rule, teacher-abstention rule, or
candidate-mass transport rule.

The new candidate is:

```text
Topology-prior calibration
```

It uses dual-space diffusion only to estimate a class-level target prior:

```text
agreement anchors -> task/CLIP graph diffusion -> mean graph posterior
```

The mean graph posterior calibrates both source/task and CLIP probability
distributions before the existing DUET/PLMatch pseudo-label and KL path.
Individual graph labels are not used by the training loss.

## Why This Is Not Another Failed ACCD Variant

Stopped ACCD variants made sample-level decisions:

| Stopped family | Action |
|---|---|
| symmetric ACCD | graph posterior/hard label for selected conflicts |
| frozen/reversible memory | state semantics for selected conflicts |
| source rescue | hard source label for graph-to-source conflicts |
| teacher abstention | remove CLIP KL for selected conflicts |
| candidate transport | redistribute source/CLIP candidate mass per selected conflict |
| counterfactual adjudicator | learned source-vs-CLIP conflict selector |

Topology-prior calibration makes no source-vs-CLIP decision for any conflict
sample. The graph is aggregated before it affects training.

## Scientific Gate

Run the no-adaptation probe before training:

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_topo_prior_probe.sh
```

The probe compares raw, `both_prior`, and `topo_prior` mixed top-1 accuracy on
the target-Clipart bottleneck tasks:

```text
A->C, P->C, R->C
```

Training is justified only if the probe reports:

```text
decision = pass_training_gate
```

The default gate requires `topo_prior` to beat `both_prior` by at least `0.05`
points on at least two of the three tasks and have positive mean delta.
Ground-truth labels are used only for reporting this gate.

## Training Command

If and only if the probe passes:

```bash
bash tools/run_office_home_topo_prior_clipart.sh
```

Compare against:

| Task | both_prior | DUET paper |
|---|---:|---:|
| A->C | 72.78 | 73.60 |
| P->C | 72.81 | 73.70 |
| R->C | 72.97 | 74.00 |

Do not expand to all 12 tasks unless at least two target-Clipart tasks improve
over `both_prior`, and at least one materially closes the gap to the published
DUET number.

## Implementation

Core changes:

```text
duet-sfda-main/src/utils/conflict_diffusion.py
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/cfgs/office-home/topo_prior.yaml
```

Probe and run scripts:

```text
duet-sfda-main/tools/analyze_topology_prior_calibration.py
duet-sfda-main/tools/run_office_home_topo_prior_probe.sh
duet-sfda-main/tools/run_office_home_topo_prior_clipart.sh
```

## Status

The no-adaptation probe failed the training gate:

| Task | both_prior mixed top-1 | topo_prior mixed top-1 | Delta |
|---|---:|---:|---:|
| A->C | 61.1226 | 57.2050 | -3.9176 |
| P->C | 61.9244 | 49.5533 | -12.3711 |
| R->C | 61.9244 | 50.7904 | -11.1340 |
| Mean | - | - | -9.1409 |

`topo_prior` had zero passing tasks and is stopped before training. The raw
result is archived as `topology_prior_probe_result.json`.

## Postmortem

The failure is too large for threshold tuning. The graph posterior mean is a
spiky anchor/connectedness estimate:

| Task | graph prior max | graph prior entropy |
|---|---:|---:|
| A->C | 0.062244 | 3.724241 |
| P->C | 0.085621 | 3.633934 |
| R->C | 0.114347 | 3.628017 |

The implemented `topo_prior` divided teacher probabilities by this graph prior.
That is an inverse-prior operation, not target-prior alignment. `both_prior`
works because it divides by each teacher's own predicted prior, implicitly
moving predictions toward a uniform class target. If the graph posterior mean
is interpreted as a target-domain class prior, the aligned formula should be:

```text
q(y|x) proportional to p(y|x) * (pi_target(y) / pi_teacher(y))^gamma
```

Therefore `topo_prior` itself remains a failed branch. A corrected target-prior
alignment may be tested as a separate stage with its own no-training gate.
