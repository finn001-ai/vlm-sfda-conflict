# Stage 17: Stage14 Cycle-4 Trajectory Ensemble

Date: 2026-07-19

## Starting Evidence

Stage14 was re-evaluated with the project-selected peak-accuracy protocol:

| Seed | Stage14 peak mean | Delta vs DUET |
|---|---:|---:|
| 2020 | 84.9000 | +0.1833 |
| 2021 | 84.7692 | +0.0525 |
| 2022 | 84.6783 | -0.0383 |
| Mean | 84.7825 | +0.0658 |

The seed-mean standard deviation is 0.1114, above the 0.10 gate. Across the 36
task runs, 23 final checkpoints are below an earlier peak. Peak locations are
concentrated at 50% and 75% of cycle 4.

Stage14 is therefore retained as the base method. The remaining problem is
late trajectory oscillation, especially for seed 2022, rather than missing
conflict information or insufficient pseudo-label coverage.

## Method

Training, pseudo labels, losses, prompts, graph residual, target-head mix, and
all Stage14 coefficients remain unchanged.

During cycle 4, capture complete model snapshots at fixed evaluation
intervals:

```text
interval 2/4 = 50%
interval 3/4 = 75%
interval 4/4 = 100%
```

Each snapshot contains:

```text
netF + netB + target_head
```

Snapshots are cloned through CPU `state_dict` tensors, not `deepcopy`, so the
weight-normalized target head is supported.

Evaluate two fixed trajectory ensembles:

```text
at 75%: mean logits of snapshots {50%, 75%}
at 100%: mean logits of snapshots {50%, 75%, 100%}
```

The Stage17 primary result is the higher accuracy of these two trajectory
ensemble records, following the project-selected peak protocol. The ordinary
online Stage14 checkpoints from the same run are extracted separately as a
matched control.

This is a temporal model-trajectory ensemble, not EMA feedback, a new loss, or
a per-sample graph decision rule.

## Config

```text
MODEL.METHOD = temporal_precision_head_trajectory
DCCL.TARGET_HEAD_ADAPT = True
DCCL.TARGET_HEAD_VARIANT = blend
DCCL.TARGET_HEAD_MIX = 0.3
DCCL.TARGET_HEAD_EMA = False
DCCL.TRAJECTORY_ENSEMBLE = True
DCCL.TRAJECTORY_SNAPSHOT_INTERVALS = [2, 3, 4]
```

## Step 1: Seed-2022 Gate

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_head_trajectory_seed2022.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_trajectory_seed2022_accuracy.csv
output/uda/office-home/temporal_precision_head_trajectory_seed2022_summary.json
output/uda/office-home/temporal_precision_head_trajectory_seed2022_peak_gap.json
output/uda/office-home/temporal_precision_head_trajectory_seed2022_online_accuracy.csv
output/uda/office-home/temporal_precision_head_trajectory_seed2022_online_summary.json
```

Gate:

```text
trajectory peak mean must exceed DUET 84.7167
trajectory peak mean must exceed the matched online peak mean
compare with historical Stage14 seed-2022 peak 84.6783
no severe task collapse
```

If Step 1 fails, archive it and stop trajectory snapshot combinations. The
next method is dataset-level stable class-pair conflict flow: estimate
persistent source-to-CLIP disagreement transitions across cycles and constrain
a low-rank classifier adapter to that aggregate confusion subspace.

## Step 2: Three-Seed Stability

Run only if Step 1 passes:

```bash
cd /openbayes/home/vlm-sfda-conflict/duet-sfda-main
bash tools/run_office_home_temporal_precision_head_trajectory_seed_sweep.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_head_trajectory_seed_sweep_accuracy.csv
output/uda/office-home/temporal_precision_head_trajectory_seed_sweep_summary.json
```

Gate:

```text
all peak-selected seed means exceed 84.7167
seed-mean standard deviation <= 0.10
aggregate mean exceeds Stage14 peak mean 84.7825
```

If Step 2 fails, move to stable class-pair conflict flow. If it passes,
validate independent source checkpoints.

## Protocol Note

Peak selection uses target-domain labels and is recorded as a
best-checkpoint/oracle evaluation protocol. The fixed ensemble membership
itself does not use per-task labels.

## Status

```text
implementation complete
local validation passed (26 tests)
cloud result pending
```
