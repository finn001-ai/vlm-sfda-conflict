# Stage 11: Temporal Precision Memory

Date: 2026-07-18

## Rationale

Stage 10 shows that graph-temporal residual regularization is the first
graph-temporal training interface to improve two of three target-Clipart tasks,
but it still does not reach the DUET paper line.

The residual diagnostics expose a more fundamental bottleneck than loss design:

```text
valid pseudo labels grow to about 4000/4365 samples
valid pseudo-label accuracy drops to about 76-77%
```

The default DCCL pseudo-label memory is monotonic:

```text
label_mask_t = label_mask_{t-1} OR current_source_clip_agreement
```

Once a sample enters the supervised CE pool, it stays there even if later
source/CLIP agreement disappears or changes. This stage changes pseudo-label
admission rather than adding another loss.

## Method

Use temporal precision memory:

```text
DCCL.PL_MEMORY = stable
DCCL.PL_STABLE_CYCLES = 2
DCCL.PL_STABLE_MEMORY = reversible
DCCL.PL_MEMORY_WARMUP_CYCLES = 1
```

Cycle 1 uses current source/CLIP agreement as warmup. After warmup, a sample is
supervised only if the same agreement label remains stable for two consecutive
cycles. Reversible memory demotes labels when current agreement is no longer
stable.

The graph-temporal residual from stage 10 is kept, but the main change is the
pseudo-label memory/admission mechanism.

## Implementation

```text
duet-sfda-main/cfgs/office-home/temporal_precision_residual.yaml
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/tools/run_office_home_temporal_precision_residual_clipart.sh
```

## Cloud Command

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_temporal_precision_residual_clipart.sh
```

Bring back:

```text
output/uda/office-home/temporal_precision_residual_clipart_accuracy.csv
output/uda/office-home/temporal_precision_residual_dynamics_probe.json
```

## Gate

Compare against both residual and DUET:

| Task | both_prior | graph_temporal_residual | DUET paper |
|---|---:|---:|---:|
| A->C | 72.78 | 72.99 | 73.60 |
| P->C | 72.81 | 73.15 | 73.70 |
| R->C | 72.97 | 72.90 | 74.00 |

Proceed only if:

```text
mean target-Clipart accuracy beats graph_temporal_residual mean 73.01
at least two tasks improve over both_prior
pseudo-label accuracy is higher than stage 10 at comparable or acceptable coverage
```

If this fails, the next direction should move away from pseudo-label memory and
toward representation-level or source-model reliability analysis.
