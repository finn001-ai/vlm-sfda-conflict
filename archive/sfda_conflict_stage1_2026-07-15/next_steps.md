# Next Steps

## Immediate Next Experiment

Use the per-sample CSV files to test whether target-label-free reliability
signals can separate useful conflicts from harmful conflicts.

Start with these simple analyses:

1. Confidence separation
   - Compare source confidence and CLIP confidence for:
     - source correct, CLIP wrong
     - source wrong, CLIP correct
     - both wrong
   - Check whether max confidence, margin, or confidence difference predicts
     the correct side.

2. Class-wise reliability
   - Estimate source and CLIP reliability per predicted class using agreement
     samples.
   - For conflict samples, choose the side with higher class-wise reliability.

3. Prototype consistency
   - Build target feature prototypes from reliable agreement samples.
   - For a conflict sample, check whether its feature is closer to the source
     predicted class prototype or the CLIP predicted class prototype.

4. Temporal stability
   - During adaptation, track whether a sample's source/model prediction is
     stable across epochs.
   - Treat unstable conflicts as harmful or delayed.

## Minimal Method Candidate

Name placeholder:

> Conflict-Aware Reliability Selection (CARS)

Inputs per target sample:

```text
source_pred
source_conf
clip_pred
clip_conf
target feature
class-wise source/CLIP reliability
prototype consistency
```

Rules:

```text
if source_pred == clip_pred and confidence is high:
    use hard pseudo-label

elif source_pred != clip_pred:
    compute reliability(source side)
    compute reliability(CLIP side)
    if one side is clearly more reliable:
        use that side as pseudo-label with lower weight
    else:
        reject or delay sample

else:
    reject or use weak consistency only
```

## Minimal Ablation Plan

| Variant | Purpose |
|---|---|
| DUET / PLMatch baseline | Main comparison platform |
| Agreement-only | Tests what happens when conflicts are ignored |
| Naive confidence choice | Tests whether confidence alone is enough |
| Class-wise reliability only | Tests class reliability signal |
| Prototype consistency only | Tests feature-structure signal |
| Full conflict-aware selector | Main method |

## Main Risk

The current diagnostic uses ground-truth labels only for analysis. The method
must not use target labels. The next experiments should therefore evaluate
whether unlabeled reliability proxies can approximate the oracle conflict
partition.

## Writing Direction

The paper should not be framed as:

> Foundation models improve SFDA.

It should be framed as:

> Existing VLM-guided SFDA methods underexploit disagreement samples. We show
> that conflicts are frequent and often informative, then propose a reliability
> criterion to safely learn from useful conflicts while suppressing harmful
> ones.
