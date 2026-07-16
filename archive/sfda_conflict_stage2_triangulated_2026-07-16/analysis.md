# Analysis

## What Was Tested

For each conflict sample:

```text
source_pred != clip_pred
```

the script evaluated whether we can decide which side is correct using:

- confidence
- prototype consistency
- neighborhood support
- a triangulated combination of the above

Labels were used only for offline evaluation, not to compute the selector
scores.

## Key Result

The average results over all 12 Office-Home tasks are:

| Metric | Macro Avg |
|---|---:|
| candidate_set_recall | 72.14 |
| always_source_acc | 16.81 |
| always_clip_acc | 55.33 |
| higher_conf_acc | 54.03 |
| prototype_only_acc | 43.00 |
| neighborhood_only_acc | 41.19 |
| triangulated_acc | 50.46 |

`always_clip` remains the strongest hard pseudo-label baseline among these
simple selectors. The triangulated hard selector does not beat it.

## Interpretation

This does not weaken the conflict-sample direction. It sharpens it.

The previous stage showed that conflicts are frequent and often useful. This
stage shows that hard-selecting one side is unreliable with simple evidence.
Therefore, the method should not primarily be:

```text
choose source OR choose CLIP
```

Instead, the method should exploit the fact that, for many conflicts, the true
label is inside the two-label candidate set:

```text
{source_pred, clip_pred}
```

This motivates candidate-set learning:

```text
L_conflict = -log(p(source_pred) + p(clip_pred))
```

This avoids premature hard pseudo-label decisions while still extracting useful
training signal from disagreement samples.

## Paper Story Update

Old framing:

> Can we identify which side of a conflict is correct?

Updated framing:

> Source/VLM conflict samples frequently contain the true class in a compact
> two-label candidate set, but simple hard selectors are unreliable. We
> therefore reformulate conflict learning as candidate-set supervision with
> optional rejection of ambiguous or harmful conflicts.

## Next Implementation Step

Implement the smallest training variant in DUET/PLMatch:

```text
agreement samples:
    cross-entropy with agreed pseudo label

conflict samples:
    candidate-set loss
    L = -log(p(source_pred) + p(clip_pred))
```

Recommended first variants:

| Variant | Description |
|---|---|
| DUET / PLMatch baseline | Existing method |
| + candidate-set loss | Use all conflicts with candidate-set supervision |
| + weighted candidate-set loss | Lower weight for conflict loss |
| + candidate-set loss with confidence gap reject | Use candidate loss only for sufficiently confident conflicts |

## Important Caveat

Candidate-set recall equals useful-conflict rate in this binary-conflict
setting because `source_pred != clip_pred`; if either source or CLIP is correct,
the true label is in the candidate set. This should be stated carefully in the
paper.
