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

Important update after method discussion:

The next method should **not** be framed as simply:

```text
DUET + candidate-set loss
```

That would likely be too incremental. A static candidate-set loss is useful as
an ablation, but it should not be the final claimed contribution.

The stronger direction is:

> Model conflict samples as dynamic partial-label samples with a lifecycle:
> discover -> candidate learning -> promote / reject.

Working name:

```text
DCCL: Dynamic Conflict Candidate Learning
```

or:

```text
Conflict Lifecycle Learning
```

The method should contain three conceptual parts:

1. Conflict candidate construction

   ```text
   if source_pred != clip_pred:
       C_i = {source_pred, clip_pred}
   ```

2. Candidate-set learning for ambiguous conflicts

   ```text
   L_candidate = -log(p(source_pred) + p(clip_pred))
   ```

   This is not the whole method. It is the training action used while a sample
   is still ambiguous.

3. Dynamic state transition

   ```text
   candidate_mass = p(source_pred) + p(clip_pred)
   candidate_gap  = abs(p(source_pred) - p(clip_pred))

   if candidate_mass is high and one candidate dominates for K cycles:
       promote conflict sample to a hard pseudo-label

   elif candidate_mass stays low:
       reject or delay the sample

   else:
       keep candidate-set learning
   ```

This shifts the innovation from adding a loss to modeling the lifecycle of
source/VLM conflicts during adaptation.

## Revised Implementation Plan

Implement training variants in this order:

```text
DUET baseline
DUET + hard conflict selection
DUET + naive candidate-set loss
DUET + weighted candidate-set loss
DUET + weighted candidate-set loss + reject
DUET + dynamic candidate shrinking
Full method
```

The first three variants are diagnostic/ablation baselines. The paper's main
method should be the dynamic version, not the static loss-only version.

## Method Framing

Avoid this framing:

> We add a candidate-set loss to DUET.

Use this framing:

> We reformulate source/VLM disagreement samples as dynamic partial-label target
> samples. Instead of discarding them or forcing a hard pseudo-label, the method
> first learns from the compact candidate set and later promotes or rejects the
> sample according to model support.

Possible contribution bullets:

1. We reveal that source/VLM conflicts are frequent and informative in
   VLM-guided SFDA.
2. We show that simple hard selectors based on confidence, prototype
   consistency, or neighborhood support are not reliable enough.
3. We formulate conflict samples as dynamic partial-label samples and propose a
   conflict lifecycle mechanism with candidate learning, promotion, and
   rejection.

## Concrete Next Coding Target

Create a new method file rather than modifying DUET directly:

```text
src/methods/oh/dccl.py
cfgs/office-home/dccl.yaml
```

Start with DCCL-lite:

```text
agreement samples:
    DUET hard pseudo-label training

conflict samples:
    candidate-set loss with low weight

state tracking:
    maintain candidate_mass and candidate_gap per sample per cycle

promotion:
    if same candidate dominates for K cycles, convert to hard pseudo-label

rejection:
    if candidate_mass is below tau_low, ignore or consistency-only
```

Initial hyperparameters to try:

```text
lambda_candidate = 0.05, 0.1
tau_low = 0.4
tau_high = 0.7
gap_promote = 0.3
K = 2 cycles
```

Run first on:

```text
A->C
A->P
A->R
```

Only expand to all 12 Office-Home tasks if at least two of these three improve
over DUET.

## Important Caveat

Candidate-set recall equals useful-conflict rate in this binary-conflict
setting because `source_pred != clip_pred`; if either source or CLIP is correct,
the true label is in the candidate set. This should be stated carefully in the
paper.

Another caveat:

If the final method only improves by adding static candidate-set loss, the
novelty will likely be weak. The method needs dynamic promotion/rejection or a
clear conflict lifecycle mechanism to be defensible as more than a DUET loss
term.
