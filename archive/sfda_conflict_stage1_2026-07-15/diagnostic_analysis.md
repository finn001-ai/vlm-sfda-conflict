# Diagnostic Analysis

## 12-Task Office-Home Table

| Task | Source Acc | CLIP Acc | Conflict Rate | Useful Conflict / All | Useful Conflict / Conflicts |
|---|---:|---:|---:|---:|---:|
| A->C | 44.88 | 59.43 | 58.60 | 33.61 | 57.35 |
| A->P | 67.18 | 84.23 | 34.69 | 27.46 | 79.16 |
| A->R | 74.57 | 85.36 | 27.52 | 22.31 | 81.07 |
| C->A | 53.73 | 74.78 | 50.14 | 35.89 | 71.57 |
| C->P | 61.70 | 84.23 | 39.69 | 32.17 | 81.04 |
| C->R | 65.00 | 85.36 | 36.91 | 30.41 | 82.40 |
| P->A | 51.92 | 74.78 | 50.39 | 35.56 | 70.56 |
| P->C | 40.78 | 59.43 | 63.83 | 36.93 | 57.86 |
| P->R | 72.30 | 85.36 | 29.58 | 24.17 | 81.69 |
| R->A | 64.52 | 74.78 | 39.60 | 27.65 | 69.82 |
| R->C | 45.93 | 59.43 | 55.88 | 30.68 | 54.90 |
| R->P | 77.90 | 84.23 | 23.99 | 18.77 | 78.22 |

## Macro Summary

| Metric | Average | Min | Max |
|---|---:|---:|---:|
| Source accuracy | 60.03 | 40.78 | 77.90 |
| CLIP accuracy | 75.95 | 59.43 | 85.36 |
| Agreement rate | 57.43 | 36.17 | 76.01 |
| Conflict rate | 42.57 | 23.99 | 63.83 |
| Useful conflict / all samples | 29.63 | 18.77 | 36.93 |
| Useful conflict / conflict samples | 72.14 | 54.90 | 82.40 |

## Sample-Weighted Summary

| Metric | Value |
|---|---:|
| Total target samples across tasks | 46764 |
| Conflict samples | 19647 |
| Useful conflict samples | 13653 |
| Conflict rate | 42.01 |
| Useful conflict / all samples | 29.20 |
| Useful conflict / conflict samples | 69.49 |
| Source correct, CLIP wrong / all samples | 6.77 |
| Source wrong, CLIP correct / all samples | 22.42 |
| Both wrong conflict / all samples | 12.82 |

## Interpretation

The diagnostic results support three claims.

First, conflict samples are not rare. On average, 42.57% of target samples show
source/VLM disagreement. This makes conflict handling central rather than
peripheral in VLM-guided SFDA.

Second, conflict samples are not simply noise. Across all target samples,
29.20% are useful conflicts, meaning one predictor is correct while the other
is wrong. Among conflict samples only, 69.49% are useful conflicts. This means
agreement-only pseudo-labeling can discard a large amount of recoverable target
signal.

Third, CLIP is helpful but unsafe to trust blindly. The case `source wrong,
CLIP correct` is much larger than `source correct, CLIP wrong`, showing that
CLIP often corrects source bias. However, `source correct, CLIP wrong` is
stable across tasks, and `both wrong` conflicts are also substantial. A method
therefore needs to distinguish useful conflicts from harmful conflicts rather
than simply trusting CLIP or averaging predictions.

## Paper-Ready Wording

Possible paragraph:

> We first conduct a diagnostic study on all 12 Office-Home transfer tasks to
> examine how source-model and CLIP predictions interact under target-domain
> shift. Surprisingly, disagreement is not an edge case: source/VLM conflicts
> account for 42.0% of target samples in a sample-weighted average. More
> importantly, 69.5% of these conflict samples contain a correct prediction from
> either the source model or CLIP. This reveals that disagreement samples are not
> merely noisy pseudo-labels to be discarded; rather, they contain substantial
> underexploited supervision signal. At the same time, 12.8% of all samples fall
> into the harmful case where both predictors are wrong, motivating a
> conflict-aware reliability criterion.

## What This Enables

This stage justifies a method that explicitly separates:

- agreement samples: reliable pseudo-label training
- useful conflict samples: trust the more reliable side
- harmful conflict samples: reject, delay, or use only weak consistency

The next question is no longer whether conflict samples matter. The next
question is how to identify useful conflict without target labels.
