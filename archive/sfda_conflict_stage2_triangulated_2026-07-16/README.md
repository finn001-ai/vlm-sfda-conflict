# Stage-2 Triangulated Conflict Analysis

Date: 2026-07-16

This archive stores the offline analysis after testing whether confidence,
prototype consistency, neighborhood support, or their combination can decide
which side of a source/CLIP conflict should be trusted.

## Files

- `raw/triangulated_conflict_analysis.csv`  
  Cloud output copied from the triangulated analysis script.

- `triangulated_summary.json`  
  Macro averages and counts of how often each selector beats `always_clip`.

- `analysis.md`  
  Interpretation and next-step method decision.

## Main Conclusion

Hard selection is not reliable enough.

Macro averages on conflict samples:

| Selector / Metric | Macro Avg |
|---|---:|
| Candidate-set recall | 72.14 |
| Always trust CLIP | 55.33 |
| Higher confidence | 54.03 |
| Prototype only | 43.00 |
| Neighborhood only | 41.19 |
| Triangulated hard selector | 50.46 |

The strongest signal is not hard selection. It is that the true label is often
inside the two-label candidate set `{source_pred, clip_pred}`.

This suggests the next method should use conflict samples through candidate-set
supervision instead of forcing source/CLIP hard pseudo-label selection.
