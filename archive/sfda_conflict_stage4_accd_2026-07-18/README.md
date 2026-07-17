# Stage 4: Agreement-Anchored Candidate-Compatibility Diffusion

Date: 2026-07-18

## Decision

The untrained `dual_conflict` KL idea is abandoned. A confidence-weighted
mixture of source and CLIP distributions is still ordinary dual-teacher
distillation. It neither supplies independent evidence for conflict resolution
nor explains why the true class is sometimes absent from the two-label
candidate set.

The new method candidate is:

```text
ACCD: Agreement-Anchored Candidate-Compatibility Diffusion
```

ACCD changes pseudo-label inference before optimization. It is not introduced
as an additional loss.

## Evidence Driving The Change

The existing Office-Home diagnostics show:

| Observation | Value / conclusion |
|---|---:|
| Weighted conflict rate | 42.01% |
| Useful conflicts among conflicts | 69.49% |
| True label outside `{source, CLIP}` among conflicts | about 30.51% |
| Prototype selector vs always-CLIP | worse on all 12 tasks |
| Neighborhood selector vs always-CLIP | worse on all 12 tasks |
| Candidate loss | did not improve A->C |
| `both_prior` full Office-Home average | 84.30% |
| DUET paper average | 84.70% |

Therefore, forcing every conflict into `{source_pred, clip_pred}` is invalid.
The method first needs an unconstrained estimate over all classes, followed by
a test of whether the two-label candidate set is compatible with target-domain
structure.

## Method

For every adaptation cycle:

1. Apply the verified `both_prior` calibration to task-model and CLIP
   probabilities.
2. Select class-balanced, high-confidence agreement samples as anchors.
3. Construct a kNN graph in the adapted task-feature space and another kNN
   graph in the CLIP visual-feature space.
4. Propagate anchor labels independently on both graphs with random restart.
5. Fuse the two graph posteriors using a product of experts.
6. For a conflict sample, compute the posterior mass on
   `{source_pred, clip_pred}` before constraining the label space.
7. A conflict is eligible only when both graphs choose the same candidate, the
   candidate mass is high, and the candidate margin is sufficient.
8. Promote an eligible label only after it remains unchanged across two
   adaptation cycles.
9. Use the graph posterior as the KL target and the stable graph label as the
   hard pseudo-label only for promoted conflicts. All other samples retain the
   DUET path.

The key distinction is global, cross-space inference. Earlier prototype and
nearest-neighbor trials were static one-space selectors. ACCD uses multi-step
propagation, two independently constructed target manifolds, out-of-candidate
detection, and temporal confirmation.

## Relation To Recent Work

- DUET accepts source/CLIP agreements and treats disagreement as uncertainty.
- ProDe corrects VLM logits using source-to-target adaptation dynamics.
- DIFO++ focuses on gap regions and memory-supported source/VLM fusion.
- Feature Universe models target topology for pseudo-label propagation.
- ReCLIP propagates labels from class-text embeddings in a refined CLIP space.
- DPCLS studies global candidate-label disambiguation in conventional partial
  label learning, without the source-free dual-teacher setting.
- ACCD specifically formulates source/VLM disagreement as candidate-aware
  transductive inference over two target manifolds and explicitly detects when
  neither teacher's top-1 class is structurally plausible.

Primary sources:

- DUET: https://proceedings.neurips.cc/paper_files/paper/2025/hash/6c8dfbbd1ba3e22339e58a336cbed52b-Abstract-Conference.html
- ProDe: https://proceedings.iclr.cc/paper_files/paper/2025/hash/cd5404354496e39d37b7947d8a0d7b72-Abstract-Conference.html
- DIFO++: https://arxiv.org/abs/2604.17748
- Feature Universe: https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Measure_The_Feature_Universe_Topology-based_Pseudo_Labeling_and_Gravity_Consistency_CVPR_2026_paper.html
- ReCLIP: https://openaccess.thecvf.com/content/WACV2024/html/Hu_ReCLIP_Refine_Contrastive_Language_Image_Pre-Training_With_Source_Free_Domain_WACV_2024_paper.html
- DPCLS: https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html

## Implementation

Core files:

```text
duet-sfda-main/src/utils/conflict_diffusion.py
duet-sfda-main/src/methods/oh/accd.py
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/cfgs/office-home/accd.yaml
```

Diagnostic files:

```text
duet-sfda-main/tools/export_conflict_diagnostics.py
duet-sfda-main/tools/analyze_accd_diffusion.py
duet-sfda-main/tools/run_office_home_accd_diffusion_probe.sh
```

Training script:

```text
duet-sfda-main/tools/run_office_home_accd_clipart.sh
```

## Scientific Gate Before Training

Run the no-adaptation probe first:

```bash
bash tools/run_office_home_accd_diffusion_probe.sh
```

For each A->C, P->C, and R->C task, the probe reports:

```text
eligible_conflict_coverage
eligible_accuracy
eligible_clip_accuracy
net_correct_gain_over_clip
outside_candidate_precision
pass_training_gate
```

Training is justified only when graph-resolved conflicts cover at least 5% of
all conflicts, produce positive net correct predictions over CLIP on the same
selected subset, and have higher accuracy than CLIP on that subset. This gate
uses ground truth only for evaluation; labels are not inputs to ACCD inference.

If the gate passes, run:

```bash
bash tools/run_office_home_accd_clipart.sh
```

Target-Clipart comparison baselines:

| Task | `both_prior` | DUET paper |
|---|---:|---:|
| A->C | 72.78 | 73.60 |
| P->C | 72.81 | 73.70 |
| R->C | 72.97 | 74.00 |

Do not expand to all 12 tasks unless at least two of these three improve over
`both_prior`, and at least one approaches or exceeds the corresponding DUET
paper result.

## Risks And Falsification

- Agreement anchors can still contain wrong labels. ACCD reduces this through
  class-balanced confidence selection but does not make anchors clean.
- CLIP and task feature graphs are not statistically independent. The method
  requires empirical evidence that cross-space agreement improves precision.
- If the diffusion probe fails, do not tune graph thresholds extensively. The
  result would falsify dual-space topology as the missing reliability signal,
  and the next direction should be learned teacher-noise modeling rather than
  another graph variant.
