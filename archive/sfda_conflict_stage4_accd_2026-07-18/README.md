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

## Diffusion Probe Results

The first fixed-configuration probe passed on all three target-Clipart tasks:

| Task | Conflict rate | Candidate recall | Eligible coverage | ACCD eligible acc. | CLIP eligible acc. | Net corrections | Projected full gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| A->C | 57.43 | 59.55 | 15.92 | 72.18 | 61.65 | +42 | +0.96 |
| P->C | 62.18 | 60.46 | 15.70 | 74.41 | 66.90 | +32 | +0.73 |
| R->C | 54.73 | 58.35 | 17.58 | 68.81 | 60.48 | +35 | +0.80 |

Interpretation:

1. The candidate-set assumption is weaker after `both_prior` calibration than
   the original raw diagnostic suggested: only about 58%-60% of conflicts have
   the true label in `{source_pred, clip_pred}`.
2. ACCD selects a conservative 16%-18% subset and improves over CLIP on that
   same subset by 7.5-10.5 percentage points on every task.
3. The projected net correction is comparable to the current 0.82-1.03 point
   gap to the DUET paper values, so adaptation training is justified.
4. `outside_candidate_precision` is only 48.43%-53.35%. Low candidate mass is
   not accurate enough to reject samples and must remain diagnostic-only.

The raw result is stored as `accd_diffusion_probe.json`. The next experiment is
the fixed ACCD training run on A->C, P->C, and R->C. No graph-parameter sweep is
approved before observing these three final accuracies.

## ACCD v1 Training Results

| Task | `both_prior` | ACCD v1 | DUET paper | Delta vs prior | Delta vs DUET |
|---|---:|---:|---:|---:|---:|
| A->C | 72.78 | 73.31 | 73.60 | +0.53 | -0.29 |
| P->C | 72.81 | 72.74 | 73.70 | -0.07 | -0.96 |
| R->C | 72.97 | 73.20 | 74.00 | +0.23 | -0.80 |
| Average | 72.85 | 73.08 | 73.77 | +0.23 | -0.68 |

ACCD improves two of the three bottleneck tasks and gives the strongest local
A->C result so far, but it does not yet exceed DUET.

The cycle logs identify a state-management failure rather than a need for a
threshold sweep:

| Task | Initial anchors | Final anchors | Final persistent labels | Final label accuracy |
|---|---:|---:|---:|---:|
| A->C | 967 | 1879 | 142 | 61.27 |
| P->C | 869 | 1875 | 149 | 58.39 |
| R->C | 1000 | 1886 | 158 | 64.56 |

Two endogenous feedback paths are present:

1. Dynamic anchors almost double as adaptation progresses. Later agreements
   are generated by the model being trained and are not independent evidence.
2. A resolved conflict is permanent in v1. It remains a hard label even after
   the two graphs stop supporting it or the source/CLIP relationship changes.

## ACCD v2 Decision

ACCD v2 changes state semantics without changing graph or loss parameters:

```text
anchor memory: dynamic -> frozen_initial
resolution memory: persistent -> reversible
```

The initial anchor bank has about 93% oracle accuracy in the no-adaptation
probe, so it is frozen to avoid confirmation bias. A promoted conflict remains
active only while the current graph evidence supports the same label; loss of
support immediately demotes it to the normal DUET path.

Run A->C first:

```bash
bash tools/run_office_home_accd_v2_ac.sh
```

This is a mechanism comparison, not parameter tuning. Expand v2 only if it
exceeds ACCD v1 (`73.31`), with `73.60` as the DUET target.

## ACCD v2 Result

Two A->C runs produced:

| Run | A->C | Delta vs v1 | Delta vs DUET |
|---|---:|---:|---:|
| v2 run 1 | 73.10 | -0.21 | -0.50 |
| v2 run 2 | 73.17 | -0.14 | -0.43 |

The implementation behaved as designed: anchors stayed fixed at 966 and
unsupported labels were demoted. However, the result is consistently below v1.
The combined v2 experiment changed two state mechanisms, so it cannot identify
whether frozen anchors or reversible labels caused the regression.

Cycle-level evidence suggests testing frozen anchors separately. Compared with
v1, frozen anchors improved eligible conflict accuracy in the middle cycles,
whereas reversible memory reduced the number of active labels and did not
increase their final accuracy. The next controlled ablation is therefore:

```text
anchor memory: frozen_initial
resolution memory: persistent
all numeric parameters: unchanged
```

Run only A->C:

```bash
bash tools/run_office_home_accd_frozen_persistent_ac.sh
```

This branch is accepted only if it exceeds v1 (`73.31`). Otherwise dynamic
anchors are retained and further anchor-memory variants are stopped.

## Risks And Falsification

- Agreement anchors can still contain wrong labels. ACCD reduces this through
  class-balanced confidence selection but does not make anchors clean.
- CLIP and task feature graphs are not statistically independent. The method
  requires empirical evidence that cross-space agreement improves precision.
- If the diffusion probe fails, do not tune graph thresholds extensively. The
  result would falsify dual-space topology as the missing reliability signal,
  and the next direction should be learned teacher-noise modeling rather than
  another graph variant.
