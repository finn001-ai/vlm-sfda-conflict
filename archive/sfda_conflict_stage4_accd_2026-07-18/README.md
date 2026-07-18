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

## Frozen-Anchor Persistent Result

The controlled A->C ablation achieved `73.36`:

| Method | A->C |
|---|---:|
| `both_prior` | 72.78 |
| ACCD v1, dynamic + persistent | 73.31 |
| ACCD frozen + reversible | 73.10 / 73.17 |
| ACCD frozen + persistent | **73.36** |
| DUET paper | 73.60 |

Freezing the initial anchor bank is retained, but its isolated gain is only
`+0.05`; further anchor-memory variants are stopped.

## Asymmetric Source Rescue

The remaining training action is unnecessarily symmetric. When the graph
selects the CLIP candidate, ACCD currently converts a conflict into a hard CLIP
label even though DUET already supplies soft CLIP supervision through KL. This
adds no new teacher information and can harden CLIP errors. The actual new
signal is the opposite case: both target graphs support the source candidate
against CLIP.

The next mechanism therefore uses ACCD as an asymmetric adjudicator:

```text
graph selects source candidate -> temporal promotion and graph target
graph selects CLIP candidate   -> unchanged DUET conflict path
```

All graph parameters, anchor settings, calibration, loss weights, and temporal
stability requirements remain unchanged. Run A->C only:

```bash
bash tools/run_office_home_accd_source_rescue_ac.sh
```

This source-rescue branch must exceed `73.36` to be retained and must reach
`73.60` to meet the current A->C objective.

## Asymmetric Source Rescue Result

Source rescue achieved `73.31` on A->C, below frozen+persistent ACCD (`73.36`)
and DUET (`73.60`). The branch is therefore not retained as the final action.

The diagnostic signal remains useful. At the final cycle, the 135 currently
eligible graph-to-source conflicts had `56.30%` graph/source accuracy versus
only `21.48%` CLIP accuracy. The 99 temporally resolved samples had `66.67%`
accuracy. Thus dual-space topology reliably identifies a region where CLIP is
especially harmful, but source accuracy is still too low for hard labels.

## Topology-Gated Teacher Abstention

The next controlled action treats graph-to-source evidence as a teacher-noise
detector rather than a replacement teacher:

```text
stable graph-to-source conflict -> suppress CLIP KL only
                                -> do not add a hard source label
all other samples               -> unchanged DUET supervision
```

This differs from the failed global `conflict_kl_off` experiment (`67.31`):
global removal discards useful CLIP supervision from every conflict, whereas
this intervention abstains only on the small subset selected independently by
two target-domain manifolds. All numerical settings remain fixed.

Run A->C:

```bash
bash tools/run_office_home_accd_teacher_abstain_ac.sh
```

Retain this action only if it exceeds `73.36`; `73.60` remains the DUET target.

## Teacher Abstention Result

Topology-gated teacher abstention achieved `72.92` on A->C. It is below
source-rescue hard labeling (`73.31`), frozen+persistent ACCD (`73.36`), and
DUET (`73.60`), so the abstention branch is stopped without tuning its weight.

Although CLIP top-1 accuracy on the selected region was only about 20%, fully
removing its KL signal was harmful. This indicates that CLIP's non-top-1 class
distribution still carries useful semantic structure or regularization.

## Candidate-Mass Transport

The next action performs a minimal intervention on the CLIP teacher. For every
temporally stable graph-to-source conflict that is still supported by the two
current graphs:

```text
preserve every non-candidate class probability
preserve q(source) + q(CLIP)
redistribute only this candidate mass using the dual-graph posterior ratio
do not create a new hard pseudo-label
```

This directly addresses both observed failures: it does not force a 65%-accurate
source hard label, and it does not discard CLIP's full soft distribution. It
introduces no new loss weight or threshold.

Run A->C:

```bash
bash tools/run_office_home_accd_candidate_transport_ac.sh
```

Retain only if final accuracy exceeds `73.36`; reaching `73.60` remains the
current objective.

## Candidate-Mass Transport Result

Candidate-mass transport achieved `72.88` on A->C, below teacher abstention
(`72.92`), source rescue (`73.31`), frozen+persistent ACCD (`73.36`), and DUET
(`73.60`). The final cycle transported 54 samples with mean shifted mass
`0.6385`. Preserving total candidate mass did not make the intervention mild;
the graph ratio still moved most of that mass and behaved like a soft hard
selection.

This closes the fixed graph-action family. No sweep over transport strength,
abstention weight, graph threshold, or action mixture is planned.

## Counterfactual Adjudicator Probe

The next direction replaces hand-written conflict actions with a learned
pairwise reliability model while remaining source-free and label-free:

1. Select high-confidence source/CLIP agreement anchors.
2. Treat their shared prediction as a noisy target pseudo-label.
3. Corrupt one teacher's candidate scores to synthesize both conflict directions.
4. Learn a logistic pairwise adjudicator from expert probabilities, two graph
   posteriors, fused diffusion, task/CLIP prototype support, and class support.
5. Evaluate on real conflicts; ground truth is used only in the final report.

Run the no-adaptation probe on the existing A->C/P->C/R->C NPZ exports:

```bash
bash tools/run_office_home_counterfactual_probe.sh
```

Do not integrate the adjudicator into training unless it beats always-CLIP on
multiple tasks with positive net corrections.

## Counterfactual Adjudicator Result

The adjudicator does not pass a statistically meaningful training gate:

| Task | Always CLIP | Adjudicator | Net corrections | Projected full gain | One-sided p |
|---|---:|---:|---:|---:|---:|
| A->C | 43.6378 | 43.7575 | +3 | +0.0687 | 0.4695 |
| P->C | 46.3154 | 46.5733 | +7 | +0.1604 | 0.4014 |
| R->C | 42.1515 | 43.5329 | +33 | +0.7560 | 0.0925 |

Synthetic grouped cross-validation was `99.84%-100%`, but real-conflict
accuracy was only `43.53%-46.57%`. The model learned the synthetic corruption
mechanism and did not transfer reliably to natural conflicts. The previous
`pass_training_gate` check was too permissive because any positive integer net
gain passed; it is replaced with a one-sided paired correction test at
`p < 0.05`. None of these tasks passes, so the adjudicator is not integrated.

## Full ACCD Validation

Single-task A->C method search is now stopped. The alternative project target
is a 12-task average above DUET's `84.70`. Run the current best controlled
mechanism, frozen anchors with persistent conflict labels, in an isolated
output directory:

```bash
bash tools/run_office_home_accd_frozen_persistent_all.sh
```

The result determines whether ACCD is retained as the main method or reduced
to a negative/diagnostic finding. Do not select per-task variants when
computing the average.

## Full ACCD Result And Stop Decision

Frozen-anchor persistent ACCD achieved a 12-task average of `84.3075`:

| Method | Average | Delta vs DUET | Delta vs both_prior |
|---|---:|---:|---:|
| DUET paper | 84.7167 | - | +0.4134 |
| `both_prior` | 84.3033 | -0.4134 | - |
| ACCD frozen+persistent | 84.3075 | -0.4092 | +0.0042 |

ACCD beats the DUET paper on only 2/12 tasks and beats the same-environment
`both_prior` baseline on 5/12. Its average improvement over `both_prior` is
only `+0.0042`, so it is not an effective main method. Target-domain averages
also show no broad rescue:

| Target | ACCD | DUET paper | both_prior |
|---|---:|---:|---:|
| Clipart | 72.9500 | 73.7667 | 72.8533 |
| Art | 82.8167 | 83.3000 | 82.6400 |
| Product | 90.7800 | 90.7667 | 90.9000 |
| RealWorld | 90.6833 | 91.0333 | 90.8200 |

Stop ACCD as the paper's proposed method. Frozen/dynamic anchors, persistent/
reversible memory, symmetric/source-only hard labels, teacher abstention,
candidate-mass transport, and a counterfactual learned selector have now been
tested or probed. Further combinations within this family are not justified.

Before choosing a replacement method, audit the untouched PLMatch baseline in
the same environment. A->C has previously run below the paper value, so the
absolute paper gap may partly reflect source checkpoints or environment. Method
claims must compare against both the same-environment baseline and published
numbers.

## Risks And Falsification

- Agreement anchors can still contain wrong labels. ACCD reduces this through
  class-balanced confidence selection but does not make anchors clean.
- CLIP and task feature graphs are not statistically independent. The method
  requires empirical evidence that cross-space agreement improves precision.
- If the diffusion probe fails, do not tune graph thresholds extensively. The
  result would falsify dual-space topology as the missing reliability signal,
  and the next direction should be learned teacher-noise modeling rather than
  another graph variant.
