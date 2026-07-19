# SFDA Conflict Project: Results Through Stage 21

Date: 2026-07-19

This archive consolidates the Office-Home results for the VLM-guided SFDA
conflict-sample project. It is a cross-stage summary, not a new experimental
stage.

## Evaluation Rules

- The first row of the main table is the published DUET result.
- Task order is `AC, AP, AR, CA, CP, CR, PA, PC, PR, RA, RC, RP`.
- From the Stage14 re-evaluation onward, the primary reported value is the
  highest logged target accuracy for each task (`peak`), as requested.
- `peak` uses target labels and is therefore an oracle/best-checkpoint
  protocol. It is not a label-free SFDA model-selection procedure.
- Earlier runs without retained trajectories are reported as `final`; they
  cannot be reconstructed as `peak` from the current archive.
- A mean over one or three tasks is a local preflight mean and must not be
  compared with the 12-task DUET mean.
- Stage18 is retained as an invalid-mechanism control because its adapter was
  inactive. Stage21 has a valid three-task preflight but no full 12-task run.

## Main 12-Task Accuracy Table

The first row is DUET, as required. Bold means the row is above the DUET
12-task mean; this does not override the stability or protocol caveats.

| Stage / variant | Protocol | AC | AP | AR | CA | CP | CR | PA | PC | PR | RA | RC | RP | Mean | Delta vs DUET | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **DUET paper** | published | 73.60 | 90.40 | 91.00 | 83.60 | 90.70 | 90.90 | 82.70 | 73.70 | 91.20 | 83.60 | 74.00 | 91.20 | 84.7167 | 0.0000 | Public reference |
| Stage3 `clip_prior` | final | 72.78 | 90.88 | 91.00 | 83.27 | 90.81 | 90.57 | 82.20 | 72.44 | 90.80 | 82.53 | 72.69 | 90.79 | 84.2300 | -0.4867 | Fail |
| Stage3 `both_prior` | final | 72.78 | 90.81 | 91.00 | 83.23 | 90.92 | 90.64 | 82.12 | 72.81 | 90.82 | 82.57 | 72.97 | 90.97 | 84.3033 | -0.4134 | Retained baseline component |
| Stage4 ACCD frozen+persistent | final | 73.15 | 90.58 | 90.68 | 83.23 | 90.88 | 90.48 | 82.24 | 72.71 | 90.89 | 82.98 | 72.99 | 90.88 | 84.3075 | -0.4092 | Stop ACCD family |
| **Stage14 target head, first full run** | final | 73.54 | 90.97 | 91.12 | 83.64 | 91.15 | 90.73 | 83.35 | 73.38 | 91.14 | 83.56 | 73.86 | 91.10 | **84.7950** | +0.0783 | Single-run pass |
| **Stage14 seed 2020** | peak | 73.81 | 91.12 | 91.23 | 83.60 | 91.17 | 90.73 | 83.44 | 73.47 | 91.21 | 83.56 | 74.20 | 91.26 | **84.9000** | +0.1833 | Best observed seed |
| **Stage14 seed 2021** | peak | 73.84 | 90.83 | 90.96 | 84.10 | 91.15 | 90.57 | 83.31 | 73.47 | 90.57 | 83.52 | 73.88 | 91.03 | **84.7692** | +0.0525 | Seed pass |
| Stage14 seed 2022 | peak | 73.17 | 91.06 | 90.87 | 83.64 | 91.19 | 90.77 | 82.86 | 73.33 | 90.73 | 83.52 | 73.56 | 91.44 | 84.6783 | -0.0383 | Seed fail narrowly |
| **Stage14 three-seed mean** | peak mean | 73.6067 | 91.0033 | 91.0200 | 83.7800 | 91.1700 | 90.6900 | 83.2033 | 73.4233 | 90.8367 | 83.5333 | 73.8800 | 91.2433 | **84.7825** | +0.0658 | Stability fail: std 0.1114; seed 2022 below DUET |
| Stage15 EMA head, seed 2022 | peak | 73.97 | 90.70 | 91.00 | 83.11 | 90.72 | 90.77 | 82.32 | 74.04 | 90.84 | 82.69 | 73.65 | 90.81 | 84.5517 | -0.1650 | No oracle headroom |
| **Stage17 matched online Stage14** | peak | 73.59 | 90.83 | 90.96 | 83.68 | 91.12 | 90.75 | 83.11 | 73.52 | 90.52 | 83.52 | 73.56 | 91.51 | **84.7225** | +0.0058 | Control, not a new method gain |
| Stage17 trajectory ensemble | peak | 73.36 | 90.94 | 90.84 | 83.64 | 91.08 | 90.75 | 82.94 | 73.47 | 90.52 | 83.40 | 73.54 | 91.37 | 84.6542 | -0.0625 | Fail |
| Stage18 class-pair flow | peak | 73.24 | 91.15 | 90.98 | 83.40 | 91.42 | 90.61 | 82.78 | 72.76 | 90.77 | 83.27 | 73.61 | 91.30 | 84.6075 | -0.1092 | Invalid: active rank 0/12 |
| Stage19 pair-feature adapter | peak | 73.77 | 91.01 | 91.05 | 83.60 | 90.92 | 90.77 | 82.49 | 73.65 | 90.73 | 83.07 | 73.97 | 91.26 | 84.6908 | -0.0258 | Valid mechanism, accuracy gate fail |

For completeness, the Stage14 final-checkpoint seed means before peak
re-extraction were `84.8100`, `84.7275`, and `84.5267`; their mean was
`84.6881` with standard deviation `0.1457`. Stage15 EMA seed-2022 final mean
was `84.5033`, versus its oracle peak mean of `84.5517`.

The machine-readable version, including both Stage14 final and peak rows, is
`office_home_accuracy_master.csv`.

## Partial Training Results

These values are end-to-end target accuracies, but only for the tasks shown.
Their means are over the reported tasks, not all 12 Office-Home tasks.

| Stage / variant | Reported tasks | Accuracy | Local mean | Result |
|---|---|---|---:|---|
| Stage3 PLMatch same environment | AC/AP/AR | 72.03 / 90.52 / 90.82 | 84.4567 | Local reference only |
| Stage3 DCCL default | AC/AP/AR | 71.34 / 90.61 / 90.36 | 84.1033 | Below conservative route |
| Stage3 DCCL conservative | AC/AP/AR | 71.87 / 90.67 / 90.87 | 84.4700 | Candidate learning insufficient |
| Stage3 `auto_agree` | AC/AP/AR | 72.46 / 90.74 / 90.91 | 84.7033 | Worse than fixed `clip_prior` |
| Stage4 ACCD v1 | AC/PC/RC | 73.31 / 72.74 / 73.20 | 73.0833 | Improves 2/3 vs `both_prior`, below DUET |
| Stage9 graph teacher to both paths | AC/PC/RC | 73.15 / 72.30 / 72.71 | 72.7200 | Fail expansion gate |
| Stage9 graph teacher KL-only | AC/PC/RC | 72.74 / 72.03 / 73.08 | 72.6167 | Fail expansion gate |
| Stage10 graph-temporal residual | AC/PC/RC | 72.99 / 73.15 / 72.90 | 73.0133 | Better interface, still below DUET |
| Stage11 temporal precision memory | AC/PC/RC | 73.38 / 73.06 / 73.36 | 73.2667 | Pass stage gate; all beat `both_prior` |
| Stage12 target prototypes | AC/PC/RC | 73.06 / 72.90 / 73.47 | 73.1433 | Fail vs Stage11 |
| Stage13 target-prior balance | AC/PC/RC | 73.22 / 72.60 / 73.42 | 73.0800 | Fail vs Stage11 |
| Stage14 target head initial Clipart run | AC/PC/RC | 73.65 / 73.22 / 73.95 | 73.6067 | Pass; motivated full 12 tasks |
| Stage16 bounded residual head | 12 tasks planned | run manually stopped | - | Low intermediate accuracy; no complete result |
| Stage19 pair-feature mechanism preflight | AC | 73.68 | 73.6800 | Activation passed; proceeded to full seed 2022 |
| Stage19-C coverage fallback | CA/PA/RA | 83.60 / 82.61 / 82.98 | 83.0633 | Fail; projected full mean 84.6933 |
| Stage19-G GTR-only router | AC/PA/RA | 73.61 / 82.53 / 82.94 | 79.6933 | Valid route, fail; projected full mean 84.6700 |
| Stage20 conditional covariance | AC/PA/RA | 73.45 / 82.82 / 83.35 | 79.8733 | Valid transport, all regress vs matched Stage14 |
| Stage21 global whitened transport | AC/PA/RA | 73.24 / 83.11 / 83.40 | 79.9167 | Valid mechanism; -0.1567 vs matched Stage14 and -0.0500 vs DUET subset |

## Stage3 And Stage4 Fine-Grained Ablations

### Stage3 A-to-C candidate and calibration trials

| Trial | AC | Conclusion |
|---|---:|---|
| Candidate loss `0.005` | 71.87 | Fail |
| Candidate loss `0.01` | 71.98 | Fail |
| Candidate loss `0.02` | 71.71 | Stronger loss hurts |
| Candidate gate `tau=0.3` | 72.03 | Tie with local PLMatch |
| Remove conflict CLIP KL | 67.31 | CLIP soft distribution is necessary |
| Late candidate, start cycle 2 | 72.00 | Fail |
| Late candidate, start cycle 1 | 71.71 | Fail |
| Balanced pseudo-label top-30 | 71.71 | Fail |
| Balanced pseudo-label top-45 | 71.96 | Fail |
| Balanced top-45, minimum confidence 0.2 | 71.71 | Fail |
| `source_prior` | 72.10 | Small gain |
| `clip_prior` | 72.78 | First clear calibration gain |
| `both_prior` rapid trial | 72.74 | Promising unified component |
| `mix_prior` | 72.28 | Weaker |
| `auto_agree` | 72.46 | Unsupervised selector does not select final-best mode |

The weak-source calibration matrix was:

| Task | none | source prior | CLIP prior | both prior | mix prior | Best |
|---|---:|---:|---:|---:|---:|---|
| PA | 82.16 | 82.49 | 82.20 | 82.08 | 82.24 | source prior |
| PC | 71.04 | 71.32 | 72.19 | 72.78 | 71.89 | both prior |
| PR | 90.87 | 90.87 | 90.93 | 90.89 | 90.82 | CLIP prior |
| RA | 82.41 | 82.41 | 82.49 | 82.53 | 82.45 | both prior |
| RC | 71.98 | 72.33 | 72.88 | 72.97 | 72.78 | both prior |
| RP | 90.43 | 90.58 | 90.79 | 90.97 | 90.47 | both prior |

### Stage4 A-to-C graph-action trials

| Trial | AC | Conclusion |
|---|---:|---|
| ACCD v1 dynamic anchors + persistent labels | 73.31 | Better than prior, below DUET |
| ACCD v2 frozen + reversible, run 1 | 73.10 | Regresses |
| ACCD v2 frozen + reversible, run 2 | 73.17 | Regresses |
| Frozen anchors + persistent labels | 73.36 | Best ACCD AC ablation |
| Asymmetric source rescue | 73.31 | Fail |
| Topology-gated teacher abstention | 72.92 | Fail |
| Candidate-mass transport | 72.88 | Fail |
| Frozen+persistent full-run AC row | 73.15 | Full fixed method does not preserve AC ablation peak |

## Diagnostic And Probe Results

These are mechanism diagnostics, not end-to-end 12-task adaptation results.

| Stage | Probe | Main quantitative result | Decision |
|---|---|---|---|
| 1 | Conflict prevalence | Macro conflict 42.57%; weighted conflict 42.01%; useful conflicts macro 72.14%, weighted 69.49% | Conflict samples contain useful information |
| 2 | Static hard adjudication | Always CLIP 55.33%; confidence 54.03%; prototype 43.00%; neighbor 41.19%; triangulated 50.46%; candidate recall 72.14% | Hard selection unreliable |
| 4 | ACCD diffusion | AC/PC/RC eligible accuracy 72.18/74.41/68.81 vs CLIP 61.65/66.90/60.48 | Offline gate passes, training conversion weak |
| 4 | Counterfactual adjudicator | AC/PC/RC 43.76/46.57/43.53; one-sided p 0.4695/0.4014/0.0925 | No task passes statistical gate |
| 5 | Inverse topology prior | AC/PC/RC mixed top-1 57.21/49.55/50.79; mean delta vs `both_prior` -9.1409 | Fail before training |
| 6 | Corrected graph target prior | Standard gate reports `fail_training_gate`; exact per-task JSON was not archived | Stop graph-prior family |
| 7 | Temporal dynamics | Stable accuracy 65.50/66.47/65.59 vs CLIP 64.21/65.28/64.10; net +28/+28/+31 | Pass 3/3 |
| 8 | Soft graph-teacher fusion | Fused top-1 63.64/64.01/63.16; delta vs `both_prior` +2.52/+2.08/+1.24 | Pass 3/3 offline |
| 9 | KL-only temporal diagnostic | Stable accuracy 67.25/66.25/65.95; total net corrections +165 | Probe passes, training fails |
| 10 | Residual temporal diagnostic | Stable accuracy 66.56/66.74/66.97; total net corrections +163 | Probe passes 3/3 |
| 11 | Temporal precision diagnostic | Stable accuracy 67.43/67.81/67.02; total net corrections +176 | Probe passes 3/3 |
| 12 | Prototype diagnostic | Stable accuracy 67.21/67.41/66.62; total net corrections +177 | Probe passes, training fails |
| 13 | Balanced diagnostic | Stable accuracy 66.87/67.57/67.08; total net corrections +174 | Probe passes, training fails |
| 14 | Full target-head dynamics | Mean stable accuracy 77.63 vs stable CLIP 74.59; total net corrections +495; pass 12/12 | Dynamics signal generalizes to all tasks |

## Stage Methods And Lineage

The following explains what each stage did and which earlier stage it varied
from. This lineage is deliberately outside the accuracy table.

1. **Stage1: conflict diagnosis.** It measured source/CLIP agreement,
   disagreement, and oracle usefulness on all 12 tasks. This is the root
   evidence stage and has no parent variant.

2. **Stage2: triangulated hard selection.** It tested confidence, prototypes,
   neighbors, and their combination on Stage1 conflicts. It is a diagnostic
   variation of Stage1 and established that candidate-set recall is useful but
   per-sample hard choice is not.

3. **Stage3: conservative candidate learning and calibration.** It started
   from Stage2's two-label candidate-set observation, first testing candidate
   losses/admission and then moving to class-prior calibration. `clip_prior`,
   `both_prior`, and `auto_agree` are Stage3 internal variants. `both_prior`
   became the later baseline component.

4. **Stage4: ACCD dual-space diffusion.** It started from Stage3 `both_prior`
   and tried to resolve conflict candidates with agreement anchors, task/CLIP
   graphs, diffusion, and temporal memory. Frozen/reversible memory, source
   rescue, abstention, candidate transport, and the counterfactual adjudicator
   are Stage4 variants. The full result closed this sample-level graph-action
   family.

5. **Stage5: topology-prior calibration.** It started from Stage4's graph
   signal but aggregated graph posteriors into a dataset-level prior, instead
   of making per-sample choices. The inverse-prior formulation failed badly.

6. **Stage6: graph-target distribution alignment.** It is the corrected
   target-prior formulation of Stage5. Its no-training gate failed, closing
   graph-prior calibration rather than all uses of graph evidence.

7. **Stage7: temporal conflict dynamics.** It returned to the Stage3
   `both_prior` path after Stage6 and tested adaptation trajectories as an
   independent reliability signal. It is not a prompt or graph-rule variant.

8. **Stage8: soft graph-teacher fusion.** It combined Stage4's useful graph
   diffusion evidence with the Stage7 conclusion that soft, temporally checked
   signals are preferable to hard selection. This was an offline fusion probe.

9. **Stage9: graph-temporal training injection.** It started from the passing
   Stage8 probe and Stage7 diagnostics. Direct two-path teacher replacement and
   the KL-only subvariant both failed to convert offline gains into accuracy.

10. **Stage10: graph-temporal residual regularization.** It is a weaker
    training-interface variant of Stage9: preserve the `both_prior` teacher and
    use graph-temporal evidence only as a small residual regularizer. It
    improved two of three Clipart tasks but exposed noisy monotonic pseudo-label
    admission.

11. **Stage11: temporal precision memory.** It started from Stage10 and changed
    the pseudo-label state machine to two-cycle stable, reversible admission.
    This was the first clear mechanism improvement and remains part of the main
    baseline.

12. **Stage12: target prototypes.** It is a decision-boundary variation of
    Stage11 using stable-label target prototypes. It failed and prototype
    parameter variants were closed.

13. **Stage13: target-prior class balance.** It branched from Stage11, not from
    the failed Stage12 adapter. It rebalanced the stable pseudo-label pool by a
    target prior, but reduced accuracy.

14. **Stage14: temporal precision target head.** It branched from Stage11's
    successful memory and explicitly adapted a source-initialized target
    classifier blended with the frozen source head. It is the strongest method
    so far: peak mean `84.7825`, but stability still fails narrowly.

15. **Stage15: EMA target head.** It is a temporal smoothing variation of
    Stage14 intended to reduce seed sensitivity. Both final and oracle peak
    remained below DUET, so EMA momentum variants were stopped.

16. **Stage16: bounded residual head.** It is a constrained classifier
    parameterization variation of Stage14, motivated by Stage15 drift. The
    cloud run was manually stopped for low intermediate accuracy; no complete
    result exists.

17. **Stage17: cycle-4 trajectory ensemble.** It returned to the original
    Stage14 blend head after the peak re-evaluation. Fixed 50/75/100% snapshot
    averaging underperformed the matched online Stage14 control.

18. **Stage18: persistent class-pair logit flow.** It branched from Stage14
    after Stage17 and attempted dataset-level confusion-subspace adaptation.
    The agreement/conflict mask intersection made the adapter inactive, so its
    accuracy is an invalid mechanism test rather than a negative causal result.

19. **Stage19: aggregate pair-flow feature adapter.** It repaired Stage18 by
    freezing initial conflict pairs, aggregating soft temporal flow, and moving
    the intervention to feature space while retaining Stage14. Stage19-C
    coverage fallback and Stage19-G graph-temporal-only routing are its two
    internal variants; both preflights failed despite valid activation.

20. **Stage20: agreement-conditional covariance transport.** It branched from
    the valid-but-failing Stage19 family and replaced the learned router with
    frozen class geometry from agreement anchors. The transport was active but
    all three preflight tasks regressed at the 5% shift cap.

21. **Stage21: global agreement-whitened transport.** It is the global,
    label-free-strength-selection variation of Stage20, removing pair
    conditioning and fixed nonzero strength. The mechanism passed on AC/PA/RA,
    but its peak mean was `79.9167`, below matched Stage14 by `0.1567` and the
    DUET subset by `0.0500`. This valid preflight failure closes the current
    geometric feature-transport family before a 12-task run.

## Current Overall Conclusion

The evidence supports four claims:

1. Conflict samples contain useful target information, but static per-sample
   source-vs-CLIP adjudication is unreliable.
2. Graph diffusion and temporal dynamics are diagnostically useful; direct
   graph teacher replacement and fixed graph actions do not reliably improve
   adaptation.
3. The durable gains come from temporal pseudo-label precision and explicit
   target decision-boundary adaptation, not from prompt adjustment.
4. Stage14 is still the correct base. Its best seed reaches `84.9000`, and its
   three-seed peak mean reaches `84.7825`, but the result is not yet stable
   because seed 2022 is `84.6783` and seed-mean standard deviation is `0.1114`.

Stage21 does not change this ranking: its valid AC/PA/RA preflight remained
below both the matched Stage14 subset and the corresponding DUET subset.

Therefore the project has crossed DUET in selected runs and in the aggregate
oracle-peak mean, but it has not yet established a stable improvement over
DUET under the stated all-seed gate.
