# Stage 24: Boundary-Flip DUET

Date: 2026-07-28

Status: implementation complete; matched Office-Home preflight pending.

## Motivation

Stage14 is the strongest validated Office-Home host, but its improvement over
the DUET paper mean is small and adaptation-seed sensitive. Earlier global
promotion, topology-only resolution, class balancing, prototype correction,
and additional target-head variants have already been tested and archived.
This stage does not overwrite those experiments.

The new hypothesis is narrower: decision-boundary samples whose prediction
changes under a label-free class-frequency correction can be useful, but only
when the alternative is supported by an existing DUET view and forms one
stable, semantically plausible temporal transition.

## Recent Work Used

- [Dynamic Logits Adjustment and Exploration for Test-Time Adaptation in
  Vision-Language Models, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Dynamic_Logits_Adjustment_and_Exploration_for_Test-Time_Adaptation_in_Vision_CVPR_2026_paper.html)
  motivates class-statistics logit adjustment and consistency-gated boundary
  exploration.
- [Unveiling Non-Stationary Predictions for Test-Time Adaptation, ICML
  2025](https://icml.cc/virtual/2025/poster/44953) motivates treating a stable
  early-to-late prediction transition differently from oscillation.

The implementation is an adaptation of these ideas to source-free DUET; it is
not claimed to reproduce either method. A formal novelty review remains
required before a paper claim.

## Method Contract

1. Estimate class frequency and mean confidence only from current stable
   source/CLIP agreement anchors.
2. Apply an additive frequency/confidence penalty to the log of the current
   DUET mixed probability. The adjusted prediction is a proposal only.
3. Keep a proposal only if it flips the current and initial label, is one of
   the current source/CLIP alternatives, exceeds the confidence/margin gates,
   and passes a CLIP-text semantic gate.
4. Accept the same transition only after two consecutive refresh cycles.
   Any interruption or changed alternative is recorded as oscillation and is
   rejected by the default zero-switch policy.
5. Limit supervision independently for each ordered early/late class pair.
6. Train with low-weight late-label CE plus complementary suppression of the
   early label. All remaining samples retain the unchanged Stage14 losses.

No target ground-truth label participates in proposal generation or training.
The logged active accuracy is explicitly an oracle diagnostic only.

## Implemented Files

```text
duet-sfda-main/src/utils/boundary_flip.py
duet-sfda-main/src/methods/oh/boundary_flip_duet.py
duet-sfda-main/cfgs/office-home/boundary_flip_duet.yaml
duet-sfda-main/tools/run_office_home_boundary_flip_duet_preflight.sh
duet-sfda-main/tools/analyze_boundary_flip_duet.py
duet-sfda-main/tests/test_boundary_flip_duet.py
```

The shared dispatcher, configuration registry, DCCL training host, and method
dispatch test are extended behind `BOUNDARY_FLIP.ENABLED=False`; old method
defaults remain unchanged.

## Execution

Matched Stage14 control plus Boundary-Flip DUET on AC, PC, and RC:

```bash
cd /openbayes/home/vlm-sfda-conflict/duet-sfda-main
bash tools/run_office_home_boundary_flip_duet_preflight.sh
```

Candidate only when a matched seed-2020 control is already available:

```bash
RUN_CONTROL=0 bash tools/run_office_home_boundary_flip_duet_preflight.sh
```

Outputs:

```text
output/uda/office-home/boundary_flip_duet_preflight_seed2020/control_accuracy.csv
output/uda/office-home/boundary_flip_duet_preflight_seed2020/candidate_accuracy.csv
output/uda/office-home/boundary_flip_duet_preflight_seed2020/gate.json
```

## Predeclared Gate

The preflight passes only when:

```text
the flip mechanism is active on AC, PC, and RC
mean final delta versus the matched Stage14 control is at least +0.20 pp
at least 2/3 tasks win or tie
the worst task delta is at least -0.30 pp
```

Passing authorizes a fixed-hyperparameter all-12 Office-Home run. Failure with
an active mechanism closes this exact configuration; target labels must not be
used to tune the gates.
