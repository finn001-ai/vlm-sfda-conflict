# Stage 23: Reciprocal Conflict Boundary Learning

Date: 2026-07-24

Status: implementation complete; cloud preflight pending.

## Why This Is A New Stage

The completed VisDA structural ablation closed the previous Stage14 stack:
stable/reversible memory and the cloned blend target head both hurt, while
removing them restored coverage but still exchanged car against truck. The
failure is not only a scalar-loss problem. A directed global decision for a
pair such as car/truck is structurally invalid: within the observed
`task=car, CLIP=truck` group, 72 samples are true car and 73 are true truck.

Stage23 therefore does not add another global winner, graph teacher, source
rescue, target-head copy, or third visual encoder. It learns a
sample-dependent boundary correction for persistent *unordered* conflict
pairs.

## Relationship To DUET And DCCL

This point must remain explicit in the paper:

- The released DUET pipeline is the host and matched baseline. Its source/task
  view, CLIP view, agreement pseudo labels, TMI/DVO update, PLMatch losses, and
  monotonic memory are inherited rather than claimed as new.
- The failed Stage14 DCCL additions are removed: no prior calibration, stable
  memory, cloned target head, graph teacher, GTR, covariance transport, or
  third-view EM is active.
- The Stage23 contribution is the reciprocal boundary mechanism and its three
  coupled objectives. It is materially larger than a DUET scalar change, but
  it must still be described as a method built on the DUET training host.
- No DINO, new CLIP, or other third-party visual module is introduced. The
  existing CLIP branch already required by DUET is the only VLM branch.

For the current project lineage, the method can be named
**DCCL-v2 / Reciprocal Conflict Boundary Learning (RCBL)**. A final paper name
should be frozen only after the full gates pass.

## Method

### 1. Persistent reciprocal pair discovery

At each pseudo-label refresh, use the *uncorrected* task prediction and current
DUET CLIP prediction. Count an unordered pair `{a,b}` only if the two views
disagree. A pair is eligible when it has at least five conflicts in each of
two refresh cycles.

This deliberately stores `{a,b}`, not `a -> b`: neither view is declared the
dataset-level winner.

### 2. Two-sided stable anchors

A sample is an anchor for class `c` only when the uncorrected task view and
CLIP agree on `c` for two consecutive refresh cycles. A pair activates only
if both classes have at least eight anchors. Thus a car/truck boundary is
learned from reliable target-car and target-truck evidence, not from a global
majority among ambiguous samples.

Pair discovery and anchor memory always use uncorrected task predictions.
Consequently, the learned boundary cannot create its own future supervision.

### 3. Sample-dependent antisymmetric boundary head

A small internal MLP maps the existing task bottleneck feature to one signed
coefficient per active pair. For pair `{a,b}`, it adds `+delta` to class `a`
and `-delta` to class `b`. The correction:

- is exactly zero at initialization;
- is bounded by the sample's detached base-logit scale;
- is gated by the base probability mass on the two classes;
- conserves the pair's total logit mass;
- normalizes overlapping pairs so one high-degree class cannot accumulate an
  unbounded correction.

This is a target-domain classifier component, not a visual foundation model.

### 4. Pair-balanced residual margin

The head is trained to give opposite residual-margin signs to the two stable
anchor sides. Each side contributes one half regardless of class frequency,
and minibatch sums are normalized by the global anchor counts. The loss acts
on the learned residual margin rather than the already-confident source
margin; otherwise the zero-initialized head could remain effectively idle.

### 5. Conflict margin consistency and preservation

For active conflict samples, weak and strong views receive a Smooth-L1 loss on
their normalized pair margins. Samples that are neither active pair conflicts
nor active-pair anchors receive a preservation penalty on the normalized
residual. This concentrates capacity on the learned boundary while protecting
the released DUET path elsewhere.

### 6. Gradient isolation

The original DUET losses consume the corrected logits but see a detached
boundary residual. The boundary objectives consume detached task features and
base logits. Therefore:

- generic DUET losses cannot silently train the new head;
- boundary losses cannot silently rewrite the task backbone;
- any head activation reported by diagnostics comes from the declared
  boundary objectives.

## Fixed Hyperparameters

```text
start cycle = 2 (zero-based config value 1)
max unordered pairs = 16
minimum conflicts per cycle = 5
stable cycles = 2
minimum anchors per side = 8
hidden dimension = 128
maximum relative logit shift = 0.5
boundary LR multiplier = 1.0
residual margin = 0.5
margin / consistency / preservation weights = 0.10 / 0.05 / 0.05
```

The same boundary hyperparameters are used for VisDA-C and Office-Home. Only
the released dataset-specific DUET optimizer and training settings differ.

## Cloud Execution

Run the matched preflight first:

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_reciprocal_boundary_preflight.sh
```

This runs:

1. VisDA 25% adaptation / full-evaluation DUET control;
2. the same VisDA host path with the boundary module disabled;
3. VisDA margin-only ablation;
4. VisDA margin + consistency ablation;
5. VisDA full RCBL;
6. matched official DUET, boundary-disabled host, and full RCBL on
   Office-Home AC, PC, and RC.

The joint gate uses final checkpoints only. Oracle peaks are retained only as
diagnostics.

If and only if
`output/uda/reciprocal_boundary_preflight_gate.json` says
`pass_reciprocal_boundary_preflight`, run:

```bash
bash tools/run_reciprocal_boundary_seed2020_full.sh
```

The full script reruns a matched full-data VisDA DUET control and candidate,
then completes all 12 matched Office-Home tasks. It reuses completed AC/PC/RC
preflight logs because their budgets and configurations are identical.

## Predeclared Gates

VisDA proxy full-method gate:

```text
boundary-disabled host final within 0.10 pp of official DUET
final macro delta vs matched DUET >= +0.20 pp
car/person/truck mean delta >= +0.20 pp
each of car/person/truck delta >= 0.00 pp
other-nine mean regression <= 0.10 pp
pairs freeze, the head updates, probability correction is nonzero,
and every enabled boundary loss is active
```

Office-Home AC/PC/RC gate:

```text
each boundary-disabled host final within 0.15 pp of official DUET
mean final delta vs matched DUET >= +0.20 pp
at least 2/3 tasks win or tie
worst task delta >= -0.30 pp
mechanism valid on all three tasks
```

Seed-2020 full gates retain the VisDA class constraints. The complete
Office-Home gate requires mean final delta at least `+0.20 pp`, at least 7/12
task wins/ties, and worst task delta at least `-0.50 pp`.

## Interpretation Contract

- Passing the proxy gate authorizes full training; it is not a paper result.
- Passing one seed authorizes a fixed-hyperparameter seed sweep; it is not a
  stability claim.
- Failure with an inactive head is an implementation/mechanism failure.
- Failure with a valid active head closes this exact RCBL formulation. Do not
  tune against target labels.
- A macro gain that sacrifices any of car/person/truck fails the VisDA gate,
  even if the overall number is higher.

## Implemented Files

```text
duet-sfda-main/src/utils/reciprocal_boundary.py
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/cfgs/visda/reciprocal_boundary.yaml
duet-sfda-main/cfgs/office-home/reciprocal_boundary.yaml
duet-sfda-main/tools/run_reciprocal_boundary_preflight.sh
duet-sfda-main/tools/run_visda_reciprocal_boundary_proxy25.sh
duet-sfda-main/tools/run_office_home_reciprocal_boundary_preflight.sh
duet-sfda-main/tools/run_reciprocal_boundary_seed2020_full.sh
duet-sfda-main/tools/summarize_reciprocal_boundary_preflight.py
```
