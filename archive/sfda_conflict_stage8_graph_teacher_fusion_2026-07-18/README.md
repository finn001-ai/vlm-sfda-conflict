# Stage 8: Graph-Teacher Fusion Probe

Date: 2026-07-18

## Decision

Graph diffusion is not discarded. The stopped result is specifically the
sample-level action family: hard labels, reversible/persistent resolutions,
teacher abstention, candidate transport, learned source-vs-CLIP selection, and
graph-prior calibration.

The remaining plausible combination is:

```text
both_prior soft teacher + graph diffusion posterior
```

This stage tests that combination before training.

## Method Being Probed

For each target sample:

1. Build the current `both_prior` teacher:

```text
q_both = 0.5 * calibrate(source) + 0.5 * calibrate(CLIP)
```

2. Run dual-space diffusion from high-confidence source/CLIP agreement anchors
   to get a full-class graph posterior `q_graph`.

3. Fuse the two soft teachers with an entropy-adaptive product of experts:

```text
w_i = fusion_strength * (1 - H(q_graph_i) / log(C))
q_fused_i proportional to q_both_i^(1 - w_i) * q_graph_i^w_i
```

Default:

```text
fusion_strength = 0.5
```

High-entropy graph posteriors receive near-zero weight. No sample is promoted,
rejected, abstained, transported, or hard-selected.

## Scientific Gate

Run:

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
bash tools/run_office_home_graph_teacher_fusion_probe.sh
```

The probe writes:

```text
output/uda/office-home/graph_teacher_fusion_probe.json
```

Training is justified only if:

```text
decision = pass_training_gate
```

The default gate requires fused teacher top-1 to beat `both_prior` by at least
`0.05` points on at least two of A->C, P->C, and R->C, with positive mean delta.
Ground-truth labels are used only for reporting this gate.

## Boundary

This is the allowed way to reconsider graph diffusion:

- graph signal is continuous and soft;
- fusion starts from `both_prior`, not raw CLIP/source;
- graph confidence controls graph weight;
- no per-sample graph rule enters training.

If this probe fails, graph diffusion should be treated as useful for diagnosis
but not as the main method mechanism.

## Implementation

```text
duet-sfda-main/tools/analyze_graph_teacher_fusion.py
duet-sfda-main/tools/run_office_home_graph_teacher_fusion_probe.sh
```

## Status

No cloud result has been observed yet.
