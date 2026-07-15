# Conversation Stage Summary

## Background

The project is about Source-Free Domain Adaptation (SFDA), with a one-month
pressure to produce a defensible paper. Earlier ideas around generic
foundation-model guidance were considered too broad and unstable.

The current narrowed direction is:

> Learning from Conflict Samples in VLM-Guided Source-Free Domain Adaptation

## Research Question

Main question:

> How can VLM-guided SFDA identify and exploit useful conflict samples when the
> source/task model and CLIP/VLM disagree on target samples?

Short version:

> Can disagreement between the source-adapted model and VLM be used as a
> positive signal rather than only treated as noise?

## Core Claim

Initial claim:

> A subset of source/VLM conflict samples is informative under domain shift, and
> exploiting them with conflict-aware reliability improves SFDA over
> agreement-only or naive fusion strategies.

Stronger claim after Office-Home diagnostics:

> Across all 12 Office-Home transfer tasks, source/VLM conflicts account for
> about 42% of target samples. Importantly, about 69.5% of conflict samples
> contain a correct prediction from either the source model or CLIP, indicating
> that disagreement is not merely noise but an underexploited adaptation signal.

## Why DUET Was Used First

DUET is a good diagnostic platform because it explicitly uses agreement between
the target/task model and CLIP to assign pseudo-labels. This makes
disagreement/conflict samples a natural and visible blind spot.

DIFO++ remains important as latest related work and possibly a later baseline,
but it has more intertwined components, making it harder to isolate the effect
of a conflict-aware module within a one-month schedule.

## Stage-1 Objective

The first objective was:

> Before designing a method, verify whether conflict samples actually contain
> useful signal.

This stage ran source-vs-CLIP diagnostics on all 12 Office-Home transfer tasks.

For each target sample, the diagnostic script exported:

- image path
- true label
- source prediction and confidence
- CLIP prediction and confidence
- agreement/conflict status
- correctness cases
- source top-5 and CLIP top-5

## Current Status

The problem is verified on Office-Home:

- Conflict samples are common.
- Most conflict samples contain a correct prediction from either source or CLIP.
- CLIP often corrects source mistakes, but CLIP also misleads non-trivially.
- Harmful conflicts, where both are wrong, also exist and must be rejected or
  delayed.

The next stage should move from diagnosis to a minimal conflict-aware
reliability module.
