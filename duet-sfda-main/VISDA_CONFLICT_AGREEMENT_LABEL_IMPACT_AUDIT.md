# VisDA agreement-anchored label-impact preflight

This preflight asks one bounded question: can the effect that a candidate
label would have on a classifier loss landscape resolve DUET task/CLIP
conflicts more reliably than choosing by prediction confidence?

It is motivated by the ICML 2022 paper [Partial Label Learning via Label
Influence Function](https://proceedings.mlr.press/v162/gong22c.html), which
argues that candidate labels should be compared by how they change a model,
not only by their current loss or confidence. The audit borrows that
information principle; it does not claim to reproduce PLL-IF's optimizer or
its exact inverse Hessian.

## Why this is not a repeated selector

Previous VisDA audits compared task/CLIP confidence, source prototypes,
nearest neighbors, graph propagation, joint-evidence GMM likelihoods,
attributes, spatial occlusion, and temporal velocity. This preflight uses none
of those scores. Its reference is the parameter-loss geometry induced by the
6,777 cycle-1 DUET agreements:

1. For each agreement, form the frozen linear-head CE gradient
   `(p - one_hot(y)) outer [feature, 1]` using the shared task/CLIP pseudo
   label.
2. Estimate one global diagonal empirical Fisher from all agreements.
3. Estimate a mean reference gradient separately for every agreement pseudo
   class.
4. For each conflict and each member of `task-top2 union CLIP-top2`, compute
   the first-order improvement that the candidate's hypothetical CE update
   would produce on agreement references of that pseudo class.
5. Choose the maximum-impact candidate. No threshold or class route is fitted.

The frozen classifier head is only a diagnostic parameter space. DUET does
not train that head, so even a passing result must next survive an exact
trainable-parameter audit before any proxy run can be considered.

## Label isolation

Phase 1 reads only task probabilities, CLIP probabilities, task features,
predicted labels, admission masks, and sample indices from the already locked
pre-cycle-1 snapshot. Although that source snapshot contains a diagnostic
label array, the script does not access the array before writing:

- `visda_conflict_agreement_label_impact_label_free.npz`
- `visda_conflict_agreement_label_impact_signal_lock.json`

Only after the lock exists does phase 2 parse the proxy target list and read
the embedded labels. Those outputs are explicitly named `oracle_diagnostic`.

## Fixed rejection gate

Every check must pass:

- input and hash contracts are valid;
- agreement reference oracle accuracy is at least 90%;
- the minimum decision stability across two alternating per-class reference
  halves is at least 90%;
- top-2 union oracle coverage is at least 90%, and every class reaches 85%;
- conflict accuracy beats the best of fixed task, fixed CLIP, confidence
  choice, arithmetic fusion, and RMS fusion by at least 1.0 percentage point;
- the paired bootstrap 95% CI lower bound against that best baseline is
  positive, and every matched baseline is beaten;
- individual car and truck regression is no worse than 0.5 point, their mean
  and the other-ten mean are nonnegative;
- maximum full-proxy predicted class-mass shift versus fixed CLIP is at most
  1.0 point.

`PASS_AGREEMENT_LABEL_IMPACT_PREFLIGHT` authorizes only design review of one
exact trainable-parameter audit. It authorizes no GPU, proxy, or full training.

## Run

From `duet-sfda-main`:

```bash
bash tools/run_visda_conflict_agreement_label_impact_audit.sh
```

The required pre-cycle-1 snapshot already exists if the earlier cycle-2
conflict-memory audit completed. Expected runtime is seconds to a few minutes
on CPU with no image or checkpoint load.
