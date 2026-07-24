# VisDA Stage14 Gap Audit

This is not a Stage24 method. Stage14 remains the strongest DCCL version and is
a positive Office-Home result. This audit only diagnoses why its gain reverses
on VisDA-C. It compares Stage14 with a matched official DUET control and does
not add a loss, module, visual encoder, or training signal.

The archived three-seed Office-Home result has a Stage14 mean of 84.7825 versus
DUET's 84.7167. That is a positive mean result, although the old strict
all-seed stability gate failed because seed 2022 was 0.0383 points below DUET.
Office-Home is therefore retained as a positive control rather than treated as
evidence against Stage14.

## Run

From the repository root on the GPU server:

```bash
git pull
bash tools/run_visda_stage14_gap_audit.sh
```

The script uses the deterministic VisDA-C 25% adaptation subset with seed 2020,
while final accuracy, confusion matrices, and features are evaluated on the
full validation set. It runs:

1. official DUET with diagnostic saving enabled;
2. the current Stage14 `temporal_precision_head` configuration;
3. a matched post-hoc analysis using the same sample order.

Diagnostic saving is side-effect free: it does not change pseudo labels,
losses, optimizers, or model logits.

## Primary outputs

All reports are written under:

```text
output/uda/VISDA-C/stage14_visda_gap_audit/
```

- `stage14_visda_gap_summary.json`: automatic conclusion for the mentor's
  three checks and the next experiment.
- `per_class.csv`: DUET/Stage14 class accuracy, source-head effect, feature
  dispersion, pseudo-label precision, coverage, and selected counts.
- `pair_confusion_geometry.csv`: symmetric class-pair confusion changes,
  prototype distances, and Fisher separability.
- `directional_confusions.csv`: every true-class to predicted-class error
  change.
- `duet_confusion.csv` and `stage14_confusion.csv`: raw and row-normalized
  confusion matrices.
- `stage14_visda_feature_tsne.png` and `stage14_visda_feature_tsne.csv`: common-coordinate
  t-SNE for visual inspection. Quantitative conclusions use prototype/Fisher
  metrics rather than t-SNE.

## Interpretation

The summary distinguishes four mechanisms:

- `effective_head_harms_macro_accuracy`: remove or source-anchor the target
  head before changing the feature objective.
- `material_distribution_shift`: balance classwise pseudo-label exposure
  before adding a boundary module.
- `localized_pair_compression`: replicate the discovered pairs across seeds;
  do not hard-code car/truck.
- `global_geometry_compression`: keep the Office-Home positive control frozen
  and isolate the responsible Stage14 component only on VisDA.

The true labels are used only for post-hoc diagnosis. None of the reported
hard-class identities may be used directly as an unsupervised training oracle.

Parameter tuning comes after this audit and is restricted to the diagnosed
component. Candidate selection must use label-free statistics such as temporal
agreement, pseudo-label coverage, class-distribution drift, and source-head
logit drift. Per-class target accuracy and the true-label feature metrics in
this report are diagnostic evidence, not hyperparameter-selection criteria.
