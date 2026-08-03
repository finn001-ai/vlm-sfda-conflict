# VisDA Patch Pair-Neutralization Impact Audit

## Evidence table

| Evidence | Result | Provenance |
|---|---:|---|
| Locked selected conflicts | `613` | Prior label-free suppression signal |
| Feature first-order delta vs original DUET | `-0.000245076`; CI `[-0.0003264731232924905, -0.0001595465880641344]` | Oracle diagnostic after new lock |
| Feature first-order delta vs full KL suppression | `0.000805723`; CI `[0.0006653447723931095, 0.0009441852817513093]` | Oracle diagnostic after new lock |
| Maximum full-target class-mass shift | `0.166774` pp | Label-free target probabilities |

Decision: **REJECT**

The candidate equalizes only the task/CLIP candidate pair in the CLIP
soft target. It preserves pair mass and every non-pair probability; no
hard task label, mask change, fitted threshold, or extra loss is added.

## Gate

- input_contract_valid: `True`
- source_kl_suppression_reject_preserved: `True`
- heldout_selector_gate_passed: `True`
- selected_coverage_between_2_and_10pct: `True`
- recovered_clip_target_error_at_most_5e_6: `True`
- nonpair_target_probability_unchanged: `True`
- candidate_pair_mass_preserved: `True`
- output_first_order_gain_vs_duet_ci_lower_positive: `False`
- feature_first_order_gain_vs_duet_ci_lower_positive: `False`
- output_first_order_gain_vs_suppression_ci_lower_positive: `True`
- feature_first_order_gain_vs_suppression_ci_lower_positive: `True`
- output_negative_burden_not_worse: `True`
- feature_negative_burden_not_worse: `True`
- feature_helpful_retention_at_least_99pct: `False`
- feature_mean_norm_inflation_at_most_1_5x: `True`
- max_full_target_class_mass_shift_at_most_1pp: `True`
- class_macro_feature_first_order_delta_positive: `False`
- car_feature_first_order_delta_nonnegative: `True`
- person_feature_first_order_delta_nonnegative: `False`
- truck_feature_first_order_delta_nonnegative: `False`
- other_nine_feature_first_order_delta_nonnegative: `False`

Even PASS authorizes one exact no-update parameter audit only, not a
proxy run or training.
