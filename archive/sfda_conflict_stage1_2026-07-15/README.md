# SFDA Conflict Samples Stage-1 Archive

Date: 2026-07-15

This folder archives the first diagnostic stage for the project:

**Learning from Conflict Samples in VLM-Guided Source-Free Domain Adaptation**

The goal of this stage was not to propose the final method, but to verify
whether the research problem is real:

> In VLM-guided SFDA, are source/VLM disagreement samples merely noisy, or do
> they contain useful target-domain supervision signals?

## Files

- `conversation_stage_summary.md`  
  Condensed record of the direction, decisions, and current research question.

- `diagnostic_analysis.md`  
  Paper-useful interpretation of the 12 Office-Home diagnostic results.

- `office_home_12pair_diagnostics.csv`  
  Structured per-task diagnostic table parsed from the cloud output.

- `office_home_12pair_summary.json`  
  Macro and sample-weighted summary statistics.

- `cloud_commands.md`  
  Commands used or recommended for cloud-side reproduction.

- `next_steps.md`  
  Recommended next experiments and minimal method path.

- `raw/office_home_12pair_raw_res.txt`  
  Raw cloud output copied from `/Users/stranger/Downloads/res.txt`.

## Core Finding

Across all 12 Office-Home transfer tasks, source/VLM conflicts are common and
often useful:

- Macro-average conflict rate: **42.57%**
- Sample-weighted conflict rate: **42.01%**
- Macro-average useful conflict rate among conflicts: **72.14%**
- Sample-weighted useful conflict rate among conflicts: **69.49%**

This supports the paper's central premise: disagreement samples should not be
treated only as noise or ignored wholesale.
