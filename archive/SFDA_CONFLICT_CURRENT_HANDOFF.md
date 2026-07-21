# SFDA Conflict Project: Current Handoff

Last updated: 2026-07-22

This is the canonical handoff for starting a new conversation. It consolidates
the current state and points to detailed archives; it is not a new experiment
stage and does not replace the per-stage evidence.

## Repository State

```text
local repository: /Users/stranger/Documents/领域迁移
GitHub: https://github.com/finn001-ai/vlm-sfda-conflict.git
branch: main
latest method implementation commit: 745245f
commit title: Add gated VisDA class intervention routing
```

The local untracked files below belong to the user and must not be deleted,
reverted, or included in unrelated commits:

```text
archive/sfda_conflict_results_summary_2026-07-19/README_CN.md
duet-sfda-main/src/methods/oh/dccl_safe.py
```

Workflow:

1. Modify and test locally.
2. Commit and push to GitHub.
3. The user runs `git pull` and the supplied training script in the cloud.
4. Archive important results, tables, gates, and negative conclusions.

## Research Objective And Constraints

The project studies conflict samples between the task/source model and
CLIP/VLM in VLM-guided source-free domain adaptation.

The target is to exceed the published DUET result, not merely a weaker local
reproduction. Prompt engineering is outside the research scope. Graph methods,
losses, and parameter changes remain allowed when justified by a mechanism or
diagnostic, but do not retry simple `DUET + loss`, fixed graph rules, or the
already failed ACCD/DUET graph-rule variants.

Every new training proposal must include:

- an executable shell script;
- an explicit evaluation and checkpoint-selection protocol;
- a predeclared pass/fail gate when a cheaper preflight is possible;
- a follow-up proposal at the end if the target is not reached;
- archival of both successful and failed conclusions.

Create a new numbered stage only for a substantial method advance. Dataset
transfer, diagnostics, parameter preflights, and failed internal variants stay
under their parent stage.

## Evaluation Protocol

Office-Home task order is:

```text
AC, AP, AR, CA, CP, CR, PA, PC, PR, RA, RC, RP
```

Published DUET Office-Home accuracies:

```text
73.6, 90.4, 91.0, 83.6, 90.7, 90.9,
82.7, 73.7, 91.2, 83.6, 74.0, 91.2
mean = 84.7167
```

The user requested the highest logged accuracy rather than the final
checkpoint when reconstructing past trajectories. This value must always be
named `oracle peak`: it uses target labels for checkpoint selection and is
valid as a diagnostic or an explicitly disclosed oracle protocol, but it is
not label-free SFDA model selection. Preserve both final and peak whenever
logs permit. Do not silently report peak as ordinary final accuracy.

VisDA-C uses mean per-class validation accuracy. The current comparison target
is `91.4`.

## Established Scientific Conclusions

1. Conflict samples contain useful target information.
2. Confidence, prototype, neighbor, and dual-graph per-sample hard selection
   are unreliable.
3. Graph diffusion is diagnostically useful, but direct teacher replacement,
   fixed graph actions, graph priors, and simple graph-loss injection have not
   converted reliably into end-to-end gains.
4. Temporal persistence/precision is useful. The strongest durable mechanism
   is Stage14: reversible temporal precision plus an adapted target classifier
   anchored by the frozen source head.
5. The project should improve Stage14 through mechanism-supported changes,
   not return to prompt tuning or repeat closed graph-rule families.

Detailed Office-Home tables, diagnostics, methods, and lineage are archived in:

```text
archive/sfda_conflict_results_summary_2026-07-19/README.md
archive/sfda_conflict_results_summary_2026-07-19/office_home_accuracy_master.csv
```

## Office-Home Status

Key means:

| Method | Protocol | Mean | Delta vs DUET | Status |
|---|---|---:|---:|---|
| DUET paper | published | 84.7167 | 0.0000 | public reference |
| `both_prior` | final | 84.3033 | -0.4134 | retained component |
| ACCD frozen+persistent | final | 84.3075 | -0.4092 | family closed |
| Stage14 first full run | final | 84.7950 | +0.0783 | single-run pass |
| Stage14 seed 2020 | oracle peak | 84.9000 | +0.1833 | best observed seed |
| Stage14 three-seed mean | oracle peak mean | 84.7825 | +0.0658 | unstable |

Stage14 peak seed means are `84.9000`, `84.7692`, and `84.6783` for seeds
2020, 2021, and 2022. Seed 2022 is below DUET and the seed standard deviation
is `0.1114`, so a stable improvement has not been established. The 84.9 result
was not discarded; it is retained as the best observed oracle peak with the
correct model-selection caveat.

Stage15 through Stage22 did not improve the Stage14 base. Important closed
families include EMA/bounded residual heads, trajectory averaging, class-pair
flow, pair-feature adapters, covariance transport, agreement-whitened
transport, and three-view EM. Read their individual archive READMEs before
proposing a related mechanism.

## VisDA-C Stage14 Transfer

Data layout:

```text
duet-sfda-main/data/VISDA-C/train/<class>/<image>
duet-sfda-main/data/VISDA-C/validation/<class>/<image>
duet-sfda-main/data/VISDA-C/train_list.txt
duet-sfda-main/data/VISDA-C/validation_list.txt
duet-sfda-main/data/VISDA-C/classname.txt
```

Source checkpoints:

```text
duet-sfda-main/source/uda/VISDA-C/T/source_F.pt
duet-sfda-main/source/uda/VISDA-C/T/source_B.pt
duet-sfda-main/source/uda/VISDA-C/T/source_C.pt
```

`train` and `validation` are the two VisDA domains; this is correct. Run
`tools/train_visda_source.sh` only if those three source checkpoints are
missing. It is not an extra required step for every target experiment.

Frozen Stage14 seed-2020 VisDA result:

```text
final mean per-class accuracy = 91.04
oracle peak = 91.07
reference = 91.40
gap = -0.33
```

The peak occurs late and is only `0.03` above final, so the gap is not mainly
caused by selecting the last checkpoint. The mix-0.4 four-cycle preflight
produced `90.16` versus the matched baseline `90.15`, projected `91.08`, and
failed its full-training gate. Do not run the full mix-0.4 experiment.

Detailed VisDA evidence:

```text
archive/sfda_conflict_visda_stage14_transfer_2026-07-21/README.md
archive/sfda_conflict_visda_stage14_transfer_2026-07-21/temporal_precision_head_visda_seed2020_summary.json
archive/sfda_conflict_visda_stage14_transfer_2026-07-21/temporal_precision_head_visda_seed2020_per_class.csv
archive/sfda_conflict_visda_stage14_transfer_2026-07-21/temporal_precision_head_visda_seed2020_dynamics.json
```

## Completed Class-Routing Preflight

The existing Stage14 VisDA diagnostics found class-heterogeneous graph
corrections. Stable teacher corrections are globally useful (`+433` net), but
car contributes most of the positive correction while some classes are weak
or harmful. A label-free proxy gate then found that graph intervention rate
correlates with oracle class gain:

```text
Spearman = 0.580420
top-4 overlap = 3/4
proxy top-4 = car, plant, motorcycle, horse
```

This supported one compute-gated preflight, not a claim of success. Commit
`745245f` added optional class intervention routing. It is disabled by default:

```text
DCCL.GTR_CLASS_ROUTING = False
```

No existing Office-Home or VisDA YAML enables it. Only the new VisDA scripts
turn it on. When enabled, it redistributes existing graph-temporal residual
weights by predicted-class intervention rate and renormalizes them to preserve
the total GTR weight. Therefore existing Office-Home runs and configs are
unchanged. The implementation passed the full local suite (`105` tests), shell
syntax checks, Python compilation, and `git diff --check`.

The four-cycle preflight completed all 16 checkpoints and failed its gate:

```text
final mean per-class accuracy = 89.98
oracle peak = 90.13
matched baseline oracle peak = 90.15
matched improvement = -0.02
projected full oracle peak = 91.05
required full oracle peak = 91.40
decision = fail_full_training_gate
```

The temporal conflict diagnostic still passed (`93.17%` stable coverage and
`+567` net corrections over final CLIP), but it did not convert to end-to-end
gain. Do not run the eight-cycle class-routing script. The variant is closed
and remains inside the Stage14 VisDA transfer archive.

The gate artifact's `next` text says "mix-0.4" because of a stale shared
summarizer default; the metrics and failure decision are valid. The generator
has been corrected. Results are archived under:

```text
archive/sfda_conflict_visda_stage14_transfer_2026-07-21/
```

## Current Pending Experiment

The next predeclared proposal is still a Stage14 internal VisDA preflight. It
sets both temporal stability requirements from two to three cycles:

```text
PL_STABLE_CYCLES=3
GTR_STABLE_CYCLES=3
```

It runs five cycles so the delayed mechanism can be judged on the matched
cycles 4-5 window. The full script is gate-protected. After pulling the latest
main, the next cloud command is only:

```bash
cd /hyperai/home/vlm-sfda-conflict/duet-sfda-main
git pull
bash tools/run_visda_temporal_precision_head_stability3_preflight.sh
```

The gate requires a valid `2/2 -> 3/3` configuration, a passing temporal
dynamics diagnostic, at least `+0.20 pp` candidate improvement in the cycles
4-5 oracle peak, and a projected full oracle peak of at least `91.40` after
adding the baseline post-cycle-5 late gain. Only a decision exactly equal to
`pass_full_training_gate` permits:

```bash
bash tools/run_visda_temporal_precision_head_stability3_seed2020.sh
```

If the preflight fails, archive it and do not run the full job. First compare
the pseudo-label precision/coverage and graph-temporal correction coverage;
only then decide whether one decoupled stability axis (`PL=3,GTR=2` or
`PL=2,GTR=3`) has mechanism support. Do not launch either automatically.

## Instructions For A New Conversation

Start the new conversation with the following message:

```text
继续 SFDA 冲突样本论文项目。仓库是
/Users/stranger/Documents/领域迁移。请先 git pull 获取最新 main；类别干预
路由的实现提交是 745245f。

开始前请先完整阅读：
1. archive/SFDA_CONFLICT_CURRENT_HANDOFF.md
2. archive/sfda_conflict_results_summary_2026-07-19/README.md
3. archive/sfda_conflict_visda_stage14_transfer_2026-07-21/README.md

不要重新尝试已经关闭的 ACCD/DUET 简单图规则变体，也不要走 prompt
调整路线。图方法、loss 和参数可以在机制证据支持下合理使用。每个训练
改进必须提供脚本、执行方案、预声明 gate，并在失败时提出下一方案；只有
重大方法更新才增加 stage。Office-Home 的目标是稳定超过 DUET 84.7167，
VisDA-C 当前参考是 91.4。peak 必须明确标注为 oracle peak。

当前待办：类别干预路由四轮预检已失败并归档，不要启动其八轮训练。先让我
运行 tools/run_visda_temporal_precision_head_stability3_preflight.sh；只有 gate
明确为 pass_full_training_gate 才能运行对应八轮脚本。
```

In the new conversation, attach the latest stability-3 gate/summary/dynamics
files if that preflight has completed. The handoff plus those outputs is
sufficient to continue without replaying the old chat.
