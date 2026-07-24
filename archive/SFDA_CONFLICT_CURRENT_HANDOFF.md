# SFDA Conflict Project: Current Handoff

Last updated: 2026-07-24

This is the canonical handoff for starting a new conversation. It consolidates
the current state and points to detailed archives; it is not a new experiment
stage and does not replace the per-stage evidence.

## Repository State

```text
local repository: /Users/stranger/Documents/领域迁移
GitHub: https://github.com/finn001-ai/vlm-sfda-conflict.git
branch: main
latest method implementation: inspect the current `main` tip with `git log -1`
```

Preserve all unrelated local changes and untracked files if the worktree is
not clean.

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

## 2026-07-23 VisDA Proxy Loss Audit

The full-data five-cycle stability-3 candidate completed and failed its matched
gate. It also mixed transferred Office-Home parameters
(`CALIB_POWER=0.8`, `TARGET_HEAD_MIX=0.5`) with `PL/GTR=3/3`; do not launch its
eight-cycle job and do not repeat the coupled 3/3 setting.

Commit `4d55b89` added deterministic, class-proportional 25% VisDA adaptation
lists while preserving evaluation on the full target set. The matched proxy
reference is:

```text
GTR=0; final macro=87.83; car/person/truck mean=73.79
```

Completed proxy conclusions:

```text
GTR 0/0.05/0.10 changes car versus truck but leaves macro within 0.02 pp.
CLS=0.5 + CON=0.3 + GTR=0.05 ends at 87.96; early gain decays to +0.13.
weak-teacher consistency stop-gradient ends at 87.66 and fails.
KL_PAR=0.3 ends at 87.61; all 16 checkpoints are below the KL=0.4 reference.
```

Commit `faa7bc6` added an opt-in consistency stop-gradient flag and task-loss
magnitude diagnostics. The stop-gradient default is `False`, so existing
configs preserve legacy behavior. The KL 0.3 audit shows that the large CLIP
KL loss-value share is not evidence that it should be weakened globally.
Reducing it raises car by 0.37 pp but lowers truck by 1.43 pp, hard mean by
0.35 pp, other-nine mean by 0.17 pp, and final macro by 0.22 pp.

Full records and raw logs:

```text
archive/sfda_conflict_visda_proxy_loss_audit_2026-07-23/
```

There is no pending DCCL scalar run. Do not automatically test KL 0.5. The
KL 0.3 temporal NPZ audit is complete: the graph teacher improves car by
7.62 pp over CLIP but lowers truck by 9.81 pp, and the best simple label-free
signal predicts beneficial versus harmful graph top-1 changes with ROC AUC
only 0.587. A fixed margin-and-stability gate still exchanges car `+4.35 pp`
for truck `-3.82 pp`, so confidence-conditioned KL/graph routing is rejected.

The same-environment 25% proxy original PLMatch control is complete. It uses
13,847 images for adaptation, all 55,388 images for evaluation, and all 16
planned checkpoints. Final and oracle peak coincide at `87.93`. This is only
`+0.10 pp` above DCCL P1 `87.83`, inside the predeclared `+/-0.20 pp` tie
margin, so the decision is `matched_within_margin`, not PLMatch superiority.
The completed DCCL combined run `87.96` is similarly only `+0.03 pp` above
PLMatch.

At class level, PLMatch versus P1 gains car `+5.06 pp` and person `+3.50 pp`
but loses truck `-2.60 pp`. Its car/person/truck mean is `+1.99 pp`, while its
other-nine mean is `-0.52 pp`. Current DCCL therefore has no demonstrated
single-seed macro advantage over matched PLMatch; the difference remains a
class redistribution. The complete raw terminal record, JSONs, comparison,
and checksums are archived in the latest proxy-loss audit directory.

The complete-data, eight-cycle released `plmatch.py` control is now also
complete. This entrypoint is the full official DUET pipeline, not only an
independent PLMatch loss baseline: it runs dual-perspective pseudo labels,
TMI/DVO CLIP optimization, and PLMatch target training. Final accuracy is
`91.50` and oracle peak is `91.52`, consistent with the paper's one-decimal
`91.4` result but above DCCL Stage14 final `91.04` by `0.46 pp`.

The local effective YAML matches official code commit
`bd2644bf6a115ddb4bb64ec94fb121841c5783de`. The only local PLMatch changes
are an optional proxy list and output-name dispatch; with the full-run override
empty, both are behaviorally inert. A paper/code discrepancy remains:
Appendix Table 4 states VisDA momentum `0.999`, while the official YAML and
code use `ACTIVE.BETA=0.99`. The run follows released code. Full evidence is in:

```text
archive/sfda_conflict_visda_full_duet_control_2026-07-24/
```

This full control supersedes the proxy tie for the full-data method decision:
current DCCL is below its matched official DUET base. Do not claim a VisDA
improvement for DCCL in its current form.

## Dual-View Inheritance And Precision-Coverage Diagnosis

DCCL does not lack the DUET dual-perspective pseudo-label mechanism. It retains
the task/source-model view, CLIP image-text view, agreement test, probability
mixing, CLIP update, and the three base task losses. The dual-view construction
is inherited and must not be claimed as a DCCL novelty.

The effective Stage14 VisDA configuration modifies that base using:

```text
CALIB_MODE=both_prior
CALIB_POWER=0.5
PL_MEMORY=stable
PL_STABLE_CYCLES=2
PL_STABLE_MEMORY=reversible
TARGET_HEAD_ADAPT=True
TARGET_HEAD_MIX=0.3
TARGET_HEAD_START_CYCLE=1
GTR_PAR=0.05
```

At the matched pseudo-label refresh before cycle-4 training:

| Metric | Official DUET | DCCL Stage14 | DCCL - DUET |
|---|---:|---:|---:|
| global mixed-output accuracy | 88.94% | 87.98% | -0.96 pp |
| selected pseudo-label count | 53,372 | 47,393 | -5,979 |
| selected mixed-label accuracy | 90.42% | 93.46% | +3.04 pp |
| selected coverage | 96.36% | 85.57% | -10.80 pp |

Thus DCCL's combined intervention produces cleaner selected labels at the cost
of substantially lower coverage, while also lowering the global mixed-output
accuracy. This explains the central precision-coverage failure mode, but does
not isolate one causal component because calibration, stable memory, target
head, and GTR are coupled in the completed run.

The next method direction is baseline-preserving conflict correction: retain
the released DUET probabilities, monotonic agreement memory, and base losses;
intervene only where the task and CLIP views conflict. Any new proposal must
measure selected-label precision and coverage together and pass a matched 25%
proxy gate before a full eight-cycle job.

Detailed audit:

```text
archive/sfda_conflict_visda_full_duet_control_2026-07-24/
  dual_view_precision_coverage_audit.md
```

## Completed VisDA Structural Ablation

A compute-gated structural ablation isolated the two interventions that
activate when the VisDA gap first appears: stable/reversible pseudo-label
memory and the adapted target head. It holds `both_prior`, all three base
losses, and the deterministic 25% proxy fixed, disables GTR, and runs:

```text
V1 = monotonic memory + target head
V2 = stable/reversible memory + no target head
V3 = monotonic memory + no target head
```

V0 (`stable + head`, GTR=0, final `87.83`) and the matched official-DUET proxy
control (`87.93`) are archived references. The completed results are:

| Variant | Final | Delta vs DUET | Hard mean delta | Decision |
|---|---:|---:|---:|---|
| V1 monotonic + head | 88.03 | +0.10 | -1.6700 | fail |
| V2 stable + no head | 88.01 | +0.08 | -0.6867 | fail |
| V3 monotonic + no head | 88.18 | +0.25 | -0.3633 | fail |

Monotonic memory improves final macro by `0.17-0.20 pp` relative to stable
memory, while the target head lowers it by `0.15-0.18 pp`. Both interventions
are harmful on this proxy. V3 ranks first but obtains its gain through
car/person losses (`-4.15/-1.18 pp`) and a truck gain (`+4.24 pp`), so it
fails the predeclared no-compensation gate. Do not run any full-data
structural variant.

The originally generated gate parsed the DUET selected source-label precision
(`89.44%`) instead of its selected mixed-label precision (`90.36%`). The
summarizer now records both. This diagnostic correction does not change any
accuracy, class gate, or the `fail_proxy_gate` decision.

The full contract and result are in:

```text
VISDA_STRUCTURAL_ABLATION_STEP.md
archive/sfda_conflict_visda_structural_ablation_2026-07-24/
```

The structural transfer family is closed on VisDA. Keep the released DUET path
as the safety baseline. A subsequent method must preserve that base and add a
genuinely independent conflict signal; further recombination of the existing
task, CLIP, calibration, and graph evidence is not supported.

## Stage23 Reciprocal Conflict Boundary Learning

Stage23 is implemented and awaits cloud preflight. It returns to the released
DUET host rather than stacking more components on Stage14. This relationship
must be described honestly: DUET's dual-view mechanism is inherited; the new
contribution is a sample-dependent antisymmetric boundary head for persistent
unordered task/CLIP conflict pairs.

The method uses two-sided temporally stable agreement anchors, a pair-balanced
residual-margin loss, conflict-only weak/strong pair-margin consistency, and a
preservation loss away from active pair evidence. Discovery always uses raw,
uncorrected task predictions, and gradients are isolated between the DUET
backbone losses and boundary-head losses. It adds no DINO or other third visual
module.

Run:

```bash
cd duet-sfda-main
bash tools/run_reciprocal_boundary_preflight.sh
```

Only if the joint JSON gate passes, run:

```bash
bash tools/run_reciprocal_boundary_seed2020_full.sh
```

The proxy gate requires not only a macro gain, but also nonnegative deltas on
car, person, and truck individually, plus verified nonzero mechanism action.
The complete design, fixed hyperparameters, scripts, and interpretation
contract are in:

```text
archive/sfda_conflict_stage23_reciprocal_boundary_2026-07-24/README.md
```

## Instructions For A New Conversation

Start the new conversation with the following message:

```text
继续 SFDA 冲突样本论文项目。仓库是
/Users/stranger/Documents/领域迁移。请先 git pull 获取最新 main。

开始前请先完整阅读：
1. archive/SFDA_CONFLICT_CURRENT_HANDOFF.md
2. archive/sfda_conflict_results_summary_2026-07-19/README.md
3. archive/sfda_conflict_visda_stage14_transfer_2026-07-21/README.md
4. archive/sfda_conflict_visda_proxy_loss_audit_2026-07-23/README.md
5. archive/sfda_conflict_visda_full_duet_control_2026-07-24/README.md
6. archive/sfda_conflict_stage23_reciprocal_boundary_2026-07-24/README.md

不要重新尝试已经关闭的 ACCD/DUET 简单图规则变体，也不要走 prompt
调整路线。图方法、loss 和参数可以在机制证据支持下合理使用。每个训练
改进必须提供脚本、执行方案、预声明 gate，并在失败时提出下一方案；只有
重大方法更新才增加 stage。Office-Home 的目标是稳定超过 DUET 84.7167，
VisDA-C 当前参考是 91.4。peak 必须明确标注为 oracle peak。

当前待办：类别干预路由、稳定性3/3、GTR权重、组合CLS/CON/GTR、一致性
stop-gradient 和 KL_PAR=0.3 均已失败并归档，不运行对应八轮任务，也不盲测
KL=0.5。KL0.3时序NPZ零训练诊断也已完成，简单可靠性路由无法避免car/truck
补偿。25%代理对照虽持平，但完整数据8轮官方DUET代码路径最终91.50、
peak 91.52，分别高于DCCL最终91.04和peak 91.07约0.46/0.45pp；因此完整
数据结论是当前DCCL低于其匹配官方基线。该入口虽命名plmatch，实际运行完整
DUET（双视角伪标签、TMI/DVO、PLMatch）。本地有效参数与官方发布YAML一致；
论文附录VisDA动量0.999与官方代码BETA=0.99存在作者侧不一致，本次遵循代码。
完整审计见archive/sfda_conflict_visda_full_duet_control_2026-07-24/。

DCCL本身也完整保留DUET的任务模型+CLIP双视角伪标签；双视角属于继承机制，
不能写成DCCL创新。cycle-4匹配诊断显示DCCL将选中伪标签准确率从90.42%
提高到93.46%，但选中量从53,372降到47,393，覆盖率减少约10.80pp，同时
全域混合预测从88.94%降到87.98%。这说明当前组合提升纯度却过度损失覆盖率。
下一步应保留官方DUET双视角、单调伪标签和基础损失作为默认路径，只在两个
视角冲突样本上施加DCCL修正；具体审计见同目录
dual_view_precision_coverage_audit.md。Stage23互惠冲突边界学习已按此原则
实现：无DINO或第三视觉模块，先验证关闭边界头的dccl宿主与官方DUET等价，
再用无序冲突对、双侧稳定锚点、样本相关反对称边界头和三项隔离损失处理边界。
当前只运行tools/run_reciprocal_boundary_preflight.sh；联合gate通过后才运行
tools/run_reciprocal_boundary_seed2020_full.sh，不得跳过门控。
```

The KL `0.3` temporal NPZ files are archived. The handoff and proxy-loss audit
are sufficient to continue without replaying the old chat.
