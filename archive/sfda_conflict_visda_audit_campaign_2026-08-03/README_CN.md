# VisDA-C DUET 冲突审计总归档（2026-08-03）

状态：**本轮候选全部关闭；保留原始 DUET 作为当前安全基线。**

## 1. 归档范围

本目录覆盖 CT-DUET v1 已归档并从主线移除之后的完整研究链：

```text
起点（不含）：748e716084dc367248454fec8042dff7fdebd063
终点（包含）：f661d900d8f2a8c2cfea6ad6178aa1f6f1104f45
日期范围：2026-08-01 至 2026-08-03
提交数量：47
累计变更：167 个路径，约 40,238 行补丁
```

起点提交 `748e716` 已完成 CT-DUET v1 的归档和主线清理。本归档不重复
复制 CT-DUET、原始 DUET 完整控制或 7 月阶段数据；它们分别保存在：

```text
archive/sfda_conflict_ct_duet_v1_2026-08-01/
archive/sfda_conflict_visda_full_duet_control_2026-07-24/
archive/sfda_conflict_results_summary_2026-07-19/
archive/sfda_conflict_visda_proxy_loss_audit_2026-07-23/
archive/sfda_conflict_visda_structural_ablation_2026-07-24/
```

本次归档是非破坏性的：复制证据并保存代码恢复路径，不删除 Downloads
原文件，也不从当前主线删除审计工具。若后续需要精简主线，应另做一次明确的
清理提交。

## 2. 锁定基线和实验契约

```text
数据集：VisDA-C，Synthetic -> Real
指标：mean per-class accuracy
seed：2020
原始 DUET full final：91.50
原始 DUET oracle peak：91.52
matched proxy25 control final：87.93
```

所有目标标签只能在信号锁定后用于 `oracle diagnostic`。本轮没有将目标标签
用于训练规则、阈值、样本选择或模型选择。除已明确记录的 proxy25 候选外，
绝大多数工作是 CPU/offline 审计；没有启动新的完整 VisDA 训练。

## 3. 总体证据结论

1. task/CLIP top-2 union 对冲突样本具有较高 oracle 覆盖率，但“候选集合包含
   真值”不等于存在可靠的无标签裁决器。
2. 固定选 task、固定选 CLIP、置信度、边界距离、空间因果、属性提示、邻居、
   GMM、时间变化、原型运输、patch-to-CLS 等信号均未稳定超过 matched control。
3. 离线梯度方向改善经常不能转化为参数梯度或端到端收益；PCGrad 是最清楚的
   例子，兼容性 preflight 通过但 proxy25 最终与控制完全相同。
4. 多个候选重复出现 car/truck 或 car/person/truck 收益交换。任何依赖 oracle
   类别结果决定启用范围的规则均不合法。
5. patch-to-CLS risk-control selector 在离线 oracle 诊断中能够找到 task rescue，
   但 KL suppression、pair neutralization 和 temporal persistence 都不能把该信息
   安全转化为训练或持续推理收益，因此整个 patch 分支关闭。
6. 本轮最好的 completed proxy25 数值提升只有 `+0.04 pp`（CLIP confidence
   delay，87.97 vs 87.93），低于预声明的 `+0.20 pp` 门槛；没有任何候选获准
   进入 full VisDA。

## 4. 关键 proxy25 结果

| 候选 | Control | Candidate | Delta | 决策 |
|---|---:|---:|---:|---|
| Attribute-reliability KL | 87.93 | 87.94 | +0.01 | REJECT |
| Unresolved-memory support-conditioned CLIP | 87.93 | 87.94 | +0.01 | REJECT |
| CLIP-confidence admission delay | 87.93 | 87.97 | +0.04 | REJECT |
| Compatibility-controlled PCGrad | 87.93 | 87.93 | 0.00 | REJECT |

另有 first-cycle support-conditioned CLIP proxy 的部分终端记录，但最终 gate
文件已不在本机，因此本归档不从不完整日志重建或猜测其最终判定。

完整实验索引见 `EXPERIMENT_INDEX.csv`。

## 5. 最终 patch 分支证据

最后一个候选只在 613 个锁定冲突样本上，把 CLIP soft target 中 task/CLIP
两个候选的概率均分，同时保持候选对总质量和其他类别概率不变。其实现契约
通过，但相对原始 DUET：

```text
feature first-order delta = -0.000245076
95% CI = [-0.000326473, -0.000159547]
helpful-gradient retention = 87.005%
class-macro feature delta = -0.000551528
person delta = -0.002004477
truck delta = -0.001148471
decision = REJECT
```

这证明简单减弱或中和 CLIP 对 task/CLIP 候选对的偏好不是安全干预。离线
selector 的平均 task-rescue 收益主要由 car/plant 样本贡献，不能用目标真实
类别制定 class-specific 例外。

## 6. 目录内容

```text
README_CN.md
EXPERIMENT_INDEX.csv
MISSING_ARTIFACTS.md
SHA256SUMS
code/
  COMMITS.txt
  CHANGED_FILES.txt
  post_ct_cumulative_implementation.patch
data/pair_neutralization/
  visda_patch_cls_pair_neutralization_*.{csv,json,md,npz}
raw_terminal_logs/
  attribute_reliability_proxy25.txt
  candidate_set_gradient_audit.txt
  clip_confidence_delay_proxy25.txt
  feature_gravity_audit.txt
  pcgrad_compatibility_proxy25.txt
  pcgrad_parameter_audit.txt
  spatial_causal_audit.txt
  support_conditioned_clip_audit.txt
  support_conditioned_clip_proxy25_partial.txt
```

`SHA256SUMS` 覆盖除其自身以外的全部归档文件。NPZ 是锁定的 label-free
张量；CSV/JSON 中含目标标签的文件均在名称或字段中明确标记
`oracle_diagnostic`。

## 7. 代码恢复

查看完整提交链：

```bash
git log --reverse 748e716..f661d90
```

在 `748e716` 对应代码树上恢复累计实现：

```bash
git apply archive/sfda_conflict_visda_audit_campaign_2026-08-03/code/post_ct_cumulative_implementation.patch
```

归档标签：

```text
archive/visda-conflict-audits-20260803
```

该标签指向实验实现终点 `f661d90`，而归档目录本身由后续 archive commit
保存。

## 8. 后续限制

以下方向不应在没有新独立信息的情况下重试：

- task/CLIP 概率的 arithmetic/RMS/固定加权重混合；
- confidence/boundary/neighbor/GMM 对两个 top-1 候选的逐样本硬路由；
- top-2 partial-label/set-mass 损失；
- 仅抑制、裁剪或重排原始 DUET CLIP KL；
- 对 agreement 做 confidence delay、连续权重、shared runner-up 或 revocation；
- raw/compatibility PCGrad；
- source-prototype/OT/VSFOT 的孤立组件移植；
- patch-to-CLS selector 后接 KL suppression、pair neutralization 或旧样本记忆。

新的 GPU 实验必须先提供一种未被上述审计覆盖的独立无标签信息，并在离线
门槛中同时证明：相对 matched control 有净收益、CI 下界为正、car/person/
truck 不发生有害交换、且规则不读取目标真实标签。
