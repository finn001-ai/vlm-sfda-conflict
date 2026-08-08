# DUET-FCP + Context Transformer / Comparator 实验归档

任务：Office-Home AC（Art→Clipart），seed 2020，resnet50 + CLIP ViT-B/32，
4 cycles，4365 个 target 样本。所有 accuracy 均为 eval-only（
ground_truth_affects_training=False）。

## 汇总（AC 最终 Task 精度）

| 版本 | refiner | POWER | ACTIVE_CYCLES | 关键阈值 | 最终 Task 精度 | 日期 |
|---|---|---|---|---|---|---|
| 保守 Context Transformer | transformer | 0.5 | [1,2,3] | accept 0.75/0.20, 训练 100 步 | 72.21% | 08-07 |
| 宽松 Context Transformer | transformer | 0.8 | [1,2,3] | accept 0.55/0.10, anchor 0.7/0.8, 训练 200 步 | 72.12% | 08-08 09:40 |
| Pairwise Comparator | comparator | 0.8 | [1,2,3] | gate 0.20, weak 关闭 | **73.13%** | 08-08 12:06 |

## Run 1：保守 Context Transformer（08-07 10:19）

- `USE_WEAK_AGREEMENT=true`，anchor conf 0.90/0.90，训练 100 步/cycle
- cycle 2/3/4：resolved = 3 / 22 / 59；resolved_acc = 33% / 68% / 61%
- weak_defer_rate = 99.6% / 97.2% / 91.9%（几乎全拒）
- 最终 72.21%

## Run 2：宽松 Context Transformer（08-08 09:40）

- `USE_WEAK_AGREEMENT=true`，anchor conf 0.7/0.8，训练 200 步/cycle
- cycle 2/3/4：resolved = 119 / 222 / 289；resolved_acc = 46% / 46% / 41%
- third_class 出现（11 个 / 7 个），正确率 27% / 0%（纯噪声）
- 最终 72.12%

## Run 3：Pairwise Comparator（08-08 12:06，当前主线）

配置：`REFINER_TYPE=comparator`，`USE_WEAK_AGREEMENT=false`，anchor conf
0.7/0.8，`COMPARATOR_GATE=0.20`，`RUNNER_UP_FALLBACK=false`，训练 200 步/cycle，
synthetic conflict 只来自 strong augmentation 真实 flip。

### cycle 1（纯 FCP，无 comparator）

- valid pseudo labels：1916/4365，精度 87.00%
- 最终 Task：65.41%

### cycle 2

- synthetic conflicts：task_side=165, clip_side=195, total=360
- strict_conflicts=1364；resolved=738（54.11%）；abstain=626
- support_task=49（acc 57.14%），support_clip=689（acc 38.46%）
- resolved_acc=39.70%（293 对 / 445 错）；intervention_error_rate=60.30%
- valid pseudo labels：3844/4365，70.55%

### cycle 3

- synthetic conflicts：task_side=285, clip_side=365, total=650
- strict_conflicts=748；resolved=613（81.95%）；abstain=135
- support_task=205（46.34%），support_clip=408（27.94%）
- resolved_acc=34.09%（209 对 / 404 错）
- valid pseudo labels：4299/4365，71.60%

### cycle 4

- synthetic conflicts：task_side=253, clip_side=476, total=729
- strict_conflicts=596；resolved=528（88.59%）；abstain=68
- support_task=240（47.08%），support_clip=288（28.82%）
- resolved_acc=37.12%（196 对 / 332 错）
- valid pseudo labels：4357/4365（99.8% 覆盖），72.30%
- 最终 Task：**73.13%**

### 三轮汇总

- 总 resolved = 1879，其中正确 698、错误 1181（63% 是错标签）
- 主要病理：comparator 倾向信 CLIP（support_clip 远多于 support_task），
  而 CLIP 在真实冲突上只有 ~26~33% 正确 → support_clip 决定基本是噪声；
  support_task 子集（46~57%）反而有真信号
- label_mask 覆盖到 99.8% 后，进一步收益只能来自标签质量而非覆盖

## 待办 / 实验计划

1. `COMPARATOR_GATE` 0.20 → 0.45~0.50（precision 换 coverage）
2. 不对称门槛：trust_clip 要求 margin ≥ 0.5，trust_task 要求 margin ≥ 0.3
3. soft-only 消融：comparator 只改 KL soft target，不进 label_mask
   （`DUET_CONTEXT.SOFT_ONLY_ADMISSION=true`）
4. 补“两边都错”的 synthetic 样本（Task 翻 B、CLIP 翻 C，目标 abstain）
5. 对比 synthetic vs real conflict 的证据分布（难度/confidence/margin）
