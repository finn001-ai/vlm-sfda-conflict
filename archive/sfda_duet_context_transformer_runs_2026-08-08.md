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
| Pairwise Comparator (gate=0.4) | comparator | 0.8 | [1,2,3] | gate 0.40, weak 关闭 | **73.29%** | 08-08 13:50 |
| Comparator 只在 cycle 2 | comparator | 0.8 | [1] | gate 0.20 | **73.20%** | 08-08 14:07 |
| Comparator 全轮 hard（复跑） | comparator | 0.8 | [1,2,3] | gate 0.20 | **73.29%** | 08-08 14:16 |
| Comparator 全轮 soft-only | comparator | 0.8 | [1,2,3] | gate 0.20, SOFT_ONLY=True | **73.06%** | 08-08 14:25 |

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

## Run 4：Pairwise Comparator，COMPARATOR_GATE=0.4（08-08 13:50）

与 Run 3 完全同配置，仅 gate 0.20 → 0.40。

### cycle 2

- strict_conflicts=1356；resolved=172（12.68%，gate 0.2 时是 738）；abstain=1184
- resolved_acc=**52.91%**（gate 0.2 时 39.70%）；support_task 2 / support_clip 170
- valid pseudo labels：3341/4365，78.63%

### cycle 3

- resolved=563（68.41%）；resolved_acc=35.52%；abstain=260
- support_task=149（51.68%），support_clip=414（29.71%）
- context_margin=0.6961（远超 gate，几乎都过）

### cycle 4

- resolved=483（77.65%）；resolved_acc=37.47%；abstain=139
- support_task=185（48.65%），support_clip=298（30.54%）
- context_margin=0.7985
- 最终 Task：**73.29%**

### 关键发现

- gate 0.2→0.4 只显著影响 cycle 2（738→172，精度 39.7%→52.9%）；
  cycle 3/4 的 margin 已到 0.70/0.80，0.4 门槛形同虚设，仍 resolve 68%/78%，
  精度只有 35~37%。
- 结论：**abstain gate 不是杠杆**。comparator 的置信度严重失准——它对自己的
  错误决定同样自信，提高 gate 无法筛掉错误。
- 新信号：**cycle 2 的决定精度（52.9%）明显高于 cycle 3/4（35~37%）**，
  而置信度反而一路膨胀（0.73→0.85→0.90）。说明跨轮继续训练导致
  over-confidence / 过拟合 synthetic 模式，精度被 clip 侧拖垮。
  值得试：只训 cycle 2、之后冻结 comparator（或大幅减少后续训练步数）。

## Run 5/6/7：三组对照（08-08 14:07 / 14:16 / 14:25）

### Run 5：ACTIVE_CYCLES=[1]（comparator 只在 cycle 2 干预）

- cycle 2：resolved=722（52.8%），resolved_acc=37.67%；
  cycles 3/4 纯 FCP
- 最终 Task：**73.20%**

### Run 6：ACTIVE_CYCLES=[1,2,3] hard admission（gate 0.2 复跑）

- cycle 2/3/4：resolved=763/642/524；resolved_acc=40.1%/35.7%/35.9%
- 最终 Task：**73.29%**

### Run 7：ACTIVE_CYCLES=[1,2,3] SOFT_ONLY_ADMISSION=True

- 每轮 resolved 只做 KL soft target（699/717/621 个），hard_admission=0
- 最终 Task：**73.06%**

### 三组结论（重要）

- hard(73.29) ≈ 只在 cycle 2(73.20) ≈ soft-only(73.06)，差异 < 0.25 点，
  基本不可区分。**admission 机制（hard / soft / 单轮）不是杠杆**。
- 73.x 水平基本来自 DUET-FCP + CLIP/Task 适配本身，comparator 的仲裁
  信号目前没有可测量的训练价值。
- distribution 日志（Run 5/7）证实 synthetic/real 错配：
  - synthetic：p_task_A=0.67, task_margin=0.59, task_entropy=1.29
  - real-conflict：p_task_A=0.37, task_margin=0.22, task_entropy=2.39
  - synthetic 是“一边明显坏、一边明显好”，real 是“两边都模糊”，
    与 resolved_acc 卡在 35~40% 完全对应。
- support_task 仍是唯一有真信号的子集（46~60% vs support_clip 26~38%）。

### 下一步（按优先级）

1. 停止调 admission；先解决 comparator 决策质量：
   - 加“两边都错”的 synthetic 对（Task 翻 B、CLIP 翻 C，目标 abstain）；
   - support_clip 决定基本是噪声，考虑 asymmetric gate 或只采信
     support_task 高 margin 子集。
2. 跑 cosine_knn / prototype 对照：确认 73.x 里有没有任何成分来自
   “anchor 相似度”而非 comparator 本身。
3. 对照纯 FCP baseline（同 POWER 0.8 / CLS_PAR 0.4）确认 73.x 是否是
   基线水平；用户的 74 参考需要明确口径后对齐。
