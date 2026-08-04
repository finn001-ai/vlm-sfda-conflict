# VisDA-C Swap 冲突选边方案最终结论归档（2026-08-04）

状态：**负结果归档。** swap 冲突硬标签选边方案经 3 轮完整 VisDA-C 训练验证，
最终精度全部持平或略低于基线（91.45–91.46 vs 原始 DUET 91.50），确认无正
贡献。实现仍在主线（`duet-sfda-main/`），本归档保存结论、数据与复现路径，
不删除主线代码。

## 1. 背景与方案

Top-k conflict probe 发现一类**纯 swap 冲突**（bidirectional_cross_support）：
task top1 = A、task top2 = B、clip top1 = B、clip top2 = A（A≠B），两个模型
互相指认对方的第一名。普通 DUET 对这些样本不给硬标签（task/CLIP top1 不
一致，进不了 agreement mask）。方案对这些样本补充硬伪标签：

```text
eA = pA * qA，eB = pB * qB   （pA/pB = task top1/top2 概率，qA/qB = clip
                                top2/top1 分数，prior 校准前口径）
cycle 0：直接取 CLIP top1（B），不设门槛。
cycle >= 1：|log(eA) - log(eB)| >= D 才决策，否则 abstain（不进训练损失）。
```

两个改进开关（v2）：

- **方向门槛** `MIN_DIRECTION_ACCURACY`：离线锁定 65 个方向的 cycle-0 CLIP
  精度表，低于阈值的方向直接 abstain（保护 car/truck 等 CLIP 不可靠方向）。
- **前期停用** `LAST_ACTIVE_CYCLE`：后期（cycle 7–8）标签精度仅 60–65%，
  且大部分是重复样本，从配置的 cycle 起不再产生新标签。

## 2. 实验与结果

数据集：VisDA-C Synthetic->Real，TV，seed 2020，8 cycle，评估集 55,388 张。
基线为 2026-07-24 归档的原始 DUET full（`sfda_conflict_visda_full_duet_control_2026-07-24`）。

### 2.1 三版完整 run 的最终精度

| Run | 配置 | Final (mean per-class) | 相对基线 |
|---|---:|---:|---:|
| 基线 | 原始 DUET（plmatch，无 swap） | **91.50** | — |
| A | D=4.0 全集（8 cycle 全开） | 91.46 | −0.04 |
| B | D=2.0 + 方向≥0.8 + 前 6 cycle | 91.45 | −0.05 |

三版全部落在同一水平，没有一版超过基线。

### 2.2 逐 cycle 训练精度（cycle 末次 checkpoint）

| Cycle | 基线 | Run A | Run B |
|---:|---:|---:|---:|
| 1 | 85.76 | 86.25 | 86.22 |
| 2 | 88.84 | 88.93 | 88.94 |
| 3 | 89.78 | 89.82 | 89.87 |
| 4 | 90.33 | 90.33 | 90.38 |
| 5 | 90.65 | 90.64 | 90.68 |
| 6 | 91.13 | 91.08 | 91.12 |
| 7 | 91.31 | 91.30 | 91.31 |
| 8 | 91.50 | 91.46 | 91.45 |

### 2.3 final 类级精度（12 类顺序：aeroplane…truck）

| 类 | 基线 | Run A | Run B |
|---|---:|---:|---:|
| aeroplane | 98.71 | 98.79 | 98.77 |
| bicycle | 89.38 | 90.04 | 90.10 |
| bus | 88.98 | 89.40 | 89.08 |
| car | **80.61** | **78.65** | **79.09** |
| horse | 98.21 | 98.12 | 98.19 |
| knife | 97.78 | 97.93 | 97.93 |
| motorcycle | 95.93 | 95.89 | 95.79 |
| person | 85.12 | 84.78 | 84.85 |
| plant | 96.00 | 96.04 | 96.07 |
| skateboard | 97.37 | 97.19 | 97.11 |
| train | 95.21 | 95.30 | 95.28 |
| truck | 74.73 | 75.40 | 75.20 |
| **mean** | **91.50** | **91.46** | **91.45** |

### 2.4 swap 决策统计（oracle-diagnostic，GT 只用于评估）

| Cycle | Run A 决策/精度 | Run B 决策/精度 |
|---:|---:|---:|
| 1 | 1,949 / 76.96% | 957 / 89.97% |
| 2 | 240 / 76.25% | 426 / 82.39% |
| 3 | 329 / 75.99% | 485 / 78.76% |
| 4 | 371 / 68.73% | 473 / 69.34% |
| 5 | 436 / 66.06% | 427 / 66.51% |
| 6 | 528 / 68.94% | 483 / 65.63% |
| 7 | 515 / 62.14% | 0（停用） |
| 8 | 461 / 58.79% | 0（停用） |

## 3. 失败机制分析

1. **信号量太小**：swap 只占全部观测的 4.4%；D=2.0+方向过滤后全程只给
   ~3,300 次决策、约 2,600 个独特样本打过标签，占训练信号的 1–2%。期望
   影响本来就只有 ±0.1pp 量级。
2. **错误标签抵消收益**：归档证据上 D=2.0+方向过滤的净正确标签约 +2,116
   （决策层），但错误标签 1,052 个，集中在 car/truck 等 hard class，训练
   中收益被污染抵消。
3. **car/truck 结构性交换**：所有版本 final 都出现 car 掉 1.5–2.0pp、
   truck/bicycle 涨 0.4–0.7pp 的交换。方向过滤把 cycle 0 的 car 相关方向
   abstain 后 car 仍掉 1.5pp，说明 car 的损失不单纯来自 swap 标签本身。
4. **后期标签质量崩塌**：cycle 7–8 精度 58–62%（比二选一瞎猜只高 8–12pp），
   且 883 次决策中真正新增只有 381 个（148 个错误）。后期停用后 final
   仍无改善（91.45），说明后期标签既不是伤害主因也不是收益来源。
5. **与仓库既有结论一致**：本轮结果再次印证"逐样本硬选择不可靠"（
   `sfda_conflict_visda_audit_campaign_2026-08-03` 结论 2），以及"top-2
   候选集适合候选集监督、不适合硬标签"（
   `sfda_conflict_visda_topk_swap_analysis_2026-08-04` 结论 2.1）。

## 4. 已知限制

- 基线 91.50 为原始 DUET（plmatch，无 first-cycle prior）的复现；Run A/B
  为 DUET-FCP（含 prior）+ swap，两者相差一个 FCP 开关。未跑"FCP 无 swap"
  对照，严格归因存在 ±0.05pp 的混杂，但不改变"swap 无正贡献"的结论。
- cycle_000 冲突数 28,223 与仓库契约 28,255 不一致（差 32），未定位，
  全套分析以实际 CSV 为准。
- 方向精度表来自归档 cycle-0 数据（65 个方向），小样本方向统计噪声大但
  样本极少。
- 三版均为单 seed（2020）单 run；±0.05pp 差异在 run-to-run 噪声范围内。

## 5. 代码与复现

实现仍在主线（未删除，供后续候选集监督方案参考）：

```text
duet-sfda-main/src/utils/swap_conflict_selection.py
duet-sfda-main/src/methods/oh/duet_first_cycle_prior_swap_selection.py
duet-sfda-main/src/methods/oh/plmatch.py          （train_target/obtain_label 开关）
duet-sfda-main/cfgs/visda/duet_first_cycle_prior_swap_selection.yaml
duet-sfda-main/tools/run_visda_duet_first_cycle_prior_swap_selection.sh
duet-sfda-main/tests/test_swap_conflict_selection.py
duet-sfda-main/SWAP_CONFLICT_SELECTION.md
```

复现归档证据（伪标签决策层汇总）：

```bash
python code/analyze_swap_selection_archive.py \
  --data /Users/stranger/Downloads/task_TV_seed_2020 --out data
```

输出 `data/decision_curve.csv`（D 门槛 × 方向过滤 × 前 6 cycle 的决策/精度/
净正确），与 README 第 2.4 节及主线回归测试口径一致。

## 6. 文件清单

```text
README_CN.md
SHA256SUMS
code/swap_conflict_selection.py           # 主线规则实现副本（恢复路径）
code/analyze_swap_selection_archive.py    # 归档证据复现脚本
data/run_comparison.csv                   # 三版 run 逐 cycle 精度 + final 类级
data/swap_decision_logs.csv               # 两版 run 的逐 cycle swap 决策
data/decision_curve.csv                   # 规则配置的决策层汇总（脚本生成）
```
