# VisDA-C Top-K 冲突可利用率与 swap 选边方案分析（2026-08-04）

状态：**分析结论归档。** 数据来自 2026-08-03/04 的 Top-K conflict probe 运行
（TV，VISDA-C train→validation，seed 2020，评估集 55,388 张，8 个 cycle）。
本归档保存结论、复现脚本与统计汇总；原始样本级 CSV（约 40MB）不重复复制，
仍位于 `/Users/stranger/Downloads/task_TV_seed_2020/`。

## 1. 复现

```bash
python code/analyze_topk_swap_selection.py \
  --data /Users/stranger/Downloads/task_TV_seed_2020 --out data
```

输出 `data/*.csv`（见第 4 节文件清单）。

## 2. 核心结论

### 2.1 Top-2 作为候选集：可利用率高，但只适合候选集监督

- 冲突样本从 28,223（cycle 0，占 51.0%）缩到 3,414（cycle 7，占 6.2%）。
- top2 union（{task top1, task top2, clip top1, clip top2}）覆盖真值的比例在
  8 个 cycle 稳定在 91.4–93.5%，比 top1 union 高 7–11pp。
- top1 全错时，top2 条件恢复率 62.4%（cycle 0）→ 47.4%（cycle 7）。
- 恢复来源（cycle 0，共 3,054）：clip top2 单独 2,077（68.0%）、task top2 单独
  466（15.3%）、两者同中 511（16.7%）。正确候选平均分约 0.19–0.22，属低置信
  候选 → 结论：top2 适合做候选集监督，不适合做硬标签；早期 top2 价值近似
  "相信 CLIP 的第二高分"，后期 task 反超且整体恢复率下降。
- **注意**：cycle 0 冲突数为 28,223，与仓库契约 28,255（
  `src/utils/topk_conflict_probe.py` 的 `VISDA_CYCLE0_REGRESSION`）不一致（-32），
  全套审计工具以 28,255 为契约；该偏差未定位，跨审计可比性存疑。

### 2.2 swap（纯双向指认）结构

swap = bidirectional_cross_support：task top1 = clip top2 = A、clip top1 =
task top2 = B。逐 cycle 数量与占比见 `data/per_cycle_swap_stats.csv`：

- 数量 1,949（cycle 0）→ 峰值 3,256（cycle 1）→ 1,973（cycle 7）；占冲突比例
  6.9% → 57.8%；占全部 55,388 张仅 3.5–5.9%。
- 8 cycle 合计 19,398 个 swap 观测，占全部 443,104 个观测的 4.4%。
- swap 的 top2 union 只有 2 个类、无新增候选，条件恢复率恒为 0 → 候选集监督
  对 swap 结构性失效，需要专门的选边规则。
- 主导类别对：car↔truck 是每个 cycle 的第一大 swap（509–805 对），
  motorcycle↔bicycle 第二；方向信息强（如 cycle 7 中 task 说 car 时 task 对
  58%，task 说 truck 时 CLIP 对 64%），详见 `data/pair_orientation.csv`。

### 2.3 swap 选边方案的演进与结论

**基线（全部 19,398 个 swap）：** 加权投票（pA+qA vs pB+qB）56.5%、always
CLIP 51.3%、always task 43.0%。50% 票数规则在 cycle 0 仅 63.9%，低于直接信
CLIP（77.0%）；根源是 task softmax 峰值化（cycle 0 中 43.6% 样本 pA≥0.9），
跨模型分数直接相加不可比。

**新方案（label-free）：**

1. cycle 0 特例：直接采用 CLIP top1（精度 77.0%，全流程任何规则的最高点）。
2. cycle ≥ 1：模型内偏好比 + 决策强度门槛 D。
   - eA = pA·qA，eB = pB·qB（pA/pB 为 task top1/top2 概率，qA/qB 为 clip
     top2/top1 分数）；
   - log(eA) − log(eB) ≥ D → 标签 = A；log(eB) − log(eA) ≥ D → 标签 = B；
     否则 abstain（不产生伪标签）；
   - D 为可调超参数（e^D 即"双方交叉证据乘积"的倍数优势）。

全 8 cycle 汇总（cycle 0 = CLIP，cycle 1–7 = 偏好比+门槛），
`data/selection_curve.csv`：

| D≥ | 决策数 | 覆盖 | 正确 | 错误 | 精度 |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 19,398 | 100.0% | 11,279 | 8,119 | 58.1% |
| 0.5 | 16,062 | 82.8% | 9,651 | 6,411 | 60.1% |
| 1.0 | 13,248 | 68.3% | 8,203 | 5,045 | 61.9% |
| 1.5 | 11,013 | 56.8% | 6,997 | 4,016 | 63.5% |
| 2.0 | 9,196 | 47.4% | 6,013 | 3,183 | 65.4% |
| 3.0 | 6,577 | 33.9% | 4,494 | 2,083 | 68.3% |
| 4.0 | 4,854 | 25.0% | 3,459 | 1,395 | 71.3% |

推荐操作点：追求正确标签总数用 D≈0–0.5；追求标签可信度用 D≥2.0（默认）；
最保守 D≥4.0。

**成对方向先验（oracle-informed 上限）：** 用上一 cycle 真值估计
"方向 × cycle" 下更可靠的一方，与主规则一致区精度 67–73%（cycle 1–7），
分歧区只有 36–46%（低于随机，必须放弃），见
`data/prior_agree_disagree.csv`。部署时需 label-free 估计器（源域标定或
伪标签累积），直接沿用上一 cycle 真值属于 oracle 泄漏。

## 3. 已知限制

- 全部统计为 oracle-diagnostic：GT 只用于评估，不进入标签生成。
- 先验一致区 67–73% 用上一 cycle GT 估计，是上限而非可部署精度。
- swap 仅占全量观测 4.4%，对总精度直接贡献有限；价值在于补齐候选集监督的
  盲区，且集中覆盖 car/truck 等 hard class。
- cycle 0 契约偏差（28,223 vs 28,255）未解决。
- 实现提示词（Codex 集成方案）未随本归档保存为独立文件，见主线程对话记录；
  需要时可从对话重建。

## 4. 文件清单

```text
README_CN.md
SHA256SUMS
code/analyze_topk_swap_selection.py
data/per_cycle_swap_stats.csv
data/selection_curve.csv
data/baselines.csv
data/pair_orientation.csv
data/prior_agree_disagree.csv
```
