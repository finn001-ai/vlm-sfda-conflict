# VisDA Stage14 Dual-tier Pending 全量实验

归档日期：2026-07-27

本目录保存用户从云端回传的完整 VisDA、seed 2020、4 cycles Dual-tier
实验汇总。对应代码存档为 Git tag
`archive/stage14-dual-tier-20260727`。

## 文件

- `dual_tier_summary.json`：DUET、Stable、Monotonic 与两组 Dual-tier 的最终
  精度、伪标签统计和逐 cycle memory 动态。
- `per_class_dual_tier.csv`：逐类别比较。
- `SHA256SUMS`：原始回传文件校验值。

## 固定规则

- Stable：当前一致且同标签连续至少 2 cycles，hard CE 权重 1。
- Pending：当前一致但尚未稳定，hard CE 权重 `0.5 * mix_conf`。
- Conflict：当前 source/CLIP 不一致，hard CE 权重 0。

## 主要结果

| 方法 | 最终准确率 |
| --- | ---: |
| DUET | 90.32 |
| both_prior + Stable | 89.98 |
| none + Stable | 89.99 |
| both_prior + Monotonic | 90.07 |
| none + Monotonic | 90.17 |
| both_prior + Dual-tier | 90.02 |
| none + Dual-tier | 90.10 |

最好的 Dual-tier 为 `none + dual_tier`，相对 Stable 提高 0.11，但仍低于
Monotonic 0.07、低于 DUET 0.22。

Cycle 2 中 `none + dual_tier` 有 22,053 个 Pending，平均权重 0.4171；
48,020 个当前一致样本的有效监督权重只有 35,165.25。结果说明恢复 Pending
有效，但当前权重仍偏保守。

## 结论边界

Dual-tier 只恢复 Pending，没有利用 Conflict。自动 JSON 中的
`dual_tier_interacts_with_prior` 是阈值判定结果；不能据此声称 Conflict
问题已经解决，也不能声称 Dual-tier 已达到最优。
