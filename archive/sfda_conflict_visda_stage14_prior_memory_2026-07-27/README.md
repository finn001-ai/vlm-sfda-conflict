# VisDA Stage14 prior × memory 全量因果实验

归档日期：2026-07-27

本目录保存用户从云端回传的完整数据、seed 2020、4 cycles 因果实验结果。
实验比较 `CALIB_MODE ∈ {both_prior, none}` 与
`PL_MEMORY ∈ {stable, monotonic}`，并包含 DUET 复现门禁。

## 文件

- `reproduction_gate.json`：DUET/Stage14 复现有效性检查。
- `factorial_summary.json`：2×2 因果实验汇总。
- `per_class_factorial.csv`：逐类别结果。
- `raw/*.txt`：四个 Stage14 组合的原始日志。
- `SHA256SUMS`：上述原始文件校验值。

## 结论边界

自动判定为 `cause_supported_but_duet_not_recovered`。`none + monotonic`
最终为 90.17，仍低于 DUET 90.32。该实验支持稳定筛选损害覆盖率，但不能证明
剩余差距完全由 memory 或 prior 单独造成。
