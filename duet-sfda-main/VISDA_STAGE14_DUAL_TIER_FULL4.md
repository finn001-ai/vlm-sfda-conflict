# Stage14 VisDA 双层时序监督实验

本实验只验证一个问题：Stage14 从 Cycle 2 开始退化，是否主要因为二值
`stable` memory 把大量“当前一致、但只出现一个 cycle”的 Pending 样本从
hard CE 中全部删除。

不加入 DINO、其他视觉编码器或困难类别专用模块。Stage14 的 target head、
GTR、CE/consistency/KL 系数全部保持不变。

## 新 memory 的固定规则

Cycle 1 保持原 Stage14 预热，所有当前一致样本的 CE 权重为 1。从 Cycle 2
开始：

| 状态 | 条件 | hard CE 权重 |
| --- | --- | --- |
| Stable | 当前 source/CLIP 一致，且同一标签连续至少 2 cycles | `1.0` |
| Pending | 当前一致，但尚未连续 2 cycles | `0.5 * mix_confidence` |
| Conflict | 当前 source/CLIP 不一致 | `0` |

伪标签始终使用当前 cycle 的融合预测。该模式不会像 `monotonic` 一样让后来
已冲突的历史样本继续参加 hard CE。

## 为什么运行两组

已有完整数据因果实验表明，固定 `both_prior` 的净效应很小，但它会改变类别
分配。因此不能只跑一个 calibration 条件，否则无法判断 memory 改进是否依赖
prior。固定运行：

- `both_prior + dual_tier`
- `none + dual_tier`

并与已经完成的 DUET、`stable` 和 `monotonic` 五组日志自动匹配。所有实验均
使用完整 55,388 个 VisDA target 样本、seed 2020、4 cycles。

## 云端运行

先拉取最新提交，然后在仓库根目录执行：

```bash
git pull
bash tools/run_visda_stage14_dual_tier_full4.sh
```

脚本不会重跑已经完成的五组 reference；如果 reference 缺失，会明确要求先
运行：

```bash
bash tools/run_visda_stage14_prior_memory_full4.sh
```

单 GPU 环境不需要设置 `GPU_ID`，脚本也不会覆盖字符串类型的配置项。

## 输出与判定

输出目录：

```text
output/uda/VISDA-C/stage14_dual_tier_full4_seed2020/
```

- `dual_tier_summary.json`：最终精度、相对 stable/monotonic/DUET 的效应、
  每 cycle 的 Stable/Pending/Conflict 数量和有效监督权重；
- `per_class_dual_tier.csv`：逐类别变化，检查是否仍是类别间补偿；
- 每个新实验的 `temporal_diagnostics/*.npz`：新增 `memory_weight`、
  `current_mask`、`stable_mask`、`pending_mask` 和 `conflict_mask`。

只有最终 checkpoint 超过 DUET 至少 0.10 个百分点，脚本才建议进入 8-cycle
确认。脚本本身不会自动启动 8-cycle、seed sweep 或 Office-Home。
